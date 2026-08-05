from re import search
import ollama
from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat import Chat
from src.agent.memory import Memory
from src.tools.search import is_connected, SearchAgent
from src.tools.file_reader import FileReader

from src.config import (
    MODEL,
    MEM_MANUAL_PROMPT,
    CHROMADB_DIR,
    DROPBOX_DIR,
    MODEL_MAX_TOKENS,
    MAX_RESULTS,
    AUTO_READ_DROPBOX_TOKENS,
    AUTO_WEBSEARCH_TOKENS
)

def _last_token_usage(messages: list[dict]) -> tuple[int, int]:
    """Returns prompt_tokens and output_tokens from the latest assistant message."""
    for msg in reversed(messages):

        # Only the 'assistant' message have token counts.
        if msg["role"] == "assistant":
            return msg.get("prompt_tokens", 0), msg.get("output_tokens", 0)

    return 0, 0 # no previous assistant message yet (first turn)

def _detect_cmd(prompt: str) -> tuple[str | None, str]:
    """Extracts shortcut if detected."""
    question_trimmed = prompt.strip()

    if question_trimmed.startswith(f"/"):
        parts           = question_trimmed.split(" ", 1)
        cmd             = parts[0]
        cleaned_text    = parts[1].strip() if len(parts) > 1 else ""
        return cmd, cleaned_text
    return None, prompt


class Agent:
    def __init__(
        self,
        model: str = MODEL,
        session: str | None = None,
        project: str | None = None
    ):
        self._validate_model(model)

        if MODEL_MAX_TOKENS:
            self.tokens = Tokens(model, MODEL_MAX_TOKENS)
        else:
            self.tokens = Tokens(model)

        self.model          = model  # model is specified in the config file
        self.chat           = Chat(session=session)
        self.memory         = Memory(chat=self.chat, project=project)
        self.search_agent   = SearchAgent()
        self.project        = project
        self.file_reader    = FileReader(session=session)

    @property
    def max_tokens(self) -> int:
        """Dynamically fetches the current token ceiling from 'self.token'."""
        return self.tokens.max_tokens

    # ===================================
    # Model validation
    # ===================================

    def _validate_model(self, model: str):
        """Check if model is installed via Ollama."""
        try:
            # Fetch all downloaded models
            local_models = [m['model'] for m in ollama.list().get('models', [])]

            # Check match
            if model not in local_models:
                raise ValueError(
                    f"Error: Unknown or unavailable model: '{model}'."
                    f"Run 'ollama pull {model}' to install model."
                )

        except Exception as e:
            # If ollama is down
            if isinstance(e, ValueError):
                raise e
            raise RuntimeError(f"Error: Could not connect to local Ollama service: {e}")

    # ===================================
    # Token management
    # ===================================

    def _manage_token_budget(self, prompt: str):
        """
        Reserves extra tokens for model response. If exceeds
        maximum tokens, the model summarise previous messages
        to free up token space.
        """
        reserve                     = 1000 # (tokens)
        current_history_tokens      = self.tokens.count_history_tokens(self.chat.to_llm())
        estimate_next               = current_history_tokens + (len(prompt) // 4)

        # If not enough tokens for the next model output
        if (self.tokens.max_tokens - estimate_next - reserve) < 0:
            print("Current token exceeds threshold, compressing session...")
            self.chat.compression(self.model)
            print("Compression complete, continue session.")

    # ===================================
    # Commands
    # ===================================

    def _cmd_forget(self, prompt: str):
        """Find exact match in memory from user prompt and remove said entry."""
        if not prompt:
            return "Please specify what to forget."

        match_data = self.memory.get_exact_match(prompt)
        if not match_data:
            return f"Error: No matching memory found."

        target_id, matched_content = match_data

        print(f"Warning: [ChromaDB] Initializing delte sequence for: '{matched_content}...'")
        choice = input("Press [Enter] to confirm deletion or type [c] to cancel:")

        if choice == "c":
            return "Deletion cancelled."

        self.memory.delete_from_db([target_id])
        return "Entry deleted."

    def _cmd_memorise(self, prompt: str):
        """Extract key info from user prompt and save entries to memory."""
        if not prompt:
            return "Please specify what to memorize."

        print(f"Extracting content from user's input...")
        created_ids, prompt_tokens, output_tokens = self.memory.extract_to_db(
            context=self.chat.to_llm(),
            prompt=prompt,
            source="explicit",
            manual=True
        )

        # print exactly the content that is stored
        saved_entries = self.memory.get_entries_by_ids(created_ids)

        if saved_entries:
            print("Content saved to database:")
            for entry in saved_entries:
                print(f"    - [{entry['category']}] {entry['content']}")
        else:
            return "Error: No data was extracted by the model."

        choice = input("Press [Enter] to continue or type [u] to undo:")

        if choice == "u":
            self.memory.delete_from_db(created_ids)
            return "Entry deleted."

        # Only save messages if continue
        confirmation = "Memory saved to database."
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external"
        )
        self.chat.append_assistant_message_with_metadata(
            content=confirmation,
            state="external",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        return confirmation

    def _cmd_recall(self, prompt: str):
        """Retrieve and print relevant entries accourding to user prompt."""
        if not prompt:
            return "Please specify what to recall."

        recalled_entries = self.memory.retrieve_relevant_entry(prompt, limit=4)

        if recalled_entries:
            recalled = "Found matching entries in long-term memory:\n" + "\n".join(
                f"- [{entry['category']}] {entry['content']}" for entry in recalled_entries
            )
        else:
            return f"Error: No matching memories found."

        # Save user's question
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external"
        )

        # Add recalled memory to chat (Embedding model: no tokens counts)
        self.chat.append_assistant_message_with_metadata(
            content=recalled,
            state="external"
        )

        return recalled

    def _cmd_search(self, prompt: str) -> str | None:
        """Generates, search and answer query based on user prompt."""
        if not prompt:
            return "Please specify what to search."

        print("Generating query...")
        query = self.search_agent.generates_query(
            context=self.chat.to_llm(),
            prompt=prompt
        )

        response, query_with_urls, prompt_tokens, output_tokens, search = self.search_agent.web(
            query=query,
            context=self.chat.to_llm(),
            prompt=prompt,
            max_results=MAX_RESULTS
        )

        # Save user messages
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external"
        )

        # Save assistant messages
        self.chat.append_assistant_message_with_metadata(
            content=response,
            state="external",
            query_with_urls=query_with_urls,
            search=True,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        return None

    # ===================================
    # Extra context
    # ===================================

    def _file_context(
            self,
            context: list[dict],
            prompt: str
        ) -> tuple[list[str], list[str]] | None:
        """
        Model decides to read which previously uploaded files.

        Steps:
        1. Requires model as controller to return 'True' or 'False' to read file.
        2. If 'True', model return list of filenames available in that session.
        3. Model choose which file(s) to read, return filename(s).
        """
        output = self.file_reader.require_file_or_not(context, prompt)

        # Model decide to read files
        if output == True:
            found_files, not_found_files = self.file_reader.get_filenames(context, prompt)

        # Model decide not to read files
        else:
            return

        return found_files, not_found_files

    def _add_extra_context(
        self,
        messages: list[dict],
        memory_entries: str,

        file_contents: str,
        dropbox_files: list[str],
        read_dropbox: bool,

        search_results: str,
        query_with_urls: list[dict[str, list[str]]],
        search: bool,

        list_of_files: list[str] | None,

        prompt: str
    ):
        """If memory entries are given model reads the list of files and response, else skip."""
        combined_prompt = f"{memory_entries}\n{file_contents}\n{search_results}\nUser input:\n{prompt}"

        answer, prompt_tokens, output_tokens = self.file_reader.read_files_with_context_prompt(
            context=messages,
            context_prompt=combined_prompt,
            list_of_files=list_of_files,
            prompt=combined_prompt
        )

        # Save user message
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external",
            attachments=list_of_files
        )

        # Save assistant message
        self.chat.append_assistant_message_with_metadata(
            content=answer,
            state="external", 
            attachments=dropbox_files,
            query_with_urls=query_with_urls,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

    # ===================================
    # Execution
    # ===================================

    def ask(
        self,
        prompt: str,
        auto_memory_retrieve: bool = True,
        auto_web_search: bool = True,
        auto_read_dropbox: bool = True,
        extra_context: bool = False,
        list_of_files: list[str] | None = None
    ) -> None | str:
        """
        Model decide what memories to read.

        Manage tokens, compress session if needed.

        Note:
        - Only the user question and LLM response will be
          stored into chat history.
        - Add an option to allow user to toggle memory retrieve.

        Process:
        1. Model decides from {prompt} --> {memory_entries}
        2. Model decides from {memory_entries} + {prompt} --> {file_content} from dropbox
        3. Model decides from {memory_entries} + {file_content} + {prompt} --> {search_results}

        User entry:

        Relevant memory entries:
        {memory_entries}
        ========================================
        Context from file:
        Filename = {filename}
        {file_content}
        ========================================
        Search results:
        {search_results}
        ========================================
        Here are the required context:
        Context from file:
        Filename = {filename}
        {file_content}
        ========================================
        User input:
        {prompt}
        """
        self._manage_token_budget(prompt)

        cmd, user_prompt = _detect_cmd(prompt)

        if cmd:
            # Only use 'user_prompt' as 'prompt' here

            if cmd == "/forget":
                # User's question were not saved
                response = self._cmd_forget(user_prompt)
                if response:
                    print(response)

                # ==============
                # // End here //
                # ==============
                return

            if cmd == "/memorise":
                # User's question were saved
                response = self._cmd_memorise(user_prompt)
                if response:
                    print(response)

                # ==============
                # // End here //
                # ==============
                return

            if cmd == "/recall":
                # User's question and agent's response were saved
                response = self._cmd_recall(user_prompt)
                if response:
                    print(response)

                # ==============
                # // End here //
                # ==============
                return

            if cmd == "/search":
                # User's question were saved
                response = self._cmd_search(user_prompt)
                if response:
                    print(response)

                # ==============
                # // End here //
                # ==============
                return

        messages = self.chat.to_llm()

        memory_entries = self.memory.toggle_auto_add_memory_entry(
            auto_memory_retrieve,
            messages,
            prompt
        )

        file_contents, found_file_paths, read_dropbox = self.file_reader.toggle_auto_read_dropbox(
            self.max_tokens,
            auto_read_dropbox,
            messages,
            prompt
        )

        search_results, query_with_urls, search = self.search_agent.toggle_auto_web_search(
            self.max_tokens,
            auto_web_search,
            messages,
            prompt,
            memory_entries
        )

        # Toggle extra_context (read files)
        if extra_context == True:
            self._add_extra_context(
                messages=messages,
                memory_entries=memory_entries,

                file_contents=file_contents,
                dropbox_files=found_file_paths,
                read_dropbox=read_dropbox,

                search_results=search_results,
                query_with_urls=query_with_urls,
                search=search,

                list_of_files=list_of_files,
                prompt=prompt
            )

            # ==============
            # // End here //
            # ==============
            return

        # Combines all context and prompts into one entry and append to messages
        messages.append({
            "role": "user",
            "content": f"{memory_entries}\n{file_contents}\n{search_results}\nUser input:\n{prompt}"
        })

        # Model answer
        answer, prompt_tokens, output_tokens = LLM.model_response(messages, model = self.model)

        # Save messages
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external"
        )
        self.chat.append_assistant_message_with_metadata(
            content=answer,
            state="external",
            attachments=found_file_paths,
            query_with_urls=query_with_urls,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        # Add autosave to memory function (Toggle on/off)

        # ==============
        # // End here //
        # ==============
        return
