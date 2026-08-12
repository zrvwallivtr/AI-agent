from re import search
import ollama
from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat import Chat
from src.agent.memory import Memory
from src.agent.cmd_functions import Command
from src.agent.extra_context import ExtraContext
from src.tools.search import is_connected, SearchAgent
from src.tools.file_reader import FileReader

from src import config


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
        model: str | None = None,
        session: str | None = None,
        project: str | None = None
    ):
        model = config.MODEL if model is None else model

        self._validate_model(model)

        if config.MODEL_MAX_TOKENS:
            self.tokens = Tokens(model, config.MODEL_MAX_TOKENS)
        else:
            self.tokens = Tokens(model)

        self.model          = model  # model is specified in the config file
        self.project        = project
        self.chat           = Chat(session=session)
        self.memory         = Memory(project=project)
        self.file_reader    = FileReader(session=session)
        self.command        = Command(model=model, session=session, project=project)
        self.extra_context  = ExtraContext(model=model, session=session, project=project)
        self.search_agent   = SearchAgent()

    @property
    def get_model_max_tokens(self) -> int:
        """Dynamically fetches the current token ceiling from 'self.token'."""
        return self.tokens.model_max_tokens

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
        if (self.tokens.model_max_tokens - estimate_next - reserve) < 0:
            print("Current token exceeds threshold, compressing session...")
            self.chat.compression(self.model)
            print("Compression complete, continue session.")

    # ===================================
    # Extra context
    # ===================================

    def _add_extra_context(
        self,
        messages: list[dict],
        memory_entries: str,

        file_contents: str,
        dropbox_files: list[Path],
        read_dropbox: bool,

        search_results: str,
        query_with_urls: list[dict[str, list[str]]],
        search: bool,

        file_paths: list[Path] | None,

        prompt: str
    ):
        """If memory entries are given model, reads the list of files and response, else skip."""
        added_file_contents = self.file_reader.read_files_with_context_prompt(
            context=messages,
            file_paths=file_paths,
        )

        # Store file into dropbox
        for path in file_paths:
            content, _ = self.file_reader.load_file_content(path)
            self.file_reader.store_file_in_dropbox(content, path)

        messages.append({
            "role": "user",
            "content": f"{memory_entries}\n{file_contents}\n{search_results}\n{added_file_contents}User input:\n{prompt}"
        })

        answer, prompt_tokens, output_tokens = LLM.model_response(messages, self.model)

        # Save user message
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external",
            attachments=file_paths
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

        # Update dropbox metadata
        print("Updataing dropbox metadata...")
        self.file_reader._add_metadata_and_summary(file_paths)

    # ===================================
    # Execution
    # ===================================

    def ask(
        self,
        prompt: str,
        enable_extra_context: bool = False,
        enable_auto_memory_retrieve: bool = config.ENABLE_AUTO_MEMORY_RETRIEVE,
        enable_auto_memory_store: bool = config.ENABLE_AUTO_MEMORY_STORE,
        enable_auto_web_search: bool = config.ENABLE_AUTO_WEB_SEARCH,
        enable_auto_read_dropbox: bool = config.ENABLE_AUTO_READ_DROPBOX,
        file_paths: list[Path] | None = None
    ) -> None | str:
        """
        Model decide what memories to read.

        Manage tokens, compress session if needed.

        Note:
        - Only the user question and LLM response will be
          stored into chat history.

        Process:
        1. Model decides from {prompt} --> {memory_entries}
        2. Model decides from {memory_entries} + {prompt} --> {file_content} from dropbox
        3. Model decides from {memory_entries} + {file_content} + {prompt} --> {search_results}
        """
        self._manage_token_budget(prompt)

        cmd, user_prompt = _detect_cmd(prompt)

        if cmd:
            # Only use 'user_prompt' as 'prompt' here

            if cmd == "/forget":
                response = self.command.cmd_forget(user_prompt)
                if response:
                    print(response)
                # ==============
                # // End here //
                # ==============
                return

            if cmd == "/memorise":
                response = self.command.cmd_memorise(prompt=user_prompt)
                if response:
                    print(response)
                # ==============
                # // End here //
                # ==============
                return

            if cmd == "/recall":
                # User's question and agent's response were saved
                response = self.command.cmd_recall(
                    model_max_tokens=self.get_model_max_tokens,
                    prompt=user_prompt,
                    enable_extra_context=enable_extra_context,
                    enable_auto_memory_retrieve=enable_auto_memory_retrieve,
                    enable_auto_memory_store=enable_auto_memory_store,
                    enable_auto_web_search=enable_auto_web_search,
                    enable_auto_read_dropbox=enable_auto_read_dropbox,
                    limit=4,
                    file_paths=file_paths
                )
                if response:
                    print(response)
                # ==============
                # // End here //
                # ==============
                return

            if cmd == "/search":
                # User's question were saved
                response = self.command.cmd_search(user_prompt)
                if response:
                    print(response)

                self.memory.toggle_auto_store_memory_entry(
                    enable_auto_memory_store= True,
                    model_max_tokens=self.get_model_max_tokens,
                    context=self.chat.to_llm()
                )
                # ==============
                # // End here //
                # ==============
                return

        messages = self.chat.to_llm()

        # Toggle extra_context (read files)
        if enable_extra_context == True:
            added_file_contents = self.file_reader.read_files_with_context_prompt(
                context=messages,
                file_paths=file_paths,
            )

            # Store file into dropbox
            for path in file_paths:
                content, _ = self.file_reader.load_file_content(path)
                self.file_reader.store_file_in_dropbox(content, path)

        else:
            added_file_contents = ""

        memory_entries = self.memory.toggle_auto_retrive_memory_entry(
            enable_auto_memory_retrieve=enable_auto_memory_retrieve,
            prompt=prompt
        )

        file_contents, found_file_paths, read_dropbox = self.file_reader.toggle_auto_read_dropbox(
            messages=messages,
            prompt=prompt,
            memory_entries=memory_entries,
            enable_extra_context=enable_extra_context,
            added_file_contents=added_file_contents,
            enable_auto_read_dropbox=enable_auto_read_dropbox
        )

        search_results, query_with_urls, search = self.search_agent.toggle_auto_web_search(
            messages=messages,
            prompt=prompt,
            memory_entries=memory_entries,
            file_contents=file_contents,
            enable_extra_context=enable_extra_context,
            added_file_contents=added_file_contents,
            enable_auto_web_search=enable_auto_web_search
        )
        
        # Model answer
        if enable_extra_context:
            messages.append({
                "role": "user",
                "content": f"# Retrieved memory entries\n\n{memory_entries}\n\n---\n\n# Previously uploaded files\n\n{file_contents}\n\n---\n\n# Web search results\n\n{search_results}\n\n---\n\n# User input\n\n{prompt}\n\n## Uploaded files\n\n{added_file_contents}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"# Retrieved memory entries\n\n{memory_entries}\n\n---\n\n# Previously uploaded files\n\n{file_contents}\n\n---\n\n# Web search results\n\n{search_results}\n\n---\n\n# User input\n\n{prompt}\n\n"
            })
        answer, prompt_tokens, output_tokens = LLM.model_response(messages=messages, model = self.model)

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

        # Update dropbox metadata
        print("Updataing dropbox metadata...")
        self.file_reader._add_metadata_and_summary(file_paths)
        
        self.memory.toggle_auto_store_memory_entry(
            enable_auto_memory_store= True,
            model_max_tokens=self.get_model_max_tokens,
            context=self.chat.to_llm()
        )

        # ==============
        # // End here //
        # ==============
        return
