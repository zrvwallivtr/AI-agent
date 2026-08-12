from re import search
import ollama
from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat import Chat
from src.agent.memory import Memory
from src.agent.extra_context import ExtraContext
from src.tools.search import is_connected, SearchAgent
from src.tools.file_reader import FileReader
from src import config


class Command:
    def __init__(
        self,
        model: str | None = None,
        session: str | None = None,
        project: str | None = None
    ):
        self.model          = model
        self.chat           = Chat(session=session)
        self.memory         = Memory(project=project)
        self.file_reader    = FileReader(session=session)
        self.extra_context  = ExtraContext(model=model, session=session, project=project)
        self.search_agent   = SearchAgent()
        self.project        = project

    # ========================================================
    # Memory
    # ========================================================

    def cmd_forget(self, prompt: str) -> str:
        """
        Find exact match in memory from user prompt and remove said entry.
        Note: User's question won't be saved.
        """
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

    def cmd_memorise(
        self,
        prompt: str,
    ) -> str:
        """
        Extract key info from user prompt and save entries to memory.
        Note: User's question will be saved directly, this function
              will then generate a pre-written assistant message.
        """
        if not prompt:
            return "Please specify what to memorize."

        messages = self.chat.to_llm()
        messages.append({
            "role": "user",
            "content": prompt
        })

        print(f"Extracting content from user's input...")
        created_ids, prompt_tokens, output_tokens = self.memory.extract_entries_and_store_to_db(
            context=messages,
            prompt=prompt,
            source="manual",
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
        confirmation = "Important information(s) has been extracted and uploaded to ChromaDB."
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external"
        )
        self.chat.append_assistant_message_with_metadata(
            content=confirmation,
            state="internal",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        return "Memory saved to database."

    def cmd_recall(
        self,
        model_max_tokens: int,
        prompt: str,
        enable_extra_context: bool,
        enable_auto_memory_retrieve: bool,
        enable_auto_memory_store: bool,
        enable_auto_web_search: bool,
        enable_auto_read_dropbox: bool,
        limit: int,
        file_paths: list[Path] | None = None
    ) -> str | None:
        """Retrieve and print relevant entries accourding to user prompt."""
        if not prompt:
            return "Please specify what to recall."

        messages = self.chat.to_llm()

        file_contents, found_file_paths, read_dropbox = self.file_reader.toggle_auto_read_dropbox(
            model_max_tokens=model_max_tokens,
            enable_auto_read_dropbox=enable_auto_read_dropbox,
            messages=messages,
            prompt=prompt
        )

        search_results, query_with_urls, search = self.search_agent.toggle_auto_web_search(
            model_max_tokens=model_max_tokens,
            enable_auto_web_search=enable_auto_web_search,
            messages=messages,
            prompt=prompt,
        )

        # Toggle extra_context (read files)
        if enable_extra_context == True:
            self.extra_context.all(
                messages=messages,
                file_contents=file_contents,
                dropbox_files=found_file_paths,
                read_dropbox=read_dropbox,
                search_results=search_results,
                query_with_urls=query_with_urls,
                search=search,
                file_paths=file_paths,
                prompt=prompt
            )

        recalled_entries = self.memory.retrieve_relevant_entry(prompt, limit=limit)

        if recalled_entries:
            recalled = "Found matching entries in long-term memory:\n" + "\n".join(
                f"- [{entry['category']}] {entry['content']}" for entry in recalled_entries
            )
        else:
            recalled = f"No matching memories found."

        answer, prompt_tokens, output_tokens = LLM.response_memory_recall(
            model=self.memory.model,
            system_prompt=config.MEM_RECALL_INTERPRET_PROMPT,
            recalled=recalled,
            prompt=prompt,
            context=messages
        )

        # Save user's question
        self.chat.append_user_message_with_metadata(
            content=prompt,
            state="external"
        )

        # Add recalled memory to chat (Embedding model: no tokens counts)
        self.chat.append_assistant_message_with_metadata(
            content=answer,
            state="external",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        return

    # ========================================================
    # Search
    # ========================================================

    def cmd_search(self, prompt: str) -> str | None:
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
            max_results=config.MAX_RESULTS
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
