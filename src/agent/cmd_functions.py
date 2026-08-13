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
from src.logger import get_logger


logger = get_logger(__name__)


class Command:
    def __init__(
        self,
        model: str | None = None,
        session: str | None = None,
        project: str | None = None
    ):
        self.model          = model
        self.session        = session
        self.project        = project
        self.chat           = Chat(session=session)
        self.memory         = Memory(project=project)
        self.file_reader    = FileReader(session=session)
        self.extra_context  = ExtraContext(session=session)
        self.search_agent   = SearchAgent()

    # ========================================================
    # Memory
    # ========================================================

    def cmd_forget(self, prompt: str) -> str:
        """
        Find exact match in memory from user prompt and remove said entry.
        Note: User's question won't be saved.
        """
        if not prompt:
            logger.warning("Command '/forget' aborted: No prompt was provided")
            return "Please specify what to forget."

        match_data = self.memory.get_exact_match(prompt)

        if not match_data:
            logger.warning("Command '/forget' failed: No matching memory entry found")
            return f"Error: No matching memory found."

        target_id, matched_content = match_data

        print(f"Warning: [ChromaDB] Initializing delte sequence for: '{matched_content}...'")
        choice = input("Press [Enter] to confirm deletion or type [c] to cancel:")

        if choice == "c":
            logger.info("Memory deletion cancelled by the user")
            return "Deletion cancelled."

        self.memory.delete_from_db([target_id])
        logger.info("Deleted memory entry from database (target_id=%s)", target_id)
        return "Entry deleted."

    def cmd_memorise(self, prompt: str) -> str:
        """
        Extract key info from user prompt and save entries to memory.
        Note: User's question will be saved directly, this function
              will then generate a pre-written assistant message.
        """
        if not prompt:
            logger.warning("Command '/memorise' aborted: No prompt was provided")
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
            logger.info("Extracted and saved %d memory entry(s) to database", len(created_ids))
        else:
            logger.warning("Command '/memorise' failed: No memory entries extracted by model")
            return "Error: No data was extracted by the model."

        choice = input("Press [Enter] to continue or type [u] to undo:")

        if choice == "u":
            self.memory.delete_from_db(created_ids)
            logger.info("Memory creation undone by user: Removed %d entry(s)", len(created_ids))
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
            logger.warning("Command '/recall' aborted: No prompt was provided")
            return "Please specify what to recall."

        messages = self.chat.to_llm()

        # Auto read dropbox
        file_contents, found_file_paths, read_dropbox = self.file_reader.toggle_auto_read_dropbox(
            messages=messages,
            prompt=prompt,
            memory_entries="",
            enable_extra_context=enable_extra_context,
            added_file_contents="",
            enable_auto_read_dropbox=enable_auto_read_dropbox
        )
        if file_contents:
            logger.info("File(s) contents was retrieved from session dropbox '%s': %s", self.file_reader.dropbox_dir, str(found_file_paths))
            logger.info("Retieved %d file(s) from session dropbox: %s", len(found_file_paths), self.file_reader.dropbox_dir)
        else:
            logger.info("No files retrieved from dropbox")

        # Auto web search
        search_results, query_with_urls, search = self.search_agent.toggle_auto_web_search(
            messages=messages,
            prompt=prompt,
            memory_entries="",
            file_contents=file_contents,
            enable_extra_context=enable_extra_context, added_file_contents="",
            enable_auto_web_search=enable_auto_web_search
        )
        if search_results:
            logger.info("Retrieved web search results across %d query(s)", len(query_with_urls))
        else:
            logger.info("No web search results retrieved")

        # Toggle extra_context (read files)
        if enable_extra_context == True:
            added_file_contents = self.extra_context.all(
                messages=messages,
                enable_extra_context=enable_extra_context,
                file_paths=file_paths,
            )
            logger.info("Retrieved extra context from uploaded files")

        # Toggle retrieve memory entry(s)
        recalled_entries = self.memory.toggle_auto_retrive_memory_entry(
            enable_auto_memory_retrieve=enable_auto_memory_retrieve,
            prompt=prompt
        )
        if recalled_entries:
            recalled = "Found matching entries in long-term memory:\n" + "\n".join(
                f"- [{entry['category']}] {entry['content']}" for entry in recalled_entries
            )
            logger.info("Retrieved %d relevant memory entry(s)", len(recalled_entries))
        else:
            logger.info("No matching memory entries retrieved")
            recalled = f"No matching memories found."

        answer, prompt_tokens, output_tokens = LLM.response_memory_recall(
            model=self.memory.model,
            system_prompt=config.MEM_RECALL_INTERPRET_PROMPT,
            recalled=recalled,
            prompt=prompt,
            context=messages
        )

        # Save user's question
        self.chat.append_user_message_with_metadata(content=prompt, state="external")

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
            logger.error("Command '/search' aborted: No prompt was provided")
            return "Please specify what to search."

        print("Generating query...")
        query = self.search_agent.generates_query(
            context=self.chat.to_llm(),
            prompt=prompt
        )
        if query:
            logger.info("Generated search query: %s", query)

        response, query_with_urls, prompt_tokens, output_tokens, search = self.search_agent.web(
            query=query,
            context=self.chat.to_llm(),
            prompt=prompt,
            max_results=config.MAX_RESULTS
        )
        logger.info("Processed web search response")

        # Save user messages
        self.chat.append_user_message_with_metadata(content=prompt, state="external")

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
