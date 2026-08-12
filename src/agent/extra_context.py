from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat import Chat
from src.tools.search import is_connected, SearchAgent
from src.tools.file_reader import FileReader

from src import config


class ExtraContext:
    def __init__(
        self,
        model: str | None = None,
        session: str | None = None,
        project: str | None = None
    ):
        model = config.MODEL if model is None else model

        if config.MODEL_MAX_TOKENS:
            self.tokens = Tokens(model, config.MODEL_MAX_TOKENS)
        else:
            self.tokens = Tokens(model)

        self.model          = model  # model is specified in the config file
        self.chat           = Chat(session=session)
        self.file_reader    = FileReader(session=session)
        self.search_agent   = SearchAgent()

    def all(
        self,
        messages: list[dict],
        file_contents: str,
        dropbox_files: list[Path],
        read_dropbox: bool,
        search_results: str,
        query_with_urls: list[dict[str, list[str]]],
        search: bool,
        file_paths: list[Path] | None,
        prompt: str,
        memory_entries: str = "",
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
