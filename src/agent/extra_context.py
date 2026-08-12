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
        file_paths: list[Path] | None,
        prompt: str,
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

        file_content = f"# Newly uploaded file\n\n{added_file_contents}\n\n---\n\n"

        # Update dropbox metadata
        print("Updataing dropbox metadata...")
        self.file_reader._add_metadata_and_summary(file_paths)
