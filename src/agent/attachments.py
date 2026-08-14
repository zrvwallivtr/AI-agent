from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat import Chat
from src.tools.search import is_connected, SearchAgent
from src.tools.file_reader import FileReader
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


class Attachments:
    def __init__(self, session: str | None = None):
        self.session        = session
        self.file_reader    = FileReader(session=session)

    def files(
            self,
            messages: list[dict],
            enable_attachments: bool,
            file_paths: list[Path] | None
    ) -> str:
        """Return Attachment(s) contents"""
        if enable_attachments == False:
            return ""

        if not file_paths:
            logger.error("Unable to add extra context: No file path provided")
            return ""

        logger.info("File(s) attached. Extracting content")
        added_file_contents = self.file_reader.read_files_with_context_prompt(
            context=messages,
            file_paths=file_paths
        )
        if not added_file_contents:
            logger.error("Extraction failed: Skipping")
            return ""
        logger.info("Extracted attachment(s) content")

        # Store file into dropbox
        logger.info("Storing file(s) into session dropbox '%s': %s", self.file_reader.dropbox_dir, file_paths)
        for path in file_paths:
            content, _ = self.file_reader.load_file_content(path)
            self.file_reader.store_file_in_dropbox(content, path)
        return added_file_contents
