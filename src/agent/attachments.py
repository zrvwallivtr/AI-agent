from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat_logs import ChatLogs
from src.tools.search import is_connected, SearchAgent
from src.tools.parsers import Parsers
from src.tools.doc_knowledge_base import DocKnowledgeBase
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


class Attachments:
    def __init__(self, sess_name: str | None = None):
        self.sess_name      = sess_name
        self.parsers        = Parsers()
        self.doc_kw_bs      = DocKnowledgeBase(sess_name=self.sess_name)

    def files(
        self,
        is_attchmnt: bool,
        file_paths: list[Path] | None
    ) -> dict[str, str]:
        """Return Attachment(s) contents"""
        if not is_attchmnt:
            return {}

        if not file_paths:
            logger.error("Unable to add extra context: No file path provided")
            return {}

        file_dict = {}
        for path in file_paths:
            cont = self.parsers._read_file(path)
            if not cont:
                logger.error("Extraction failed: Skipping")
                return {}
            logger.info("Extracted attachment(s) content")
            file_dict[path] = cont
        return file_dict
