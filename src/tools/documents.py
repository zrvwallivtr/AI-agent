import psycopg2
import ollama
import json
import mimetypes
from psycopg2 import sql
from pathlib import Path
from typing import Any

from src.agent.chat_logs import ChatLogs
from src.tools.parsers import Parsers
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


class Document:
    def __init__(
        self,
        sess_name: str | None = None,
    ):
        self.conn   = psycopg2.connect(
            dbname=config.DBNAME,
            user=config.USER,
            password=config.PASSWORD,
            host=config.HOST,
            port=config.PORT
        )
        self.cur    = self.conn.cursor()

        self.sess_name = sess_name
        self.emb_model = config.EMBED_MODEL

        self.chat_logs  = ChatLogs(sess_name=self.sess_name)
        self.parsers    = Parsers()

        self.doc_metadata   = self._get_all_documents_metadata()
        self.doc_names      = self._get_all_documents_names()

    # ================================================
    # FILE METADATA
    # ================================================

    def get_document_metadata_from_path(self, path: Path) -> tuple[str, str, int] | None:
        """Return name, mime type and size bytes from given path."""
        if not path.exists():
            return None

        # Prevent duplicated name in the metadata
        name = path.name
        counter = 1
        while name in self.doc_names:
            name = f"{path.stem}({counter}){path.suffix}"
            counter += 1

        mime, _ = mimetypes.guess_type(path)
        mime = mime or "unknown"
        size = path.stat().st_size
        return name, mime, size

    # ================================================
    # ADD DOCUMENTS INTO KNOWLEDGE BASE
    # ================================================

    def add_document_to_kw_bs(
        self,
        doc_name: str,
        embeddings: list[float],
        cont: str,
        mime: str,
        size: int
    ) -> str:
        """Upload documents to knowledge base."""
        metadata = {
            "document_name": doc_name,
            "mime_type": mime,
            "size_bytes": size
        }
        try:
            self.cur.execute(
                """
                INSERT INTO knowledge_base (session_id, type, embedding, content, metadata)
                VALUES (%s, %s, %s::vector, %s, %s)
                """,
                (self.chat_logs.sess_id, "document", str(embeddings), cont, json.dumps(metadata))
            )
            self.conn.commit()
            return f"Added document '{doc_name}' to knowledge base"

        except Exception as e:
            self.conn.rollback()
            return f"Database insert error: {e}"

    # ================================================
    # FROM DOCUMENTS IN KNOWLEDGE BASE
    # ================================================

    def _get_all_documents_metadata(self) -> list[tuple[Any, Any]] | str:
        """Return a list of all uploaded documents data in the database."""
        try:
            self.cur.execute(
                """
                SELECT created_at, metadata
                FROM knowledge_base
                WHERE session_id = %s AND type = %s
                """,
                (self.chat_logs.sess_id, "document")
            )
            rows = self.cur.fetchall()
            if not rows:
                return f"Failed to retrieve documents: Session '{self.sess_name}' knowledge base is empty"
            return rows

        except Exception as e:
            self.conn.rollback()
            return f"Database query documents data error: {e}"

    def _get_all_documents_names(self) -> list[str]:
        """Return a list of all uploaded documents names in the database."""
        rows = self.doc_metadata
        if isinstance(rows, str):
            return []

        # Unpack data in metadata
        doc_names = []
        for _, metadata in rows:
            data_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            doc_names.append(data_dict["document_name"])
        return doc_names

    def get_attachments_content(
        self,
        is_attchmnt: bool,
        doc_paths: list[Path] | None
    ) -> dict[Path, str] | None:
        """Return Attachment(s) contents"""
        if not is_attchmnt:
            return

        if not doc_paths:
            logger.error("Unable to add extra context: No path(s) provided")
            return

        attchmnt_dict = {}
        for path in doc_paths:
            cont = self.parsers.read_document(path)
            if not cont:
                logger.error("Extraction failed: Skipping")
                return
            logger.info("Extracted attachment(s) content")
            attchmnt_dict[path] = cont
        return attchmnt_dict
