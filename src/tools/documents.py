import psycopg2
import ollama
import json
import mimetypes
import hashlib
from psycopg2 import sql
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

from src.agent.chat_logs import ChatLogs
from src.agent.models.embed import Embed
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
        self.embed      = Embed()

        self.doc_metadata   = self._get_all_documents_metadata()
        self.doc_names      = self._get_all_documents_names()

    # ================================================
    # CONTENT HASH
    # ================================================

    def _hash_content(self, cont: str) -> str:
        """
        Return a SHA-256 hash of raw text content,
        used for dedupe before embedding.
        """
        return hashlib.sha256(cont.encode("utf-8")).hexdigest()

    def _is_doc_cont_exist(self, new_hash: str) -> bool:
        """
        If the same document content was uploaded before (same
        hash, accross sessions), skip re-embedding. Return True
        if a duplicate exist.
        """
        self.cur.execute(
            """
            SELECT id FROM knowledge_base WHERE content_hash = %s
            """,
            (new_hash,)
        )
        existing = self.cur.fetchone()
        return existing is not None

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
        size: int,
        cont_hash: str,
        exprs_at: datetime | None = None,
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
                INSERT INTO knowledge_base (session_id, type, embedding, content, content_hash, expires_at, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
                """,
                (
                    self.chat_logs.sess_id,
                    "document",
                    str(embeddings),
                    cont,
                    cont_hash,
                    exprs_at,
                    json.dumps(metadata)
                )
            )
            self.conn.commit()
            return f"Added document '{doc_name}' to knowledge base"

        except Exception as e:
            self.conn.rollback()
            return f"Database insert error: {e}"

    def embed_txt_and_add_doc_to_kw_bs(self, path: Path, cont: str) -> str:
        """Embed text document(s) and upload to knowledge base."""
        doc_data = self.get_document_metadata_from_path(path)
        if not doc_data:
            return f"Error reading '{path}': Path does not exist"

        # Hash raw content before embedding and check duplicates
        cont_hash = self._hash_content(cont)
        if self._is_doc_cont_exist(cont_hash):
            return "Document already in knowledge base. Skipping re-embed"

        # Embed content
        cont, embeddings = self.embed.embedding_content(cont)
        if not embeddings:
            return "Failed to generate vector embedding: Failed to save entry"

        # Add embeddings to database
        name, mime, size = doc_data
        return self.add_document_to_kw_bs(
            doc_name=name, embeddings=embeddings, cont=cont, mime=mime, size=size, cont_hash=cont_hash
        )

    def embed_and_add_doc_to_kw_bs(self, path: Path) -> str:
        """Embed document(s) and upload to knowledge base."""
        doc_data = self.get_document_metadata_from_path(path)
        if not doc_data:
            return f"Error reading '{path}': Path does not exist"

        # Read with correct parsers
        cont = self.parsers.read_document(path)

        # Hash raw content before embedding and check duplicates
        cont_hash = self._hash_content(cont)
        if self._is_doc_cont_exist(cont_hash):
            return "Document already in knowledge base. Skipping re-embed"

        # Embed content
        cont, embeddings = self.embed.embedding_content(cont)
        if not embeddings:
            return "Failed to generate vector embedding: Failed to save entry"

        # Add embeddings to database
        name, mime, size = doc_data
        return self.add_document_to_kw_bs(
            doc_name=name, embeddings=embeddings, cont=cont, mime=mime, size=size, cont_hash=cont_hash
        )


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
