import psycopg2
import ollama
import json
import mimetypes
import hashlib
from psycopg2 import sql
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

from src.config.memory import RETRIEVE_MEM_ENTRY_LIMIT
from src.config.files_and_directories import UPLOAD_DIR
from src.config.postgres import conn
from src.agent.chat_logs import ChatLogs
from src.agent.models.embed import Embed
from src.tools.documents.basic_parsers import BasicParsers
from src.tools.documents.document_reader import DocumentReader
from src.logger import get_logger


log = get_logger(__name__)


class DocumentKnowledgeBase:
    def __init__(
        self,
        conn,
        chat_logs: ChatLogs,
        sess_name: str | None = None,
    ):
        self.conn   = conn
        self.cur    = self.conn.cursor()

        self.sess_name = sess_name
        self.qry_limit  = RETRIEVE_MEM_ENTRY_LIMIT

        self.chat_logs  = chat_logs
        self.doc_reader = DocumentReader()
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
            log.warning("'%s' does not exists in '%s'", path.name, UPLOAD_DIR)
            return

        # Prevent duplicated name in the metadata
        name = path.name
        counter = 1
        while name in self.doc_names:
            name = f"{path.stem}({counter}){path.suffix}"
            counter += 1

        mime, _ = mimetypes.guess_type(path)
        mime = mime or "unknown"
        size = path.stat().st_size
        log.debug("Metadata extracted from %s", path.name)
        return name, mime, size


    # ================================================
    # ATTACHMENTS
    # ================================================

    def get_attachments_content(
        self,
        is_attchmnt: bool,
        attch_paths: list[Path] | None
    ) -> dict[Path, str] | None:
        """Return content in attachment(s)."""
        if not is_attchmnt:
            return

        if not attch_paths:
            return

        attchmnt_dict = {}
        for path in attch_paths:

            cont = self.doc_reader.read_document(path)
            if not cont:
                log.warning("Failed to extract content from '%s'. Skipping", path.name)
                return

            log.info("Content extracted from '%s'", path.name)
            attchmnt_dict[path] = cont

        return attchmnt_dict


    # ================================================
    # ADD DOCUMENTS INTO KNOWLEDGE BASE
    # ================================================

    def _add_document_to_kw_bs(
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
            log.info("Added document '%s' to session '%s' knowledge base", doc_name, self.sess_name)
            return f"Added document '{doc_name}' to knowledge base"

        except Exception as e:
            self.conn.rollback()
            log.warning("Database insert error: %s", e)
            return f"Database insert error: {e}"


    def embed_txt_and_add_doc_to_kw_bs(self, path: Path, cont: str) -> str:
        """Embed text document(s) and upload to knowledge base."""
        doc_data = self.get_document_metadata_from_path(path)
        if not doc_data:
            log.warning(
                "Failed to read '%s': File does not exists in %s",
                path.name,
                UPLOAD_DIR
            )
            return f"Error reading '{path}': Path does not exist"

        # Hash raw content before embedding and check duplicates
        cont_hash = self._hash_content(cont)
        if self._is_doc_cont_exist(cont_hash):
            log.info(
                "Document already exists in session '%s' knowledge base. Skipping re-embed",
                self.sess_name
            )
            return "Document already in knowledge base. Skipping re-embed"

        # Embed content
        cont, embeddings = self.embed.embedding_content(cont)
        if not embeddings:
            log.warning(
                "Unable to save document to session '%s' knowledge base: Failed to generate vector embedding for '%s'",
                self.sess_name,
                path.name
            )
            return f"Unable to save document to session '{self.sess_name}' knowledge base: Failed to generate vector embedding"

        # Add embeddings to database
        name, mime, size = doc_data
        return self._add_document_to_kw_bs(
            doc_name=name, embeddings=embeddings, cont=cont, mime=mime, size=size, cont_hash=cont_hash
        )


    def embed_and_add_doc_to_kw_bs(self, path: Path) -> str:
        """Embed document(s) and upload to knowledge base."""
        doc_data = self.get_document_metadata_from_path(path)
        if not doc_data:
            log.warning(
                "Failed to read '%s': File does not exists in %s", path.name, UPLOAD_DIR
            )
            return f"Error reading '{path}': Path does not exist"

        # Read with correct parsers
        cont = self.doc_reader.read_document(path)

        # Hash raw content before embedding and check duplicates
        cont_hash = self._hash_content(cont)
        if self._is_doc_cont_exist(cont_hash):
            log.info(
                "Document already exists in session '%s' knowledge base. Skipping re-embed",
                self.sess_name
            )
            return "Document already in knowledge base. Skipping re-embed"

        # Embed content
        cont, embeddings = self.embed.embedding_content(cont)
        if not embeddings:
            log.warning(
                "Unable to save document to session '%s' knowledge base: Failed to generate vector embedding for '%s'",
                self.sess_name,
                path.name
            )
            return f"Unable to save document to session '{self.sess_name}' knowledge base: Failed to generate vector embedding"

        # Add embeddings to database
        name, mime, size = doc_data
        return self._add_document_to_kw_bs(
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
                log.warning(
                    "Failed to retrieve documents: Session '%s' knowledge base is empty",
                    self.sess_name
                )
                return f"Failed to retrieve documents: Session '{self.sess_name}' knowledge base is empty"
            return rows

        except Exception as e:
            self.conn.rollback()
            log.warning("Database query documents data error: %s", e)
            return f"Database query documents data error: {e}"


    def _get_all_documents_names(self) -> list[str]:
        """Return a list of all uploaded documents names in the database."""
        rows = self.doc_metadata
        if isinstance(rows, str):
            return []

        # Unpack data in metadata
        doc_names = []
        for _, metadata in rows:
            doc_names.append(metadata["document_name"])
        return doc_names


    # ================================================
    # LIST CONTENTS
    # ================================================

    def list_all_uploaded_documents(self) -> str:
        """Return a list of all document(s) in the database."""
        rows = self.doc_metadata
        # if isinstance(rows, str):
        #     return rows

        # Unpack data in metadata
        lines = [
            f"UPLOADED DOCUMENT(S)\n"
            f"====================\n"
            f"UPLOADED AT\t\t\t\tNAME"
        ]
        for time, metadata in rows:
            data_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            doc_name = data_dict.get("document_name", "Unknown")
            lines.append(f"{str(time)}\t{doc_name}")
        return "\n".join(lines)


    # ==================================================
    # QUERY KNOWNLEDGE BASE
    # ==================================================

    def query_similar_knowledge(self, qry: str) -> list[dict[str, Any]] | None:
        """Queries knowledge base for similar content."""
        kw_dict = []

        qry, qry_embeddings = self.embed.embedding_content(qry)
        self.cur.execute(
            """
            SELECT content, metadata, 1 - (embedding <=> %s) AS cosine_similarity
            FROM knowledge_base
            ORDER BY embedding <=> %s ASC
            LIMIT %s;
            """,
            (str(qry_embeddings), str(qry_embeddings), self.qry_limit)
        )
        rows = self.cur.fetchall()

        # ////////////////////////////////////////////////////
        # CURRENTLY ONLY FOR DOCUMENT KNOWLEDGE BASE
        if kw_dict:
            for cont, metadata, score in rows:
                kw_dict.append({
                    "document_name": metadata.get("document_name", "Unknown"),
                    "content": cont,
                    "similarity": score
                })
            log.info("%d retrieved from session '%s' knowledge base", len(rows), self.sess_name)
            return kw_dict
        else:
            return
        # ////////////////////////////////////////////////////


    # ==================================================
    # AUTO FUNCTIONS
    # ==================================================

    def toggle_auto_retrieve_sess_docs(
        self,
        is_auto_doc_rtve: bool,
        prompt: str,
    ) -> list[dict[str, Any]] | None:
        """Auto fetches previous documents contents ability, return relevant contents if its toggled on."""
        if is_auto_doc_rtve:
            log.debug(
                "Auto document retrieve on. Querying session '%s' knowledge_base for relevant documents",
                self.sess_name
            )
            return self.query_similar_knowledge(prompt)
        return
