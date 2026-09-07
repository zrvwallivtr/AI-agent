import psycopg2
from agent.embed_chunks import EmbedChunks
import ollama
import json
import mimetypes
import hashlib
from psycopg2 import sql
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

from src.config.models import MODEL
from src.config.memory import RETRIEVE_MEM_ENTRY_LIMIT
from src.config.files_and_directories import UPLOAD_DIR
from src.config.postgres import conn
from src.agent.chat_logs import ChatLogs
from src.agent.models.embed import Embed
from src.agent.embed_chunks import EmbedChunks
from src.tools.documents.basic_parsers import BasicParsers
from src.tools.documents.document_reader import DocumentReader
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


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

        self.model          = MODEL
        self.chat_logs      = chat_logs
        self.doc_reader     = DocumentReader()
        self.embed          = Embed()
        self.embed_chnks    = EmbedChunks()

        # self.doc_metadata   = self._get_all_documents_metadata()
        # self.doc_names      = self._get_all_documents_names()


    # ================================================
    # CONTENT HASH
    # ================================================

    def _hash_content(self, cont: str) -> str:
        """
        Return a SHA-256 hash of raw text content,
        used for dedupe before embedding.
        """
        return hashlib.sha256(cont.encode("utf-8")).hexdigest()


    def _is_doc_cont_chunk_exist(self, new_hash: str) -> tuple[str, list[float], int] | None:
        """
        Check if the same document content chunk was uploaded before (same
        hash, accross sessions). Return embeddings if a duplicate exist.
        """
        self.cur.execute(
            """
            SELECT content, embedding, prompt_tokens
            FROM knowledge_base
            WHERE content_hash = %s
            LIMIT 1
            """,
            (new_hash,)
        )
        row = self.cur.fetchone()
        if row is None:
            return None

        chnk_cont, embedding, chnk_tkns = row
        return chnk_cont, embedding, chnk_tkns


    # ================================================
    # FILE METADATA
    # ================================================

    def get_document_metadata_from_path(self, path: Path) -> tuple[str, str, int] | None:
        """Return name, mime type and size bytes from given path."""
        if not path.exists():
            app_log.warning("'%s' does not exists in '%s'", path.name, UPLOAD_DIR)
            return

        # Prevent duplicated name in the metadata
        name = path.name
        counter = 1
        while name in self.get_all_docs_names():
            name = f"{path.stem}({counter}){path.suffix}"
            counter += 1

        mime, _ = mimetypes.guess_type(path)
        mime = mime or "unknown"
        size = path.stat().st_size
        app_log.debug("Metadata extracted from %s", path.name)
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
                app_log.warning("Failed to extract content from '%s'. Skipping", path.name)
                continue

            app_log.info("Content extracted from '%s'", path.name)
            attchmnt_dict[path] = cont

        return attchmnt_dict


    # ================================================
    # ADD DOCUMENTS INTO KNOWLEDGE BASE
    # ================================================

    def _add_document_chunk_to_kw_bs(
        self,
        doc_name: str,
        chnk_idx: int,
        tol_chnks: int,
        embeddings: list[float],
        chnk_tkns: int,
        cont: str,
        mime: str,
        size: int,
        cont_hash: str,
        exprs_at: datetime | None = None,
    ) -> str:
        """Upload documents to knowledge base."""
        metadata = {
            "document_name": doc_name,
            "document_chunk_index": chnk_idx,
            "document_total_chunks": tol_chnks,
            "mime_type": mime,
            "size_bytes": size
        }

        try:
            self.cur.execute(
                """
                INSERT INTO knowledge_base (session_id, type, embedding, prompt_tokens, content, content_hash, expires_at, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s)
                """,
                (
                    self.chat_logs.get_sess_id(),
                    "document",
                    str(embeddings),
                    chnk_tkns,
                    cont,
                    cont_hash,
                    exprs_at,
                    json.dumps(metadata)
                )
            )
            self.conn.commit()
            app_log.info("Added document '%s' to session '%s' knowledge base", doc_name, self.sess_name)
            return f"Added document '{doc_name}' to knowledge base"

        except Exception as e:
            self.conn.rollback()
            app_log.warning("Database insert error: %s", e)
            return f"Database insert error: {e}"


    # ================================================
    # EMBEDDING
    # ================================================

    def embedding_paragraph_chunks_and_add_to_kw_bs(self, path: Path, cont: str | None) -> list[dict] | None:
        """
        Uses paragraph chunking method and embed each chunks and upload chunks to knowledge base.
        """
        doc_data = self.get_document_metadata_from_path(path)
        if not doc_data:
            app_log.warning(
                "Failed to read '%s': File does not exists in %s",
                path.name,
                UPLOAD_DIR
            )
            return
        name, mime, size = doc_data

        if not cont:
            cont = self.doc_reader.read_document(path)

        chnks = self.embed_chnks.paragraph_chunking(cont)
        tol_chnks = len(chnks)
        results = []

        for idx, chnk in enumerate(chnks):
            chnk_hash = self._hash_content(chnk)
            doc_chnk = self._is_doc_cont_chunk_exist(chnk_hash)

            # Skip embedding if already exists
            if doc_chnk:
                app_log.info(
                    "Document already exists in session '%s' knowledge base. Skipping re-embed",
                    self.sess_name
                )
                chnk_cont, embedding, chnk_tkns = doc_chnk
            else:
                chnk_cont, embedding, chnk_tkns = self.embed.embedding_content(chnk)

            # Skip chunks that failed
            if not embedding:
                app_log.warning(
                    "Error occur in embedding chunk in '%s'. Skipping chunk",
                    str(path)
                )
                continue

            result = self._add_document_chunk_to_kw_bs(
                doc_name=name,
                chnk_idx=idx,
                tol_chnks=tol_chnks,
                embeddings=embedding,
                chnk_tkns=chnk_tkns,
                cont=chnk_cont,
                mime=mime,
                size=size,
                cont_hash=chnk_hash
            )
            results.append({"chunk_index": idx, "status": result})

        return results


    # ================================================
    # FROM DOCUMENTS IN KNOWLEDGE BASE
    # ================================================

    def _get_all_docs_metadata(self) -> list[tuple[Any, Any]] | str:
        """Return a list of all uploaded documents data in the database."""
        try:
            self.cur.execute(
                """
                SELECT created_at, metadata
                FROM knowledge_base
                WHERE session_id = %s AND type = %s
                """,
                (self.chat_logs.get_sess_id(), "document")
            )
            rows = self.cur.fetchall()

            if not rows:
                app_log.warning(
                    "Failed to retrieve documents: Session '%s' knowledge base is empty",
                    self.sess_name
                )
                return f"Failed to retrieve documents: Session '{self.sess_name}' knowledge base is empty"
            return rows

        except Exception as e:
            self.conn.rollback()
            app_log.warning("Database query documents data error: %s", e)
            return f"Database query documents data error: {e}"


    def get_all_docs_names(self) -> list[str]:
        """Return a list of all uploaded documents names in the database."""
        rows = self._get_all_docs_metadata()
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
        rows = self._get_all_docs_metadata()
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

    def query_similar_knowledge(self, qry: str, qry_embeddings: list[float]) -> list[dict[str, Any]] | None:
        """Queries knowledge base for similar content."""
        kw_dict = []

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

        if rows:
            for cont, metadata, score in rows:
                kw_dict.append({
                    "document_name": metadata.get("document_name", "Unknown"),
                    "content": cont,
                    "similarity": score
                })
            app_log.info("%d retrieved from session '%s' knowledge base", len(rows), self.sess_name)
            return kw_dict
        else:
            return


    # ==================================================
    # AUTO FUNCTIONS
    # ==================================================

    def toggle_auto_retrieve_sess_docs(
        self,
        is_auto_doc_rtve: bool,
        prompt: str,
        prompt_embeddings: list[float]
    ) -> list[dict[str, Any]] | None:
        """Auto fetches previous documents contents ability, return relevant contents if its toggled on."""
        if is_auto_doc_rtve:
            app_log.debug(
                "Auto document retrieve on. Querying session '%s' knowledge_base for relevant documents",
                self.sess_name
            )
            return self.query_similar_knowledge(prompt, prompt_embeddings)
        return
