import psycopg2
import json
import mimetypes
import hashlib
from psycopg2 import sql
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone

from src.config import models
from src.config import files_and_directories as files_n_dir
from src.config import memory
from src.config import postgres

from src.agent import chat_logs, embed
from src.rag.documents.document_reader import DocumentReader
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

RETRIEVE_MEM_ENTRY_LIMIT    = memory.RETRIEVE_MEM_ENTRY_LIMIT
UPLOAD_DIR                  = files_n_dir.UPLOAD_DIR
ChatLogs                    = chat_logs.ChatLogs


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
        self.qry_limit = RETRIEVE_MEM_ENTRY_LIMIT

        self.chat_logs  = chat_logs
        self.doc_reader = DocumentReader()


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
            SELECT content, embeddings, prompt_tokens
            FROM knowledge_base
            WHERE content_hash = %s
            LIMIT 1
            """,
            (new_hash,)
        )
        row = self.cur.fetchone()
        if row is None:
            return None

        chnk_cont, embdings, chnk_tkns = row
        return chnk_cont, embdings, chnk_tkns


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

    def store_attachments(self, attchmnt_dict: dict[Path, dict[str, str]]) -> None:
        app_log.info(
            "Storing %d uploaded attachment(s) to session '%s' knowledge base",
            len(attchmnt_dict),
            self.sess_name
        )
        for doc_path, data in attchmnt_dict.items():
            cont = data["content"]
            format = data["format"]
            count = self.embed_and_add_to_kw_bs(
                path=doc_path, cont=cont, format=format
            )
            if not count:
                app_log.warning(
                    "Failed to store attachment '%s' to session '%s' knowledge base",
                    doc_path,
                    self.sess_name
                )
                print("Error: Failed to embed/store attachment to knowledge base")
                continue

            app_log.info(
                "Stored attachment '%s' as %s chunks to session '%s' knowledge base",
                doc_path,
                count,
                self.sess_name
            )
            print("Attachment stored to session knowledge base")
        return


    def get_attachments_content(
        self,
        is_attchmnt: bool,
        attch_paths: list[Path] | None
    ) -> dict[Path, dict[str, str]] | None:
        """Return content in attachment(s)."""
        if not is_attchmnt:
            return

        if not attch_paths:
            return

        attchmnt_dict = {}
        for path in attch_paths:

            cont, format = self.doc_reader.read_document(path)
            if not cont:
                app_log.warning("Failed to extract content from '%s'. Skipping", path.name)
                continue

            app_log.info("Content extracted from '%s'", path.name)
            attchmnt_dict[path] = {"content": cont, "format": format}

        return attchmnt_dict


    # ================================================
    # ADD DOCUMENTS INTO KNOWLEDGE BASE
    # ================================================

    def _add_to_kw_bs(
        self,
        doc_name: str,
        chnk_idx: int,
        embdings: list[float],
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
            "mime_type": mime,
            "size_bytes": size
        }

        try:
            self.cur.execute(
                """
                INSERT INTO knowledge_base (session_id, type, embeddings, prompt_tokens, content, content_hash, expires_at, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s)
                """,
                (
                    self.chat_logs.get_sess_id(),
                    "document",
                    str(embdings),
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

    def embed_and_add_to_kw_bs(
        self,
        path: Path,
        cont: str | None,
        format: str | None
    ) -> int | None:
        """
        Uses langchain text splitters for file format accordingly,
        embed each chunks and save to knowledge base.
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

        chnks = []

        if not cont and not format:
            cont, format = self.doc_reader.read_document(path)

        # Langchain text splitters
        app_log.info("Langchain Text Splitters splitting '%s' content to chunks", path.name)
        from src.rag.splitters import txt_spltr, md_spltr
        if format == "txt":
            chnks = txt_spltr.split_text(cont)
        if format == "md":
            chnks = md_spltr.split_text(cont)

        if not chnks:
            return

        count = 0
        for idx, chnk in enumerate(chnks):
            chnk_hash = self._hash_content(chnk)
            doc_chnk = self._is_doc_cont_chunk_exist(chnk_hash)

            # Skip embedding if already exists
            if doc_chnk:
                app_log.info(
                    "Document already exists in session '%s' knowledge base. Skipping re-embed",
                    self.sess_name
                )
                chnk_cont, embdings, chnk_tkns = doc_chnk

            else:
                chnk_cont, embdings, chnk_tkns = embed.embedding_content(chnk)
            hash = self._hash_content(chnk_cont)

            # Skip chunks that failed
            if not embdings:
                app_log.warning(
                    "Error occur in embedding chunk in '%s'. Skipping chunk", str(path)
                )
                continue

            # Skip upload if already exists
            if not self._is_doc_cont_chunk_exist(hash):
                result = self._add_to_kw_bs(
                    doc_name=name,
                    chnk_idx=idx,
                    embdings=embdings,
                    chnk_tkns=chnk_tkns,
                    cont=chnk_cont,
                    mime=mime,
                    size=size,
                    cont_hash=chnk_hash
                )
            count += 1

        return count


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

    def query_similar_knowledge(
        self,
        qry: str, qry_embdings: list[float],
        min_sim: float = 0.65
    ) -> list[dict[str, Any]] | None:
        """Queries knowledge base for similar content."""
        kw_dict = []

        self.cur.execute(
            """
            SELECT content, metadata, 1 - (embeddings <=> %s) AS cosine_similarity
            FROM knowledge_base
            WHERE 1 - (embeddings <=> %s) >= %s
            ORDER BY embeddings <=> %s ASC
            LIMIT %s;
            """,
            (
                str(qry_embdings),
                str(qry_embdings),
                min_sim,
                str(qry_embdings),
                self.qry_limit
            )
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
        prompt_embdings: list[float]
    ) -> list[dict[str, Any]] | None:
        """Auto fetches previous documents contents ability, return relevant contents if its toggled on."""
        if is_auto_doc_rtve:
            app_log.debug(
                "Auto document retrieve on. Querying session '%s' knowledge_base for relevant documents",
                self.sess_name
            )
            return self.query_similar_knowledge(prompt, prompt_embdings)
        return
