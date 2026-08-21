import psycopg2
import ollama
import json
import mimetypes
from psycopg2 import sql
from pathlib import Path
from typing import Any

from src.agent.llm import LLM
from src.agent.chat_logs import ChatLogs
from src.agent.tokens_handler import Tokens
from src.tools.parsers import Parsers
from src.models_database import EMB_MODEL_DIMENSION
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


class DocKnowledgeBase:
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

        self.sess_name          = sess_name
        self.model              = config.MODEL
        self.emb_model          = config.EMBED_MODEL
        self.emb_dmsion         = EMB_MODEL_DIMENSION[self.emb_model]
        self.file_or_not_prompt = config.FILE_OR_NOT_PROMPT
        self.file_list_prompt   = config.GET_FILE_LIST_PROMPT
        self.gen_summary_prompt = config.GEN_SUMMARY_PROMPT
        self.qry_limit          = config.RETRIEVE_MEM_ENTRY_LIMIT

        self.chat_logs          = ChatLogs(sess_name=self.sess_name)
        self.tokens             = Tokens(model=self.model)
        self.parsers            = Parsers()

        self._init_doc_db()
        self.ava_files          = self._get_all_filenames()

    def _init_doc_db(self):
        """Create document table if missing."""
        self.cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS VECTOR;
            """
        )
        create_kw_bs_tbl = sql.SQL("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                type VARCHAR(20) NOT NULL,
                embedding VECTOR({dimension}),
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb
            );
        """).format(dimension=sql.SQL(str(int(self.emb_dmsion))))
        self.cur.execute(create_kw_bs_tbl)
        self.conn.commit()

    # ================================================
    # EMBEDDING FILE CONTENT
    # ================================================

    def _embedding_content(self, cont: str) -> tuple[str, list[float]]:
        """Generate embedding from given texts."""
        try:
            response = ollama.embed(model=self.emb_model, input=cont)
            embeddings = response["embeddings"][0]
            if not embeddings:
                return "Error: Model failed to generate vector embedding", []
            return cont, embeddings

        except Exception as e:
            return f"Error: {e}", []

    # ================================================
    # FILE METADATA
    # ================================================

    def _get_file_metadata(self, path: Path) -> tuple[str, str, int] | None:
        """Return filename, mime type and size bytes from given file path."""
        if not path.exists():
            return None

        # Prevent duplicated filename in the metadata
        filename = path.name
        counter = 1
        while filename in self.ava_files:
            filename = f"{path.stem}({counter}){path.suffix}"
            counter += 1

        mime, _ = mimetypes.guess_type(path)
        mime = mime or "unknown"
        size = path.stat().st_size
        return filename, mime, size

    # ================================================
    # KNOWLEDGE BASE
    # ================================================

    def _add_file_to_kw_bs(
        self,
        filename: str,
        embeddings: list[float],
        cont: str,
        mime: str,
        size: int
    ) -> str:
        """Upload file to knowledge base."""
        metadata = {
            "filename": filename,
            "mime_type": mime,
            "size_bytes": size
        }
        try:
            self.cur.execute(
                """
                INSERT INTO knowledge_base (session_id, type, embedding, content, metadata)
                VALUES (%s, %s, %s::vector, %s, %s)
                """,
                (self.chat_logs.sess_id, "file_upload", str(embeddings), cont, json.dumps(metadata))
            )
            self.conn.commit()
            return f"Added file '{filename}' to knowledge base"

        except Exception as e:
            self.conn.rollback()
            return f"Database insert error: {e}"

    def _embed_and_add_file_to_kw_bs(self, path: Path) -> str:
        """Embed file(s) and upload to knowledge base."""
        file_data = self._get_file_metadata(path)
        if not file_data:
            return f"Error reading file: File path '{path}' does not exist"

        cont = self.parsers._read_file(path)
        cont, embeddings = self._embedding_content(cont)
        if not embeddings:
            return "Failed to generate vector embedding: Failed to save entry"

        filename, mime, size = file_data
        return self._add_file_to_kw_bs(filename, embeddings, cont, mime, size)

    def _empty_sess_doc_kw_bs(self):
        """Clear all document contents related to current session."""
        try:
            self.cur.execute(
                """
                DELETE FROM knowledge_base
                WHERE session_id = %s AND type = %s
                """,
                (self.chat_logs.sess_id, "file_upload")
            )
            del_count = self.cur.rowcount
            self.conn.commit()
            if del_count == 0:
                return f"Failed to clear session '{self.sess_name}' knowledge base: Session knowledge base is empty"
            return f"Emptied session '{self.sess_name}' knowledge base"

        except Exception as e:
            self.conn.rollback()
            return f"Database deletion error: {e}"

    def _get_all_file_data(self) -> list[tuple[Any, Any]] | str:
        """Return a list of all uploaded file data in the database."""
        try:
            self.cur.execute(
                """
                SELECT created_at, metadata
                FROM knowledge_base
                WHERE session_id = %s AND type = %s
                """,
                (self.chat_logs.sess_id, "file_upload")
            )
            rows = self.cur.fetchall()
            if not rows:
                return f"Failed to retrieve filenames: Session '{self.sess_name}' knowledge base is empty"
            return rows

        except Exception as e:
            self.conn.rollback()
            return f"Database query filedata error: {e}"

    def _get_all_filenames(self) -> list[str]:
        """Return a list of all uploaded filenames in the database."""
        rows = self._get_all_file_data()
        if isinstance(rows, str):
            return []

        # Unpack data in metadata
        filenames = []
        for _, metadata in rows:
            data_dict = json.loads(metadata)
            filenames.append(data_dict["filename"])
        return filenames

    def list_all_uploaded_files(self) -> str:
        """Return a list of all files in the database."""
        rows = self._get_all_file_data()
        if isinstance(rows, str):
            return rows

        # Unpack data in metadata
        lines = [
            f"UPLOADED FILES(S)\n"
            f"=================\n"
            f"UPLOADED AT\t\t\t\tFILENAME"
        ]
        for time, metadata in rows:
            data_dict = json.loads(metadata)
            filename = data_dict.get("filename", "Unknown")
            lines.append(f"{str(time)}\t{filename}")
        return "\n".join(lines)

    # ==================================================
    # QUERY KNOWNLEDGE BASE
    # ==================================================

    def query_similar_knowledge(self, qry: str) -> list[dict[str, Any]] | str:
        """Queries knowledge base for similar content."""
        kw_dict = []

        qry, qry_embeddings = self._embedding_content(qry)
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
        for cont, metadata, score in rows:
            data_dict = json.loads(metadata)
            kw_dict.append({
                "filename": data_dict.get("filename", "Unknown"),
                "content": cont,
                "similarity": score
            })
        return kw_dict if kw_dict else "Error: No content retrieved from knowledge base"

    # ==================================================
    # AUTO FUNCTIONS
    # ==================================================

    def toggle_auto_retrive_from_kw_bs(
        self,
        enable_auto_read_dropbox: bool,
        prompt: str,
    ) -> list[dict[str, Any]] | str | None:
        """Auto fetches previous file contents ability, return relevant files if its toggled on."""
        if enable_auto_read_dropbox:
            kw_dict = self.query_similar_knowledge(prompt)
        return
