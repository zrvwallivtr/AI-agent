import psycopg2
import ollama
import json
import mimetypes
from psycopg2 import sql
from pathlib import Path
from typing import Any, Literal

from src.agent.chat_logs import ChatLogs
from src.tools.parsers import Parsers
from src.models_database import EMB_MODEL_DIMENSION
from src.tools.documents import Document
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


class KnowledgeBase:
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
        self.emb_model          = config.EMBED_MODEL
        self.emb_dmsion         = EMB_MODEL_DIMENSION[self.emb_model]
        self.qry_limit          = config.RETRIEVE_MEM_ENTRY_LIMIT

        self.chat_logs          = ChatLogs(sess_name=self.sess_name)
        self.parsers            = Parsers()
        self.attchmnts          = Document(sess_name=self.sess_name)

        self._init_doc_db()
        self.doc_names          = self.attchmnts.doc_names

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
    # DOCUMENTS
    # ================================================

    def embed_and_add_doc_to_kw_bs(self, path: Path) -> str:
        """Embed document(s) and upload to knowledge base."""
        doc_data = self.attchmnts.get_document_metadata(path)
        if not doc_data:
            return f"Error reading '{path}': Path does not exist"

        cont = self.parsers._read_file(path)
        cont, embeddings = self._embedding_content(cont)
        if not embeddings:
            return "Failed to generate vector embedding: Failed to save entry"

        name, mime, size = doc_data
        return self.attchmnts.add_document_to_kw_bs(name, embeddings, cont, mime, size)

    def list_all_uploaded_documents(self) -> str:
        """Return a list of all document(s) in the database."""
        rows = self.attchmnts.doc_metadata
        if isinstance(rows, str):
            return rows

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
                "document_name": data_dict.get("document_name", "Unknown"),
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
        """Auto fetches previous documents contents ability, return relevant contents if its toggled on."""
        if enable_auto_read_dropbox:
            kw_dict = self.query_similar_knowledge(prompt)
        return

    # ==================================================
    # CLEAR CONTENTS
    # ==================================================

    def clear_sess_kw_bs(self, typ: Literal["document", "web_search"]):
        """Clear all contents related to specified type and current session."""
        try:
            self.cur.execute(
                """
                DELETE FROM knowledge_base
                WHERE session_id = %s AND type = %s
                """,
                (self.chat_logs.sess_id, typ)
            )
            del_count = self.cur.rowcount
            self.conn.commit()
            if del_count == 0:
                return f"Failed to remove session '{self.sess_name}' knowledge base: '{typ}' contents not found in database"
            return f"Emptied session '{self.sess_name}' knowledge base: {typ}"

        except Exception as e:
            self.conn.rollback()
            return f"Database deletion error: {e}"

