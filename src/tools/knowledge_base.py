import psycopg2
import ollama
import json
import mimetypes
from psycopg2 import sql
from pathlib import Path
from typing import Any, Literal

from src.agent.chat_logs import ChatLogs
from src.agent.models.embed import Embed
from src.tools.parsers import Parsers
from src.models_database import EMB_MODEL_DIMENSION
from src.tools.document_knowledge_base import DocumentKnowledgeBase
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

        self.sess_name  = sess_name

        self.chat_logs  = ChatLogs(sess_name=self.sess_name)
        self.sess_id    = self.chat_logs.sess_id
        self.embed      = Embed()
        self.emb_dim    = self.embed.emb_dim
        self.parsers    = Parsers()
        self.doc_kw_bs  = DocumentKnowledgeBase(sess_name=self.sess_name)

        self._init_doc_db()
        self.doc_names = self.doc_kw_bs.doc_names

    def _init_doc_db(self):
        """Create knowledge base table if missing."""
        self.cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS VECTOR;
            """
        )

        create_kw_bs_tbl = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id              BIGSERIAL PRIMARY KEY,
                session_id      VARCHAR(255) NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at      TIMESTAMPTZ,
                type            VARCHAR(20) NOT NULL,
                embedding       VECTOR({dimension}),
                content         TEXT NOT NULL,
                content_hash    TEXT,
                metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
            """
        ).format(dimension=sql.SQL(str(int(self.emb_dim))))
        self.cur.execute(create_kw_bs_tbl)
        # NOTE: 
        # - Expires_at and content_hash is only applicable for web search contents.

        # HNSW index - must match the distance operator used in queries
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kw_bs_embedding
            ON knowledge_base USING hnsw (embedding vector_cosine_ops);
            """
        )

        # Index for session id and knowledge type lookups
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kw_bs_session_type
            ON knowledge_base (session_id, type);
            """
        )

        # Index for expires at lookups
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kw_bs_expires
            ON knowledge_base (expires_at) WHERE expires_at IS NOT NULL;
            """
        )

        self.conn.commit()

    # ================================================
    # LIST CONTENTS
    # ================================================

    def list_all_uploaded_documents(self) -> str:
        """Return a list of all document(s) in the database."""
        rows = self.doc_kw_bs.doc_metadata
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
                (self.sess_id, typ)
            )
            del_count = self.cur.rowcount
            self.conn.commit()
            if del_count == 0:
                return f"Failed to remove session '{self.sess_name}' knowledge base: '{typ}' contents not found in database"
            return f"Emptied session '{self.sess_name}' knowledge base: {typ}"

        except Exception as e:
            self.conn.rollback()
            return f"Database deletion error: {e}"

