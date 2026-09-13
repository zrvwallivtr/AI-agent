from logging import debug
import json
import mimetypes
from psycopg2 import sql
from pathlib import Path
from typing import Any, Literal

from src.config import models

from src.agent import chat_logs
from src.models_database import EMB_MODEL_DIMENSION
from src.rag.documents.document_knowledge_base import DocumentKnowledgeBase
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

EMBED_MODEL = models.EMBED_MODEL
ChatLogs    = chat_logs.ChatLogs


class KnowledgeBase:
    def __init__(
        self,
        conn,
        chat_logs: ChatLogs,
        sess_name: str | None = None,
    ):
        self.conn   = conn
        self.cur    = self.conn.cursor()

        self.sess_name  = sess_name

        self.chat_logs  = chat_logs

        self.doc_kw_bs  = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )

        self._init_kw_bs()
        self.doc_names = self.doc_kw_bs.get_all_docs_names


    # ================================================
    # INITIALISE KNOWLEDGE BASE
    # ================================================

    def _init_kw_bs(self):
        """Create knowledge base table if missing."""
        app_log.debug("Initialising vector extension for PostgreSQL")
        self.cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS VECTOR;
            """
        )

        app_log.debug(
            "Initialising table 'knowledge_base' with vector embedding dimension of %s",
            EMB_MODEL_DIMENSION[EMBED_MODEL]
        )
        create_kw_bs_tbl = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id              BIGSERIAL PRIMARY KEY,
                session_id      VARCHAR(255) NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at      TIMESTAMPTZ,    -- Only applicable for web search contents
                type            VARCHAR(20) NOT NULL,
                embeddings      VECTOR({dimension}),
                prompt_tokens   INTEGER NOT NULL,
                content         TEXT NOT NULL,
                content_hash    TEXT,   -- Only applicable for web search contents
                metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
            """
        ).format(dimension=sql.SQL(str(int(EMB_MODEL_DIMENSION[EMBED_MODEL]))))
        self.cur.execute(create_kw_bs_tbl)

        # HNSW index - must match the distance operator used in queries
        app_log.debug("Initialising index 'idx_kw_bs_embeddings' on table 'knowledge_base' using HNSW")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kw_bs_embeddings
            ON knowledge_base USING hnsw (embeddings vector_cosine_ops);
            """
        )

        # Index for session id and knowledge type lookups
        app_log.debug("Initialising index 'idx_kw_bs_session_type' on table 'knowledge_base'")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kw_bs_session_type
            ON knowledge_base (session_id, type);
            """
        )

        # Index for expires at lookups
        app_log.debug("Initialising index 'idx_kw_bs_expires' on table 'knowledge_base'")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kw_bs_expires
            ON knowledge_base (expires_at) WHERE expires_at IS NOT NULL;
            """
        )

        self.conn.commit()


    # ==================================================
    # CLEAR CONTENTS
    # ==================================================

    def clear_sess_kw_bs(self, typ: Literal["document", "web_search"]) -> None:
        """Clear all contents related to specified type and current session."""
        app_log.info("Clearing session '%s' knowledge base", self.sess_name)
        try:
            self.cur.execute(
                """
                DELETE FROM knowledge_base
                WHERE session_id = %s AND type = %s
                """,
                (self.chat_logs.get_sess_id, typ)
            )
            del_count = self.cur.rowcount
            self.conn.commit()

            if del_count == 0:
                app_log.warning(
                    "Failed to clear session '%s' knowledge base: '%s' contents not found in database",
                    self.sess_name,
                    typ
                )
                return

            app_log.info(
                "Cleared all %s contents in sesssion '%s' knowledge base",
                typ,
                self.sess_name
            )
            return

        except Exception as e:
            self.conn.rollback()
            app_log.warning("Database deletion error: %s", e)
            return
