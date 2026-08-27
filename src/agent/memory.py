import psycopg2
import json
import uuid
import ollama
import hashlib
import re
from psycopg2 import sql
from pathlib import Path
from datetime import datetime
from typing import Literal, Any, get_args

from src.config.models import MODEL
from src.config.prompts import MEM_PROMPT, MEM_MANUAL_PROMPT
from src.config.memory import RETRIEVE_MEM_ENTRY_LIMIT, AUTO_MEMORY_STORE_TOKENS
from src.config.postgres import conn
from src.agent.chat_logs import ChatLogs
from src.agent.models.embed import Embed
from src.agent.models.llm import LLM
from src.agent.models.embed import Embed
from src.agent.tokens_handler import Tokens
from src.models_database import EMB_MODEL_DIMENSION
from src.logger import get_logger


log = get_logger(__name__)

CATEGORY_TYPES = Literal["preference", "stack", "fact", "project", "instruction", "correction"]

CATEGORIES = list(get_args(CATEGORY_TYPES))


class Memory:
    def __init__(
        self,
        conn,
        chat_logs: ChatLogs,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.conn               = conn
        self.cur                = self.conn.cursor()
        self.sess_name          = sess_name
        self.project            = project
        self.model              = MODEL
        self.mem_prompt         = MEM_PROMPT
        self.mem_manual_prompt  = MEM_MANUAL_PROMPT
        self.qry_limit          = RETRIEVE_MEM_ENTRY_LIMIT
        self.chat_logs          = chat_logs

        self.embed      = Embed()
        self.emb_dim    = self.embed.emb_dim
        self.model_tkns = Tokens(self.model)

        self._init_memory_db()


    # ============================================================
    # INITIALISE MEMORY DATABASE
    # ============================================================

    def _init_memory_db(self):
        """Create memory table if missing."""
        log.debug("Initialising vector extension for PostgreSQL")
        self.cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            """
        )

        log.debug("Initialising table 'memory' with vector embedding dimension of %s", self.emb_dim)
        create_mem_tbl = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id              BIGSERIAL PRIMARY KEY,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                embedding       VECTOR({dimension}),
                content         TEXT NOT NULL,
                content_hash    TEXT,
                category        TEXT NOT NULL,
                extraction      VARCHAR(20) NOT NULL CHECK (extraction IN ('manual', 'auto'))
            );
            """
        ).format(dimension=sql.SQL(str(int(self.emb_dim))))
        self.cur.execute(create_mem_tbl)

        # HNSW index - must match the distance operator used in queries
        log.debug("Initialising index 'idx_memory_embedding' on table 'memory' using HNSW")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_embedding
            ON memory USING hnsw (embedding vector_cosine_ops);
            """
        )

        # Index for category lookup
        log.debug("Initialising index 'idx_memory_category' on table 'memory'")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_category
            ON memory (category);
            """
        )

        # Unique constraint on content_hash to enforce dedupe at database level
        log.debug("Initialising index 'idx_memory_content_hash' on table 'memory'")
        self.cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_content_hash
            ON memory (content_hash);
            """
        )

        self.conn.commit()


    # ============================================================
    # EDIT MEMORY LOGS
    # ============================================================

    def _add_mem_embeddings(
        self,
        embeddings: list[float],
        cont: str,
        ctgry: str,
        extraction: Literal["manual", "auto"]
    ) -> str | None:
        """Add new memory entry."""
        try:
            self.cur.execute(
                """
                INSERT INTO memory_embeddings (embedding, content, category, extraction)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (str(embeddings), cont, ctgry, extraction)
            )
            row = self.cur.fetchone()
            print(row)
            self.conn.commit()

            if row:
                mem_id = str(row[0])
                log.info("New memory entry saved to memory: id=%s", mem_id)
                return mem_id
            else:
                log.warning("Failed to save memory entry to memory")
                return

        except Exception as e:
            self.conn.rollback()
            print(f"Database insert error: {e}")
            return


    def _embed_content_and_add_mem(
        self,
        cont: str,
        ctgry: str,
        extraction: Literal["manual", "auto"]
    ) -> str | None:
        """Embeds texts and adds to memory logs."""
        cont, embeddings = self.embed.embedding_content(cont)
        if not embeddings:
            log.warning("Failed to generate vector embedding. No memory entry saved")
            return

        mem_id = self._add_mem_embeddings(embeddings, cont, ctgry, extraction)
        if not mem_id:
            log.warning("Failed to retrieve memory embedding id. No memory entry saved")
            return
        return mem_id


    def delete_mem(self, ids: list[str]):
        """Removes a list vector ID reference key directly from database."""
        del_count = 0
        for id in ids:
            self.cur.execute(
                """
                DELETE FROM memory_embeddings
                WHERE id = %s;
                """,
                (id,)
            )
            del_count += 1
            self.conn.commit()

        count = len(ids) - del_count
        if not count == 0:
            log.warning("Failed to delete %d memory entry(s)", count)
            return f"Failed to delete {count} memory entry(s)"
        log.info("%d memory entry(s) deleted", del_count)
        return f"{del_count} memory entry(s) deleted"


    # ============================================================
    # QUERY MEMORY EMBEDDINGS
    # ============================================================

    def get_mem_content_from_ids(self, ids: list[str]) -> dict[str, str]:
        """Return memory dictionary from a list of ids."""
        mem_dict = {}
        for id in ids:
            self.cur.execute(
                """
                SELECT content, category
                FROM memory_embeddings
                WHERE id = %s;
                """,
                (id,)
            )
            row = self.cur.fetchone()

            if not row:
                log.warning("Failed to retrieve memory entry: Memory '%s' does not exist", id)
                continue
            mem_dict[str(row[0])] = str(row[1])

        log.info("%d memory entries retrieved", len(mem_dict))
        return mem_dict


    def query_similar_content(self, qry: str) -> list[dict[str, Any]] | None:
        """Queries database for similar content."""
        qry, qry_embeddings = self.embed.embedding_content(qry)

        self.cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s) AS cosine_similarity
            FROM memory_embeddings
            ORDER BY embedding <=> %s ASC
            LIMIT %s;
            """,
            (str(qry_embeddings), str(qry_embeddings), self.qry_limit)
        )
        rows = self.cur.fetchall()

        if rows:
            log.info("%d memory entry(s) retrieved", len(rows))
            return [
                {
                    "content": row[0],
                    "similarity": float(row[1])
                } for row in rows
            ]
        else:
            log.info("Failed to retrieve memory entry: No entry exists in memory")
            return


    # ============================================================
    # EXTRACT MEMORY
    # ============================================================

    def _format_extracted_mem(
        self,
        ext_out: str,
        extraction: Literal["manual", "auto"]
    ) -> dict[str, str]:
        """
        Format every other memory entry to the next line, trims
        out unnecessary spaces and symbols.

        Group 1 = category name
        Group 2 = everything after the brackets
        """
        mem_dict = {}

        # Slice model output into line-by-line format
        for line in ext_out.split("\n"):
            new_line = line.strip().lstrip("-*• ")

            match = re.search(r"\[([a-zA-Z\s_/]+)\]\s*(.*)", new_line)
            if match:
                ctgry_tag   = match.group(1).strip().lower()
                cont        = match.group(2).strip()

                if not cont:
                    continue

                ctgry = ctgry_tag if ctgry_tag in CATEGORIES else "fact"
                mem_dict[cont] = ctgry

        return mem_dict


    def _format_and_add_to_mem(
        self,
        ext_out: str,
        extraction: Literal["manual", "auto"]
    ) -> list[str]:
        """
        Format memory entry(s) and add to memory. Return a list
        of created database ID(s).
        """
        created_ids = []

        mem_dict = self._format_extracted_mem(ext_out, extraction)
        for cont, ctgry in mem_dict.items():

            mem_id = self._embed_content_and_add_mem(cont, ctgry, extraction)
            if mem_id:
                created_ids.append(mem_id)

        return created_ids


    def extract_and_store_mem_from_conv(
        self,
        extraction: Literal["manual", "auto"],
        prompt: str | None = None
    ) -> tuple[list[str], int, int] | None:
        """
        No prompts needed, process conversation logs,
        extracts standalone atomic facts via LLM, 
        and saved to database.

        Depends of the system prompt to decide whether to
        extract memory automatically or manually.
        """
        if extraction == "manual":
            log.debug(
                "Manual memory extraction system prompt is implemented for model '%s'",
                self.model
            )
            system_prompt = self.mem_manual_prompt # User ask model with prompt
        else:
            log.debug(
                "Auto memory extraction system prompt is implemented for model '%s'",
                self.model
            )
            system_prompt = self.mem_prompt # Extract memory automatically

        try:
            # Extract memory
            old_convs = self.chat_logs.get_old_convs()
            new_conv = self.chat_logs.get_latest_conv_turn() if prompt is None else prompt
            # ///////////////////////////////////////////////////////////////////
            # UPDATE REQUIRED FOR NEW FORMATTING
            fmt_prompt = (
                f"# All previous conversation(s)\n\n"
                f"{old_convs}\n\n"
                f"---\n\n"
                f"# New conversation\n\n"
                f"{new_conv}"
            )
            # ///////////////////////////////////////////////////////////////////
            ext_out, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
                model=self.model,
                system_prompt=system_prompt,
                prompt=fmt_prompt
            )
            created_ids = self._format_and_add_to_mem(ext_out, extraction)
            log.info("%d memory entry(s) extracted and saved", len(created_ids))
            print(f"Memory saved")
            return created_ids, p_tkns, o_tkns

        except Exception as e:
            log.error("Memory extraction synthesis failed: %s", e, exc_info=True)
            return


    # ============================================================
    # AUTO FUNCTIONS
    # ============================================================

    def toggle_auto_retrive_memory_entries(
        self,
        is_auto_mem_rtve: bool,
        prompt: str
    ) -> list[dict[str, Any]] | None:
        """Auto memory entry ability, returns memory entries if its toggled on."""
        if is_auto_mem_rtve:
            log.debug("Auto memory retrieve on. Querying memory for similar content")
            return self.query_similar_content(prompt)
        return


    # //////////////////////////////////////////////////////////////
    # FUNCTION INCOMPLETE
    def toggle_auto_store_memory_entries(
        self,
        enable_auto_memory_store: bool,
        model_max_tokens: int,
        context: list[dict],
    ):
        """
        Auto store memory no prompts needed, 
        should be at the end of every conversations.
        """
        if enable_auto_memory_store == False:
            return

        if model_max_tokens < AUTO_MEMORY_STORE_TOKENS:
            return

        response = self.extract_and_store_mem_from_conv(extraction= "auto")
        if response:
            created_ids, p_tkns, o_tkns = response
    # //////////////////////////////////////////////////////////////
