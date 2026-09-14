import psycopg2
import json
import uuid
import hashlib
import re
from psycopg2 import sql
from pathlib import Path
from datetime import datetime
from typing import Literal, Any, get_args

from src.config import models
from src.config import prompts
from src.config import memory
from src.config import postgres

from src import format_context
from src.agent import chat_logs, llm, embed
from src.models_database import EMB_MODEL_DIMENSION
from src.logger import app_logger, prompt_logger


app_log     = app_logger(f"{__name__}.app")
prompt_log  = prompt_logger(f"{__name__}.prompt")

MEM_MODEL                   = models.MEM_MODEL
EMBED_MODEL                 = models.EMBED_MODEL
MEM_PROMPT                  = prompts.MEM_PROMPT
MEM_MANUAL_PROMPT           = prompts.MEM_MANUAL_PROMPT
RETRIEVE_MEM_ENTRY_LIMIT    = memory.RETRIEVE_MEM_ENTRY_LIMIT
AUTO_MEMORY_STORE_TOKENS    = memory.AUTO_MEMORY_STORE_TOKENS

CATEGORY_TYPES = Literal[
    "preference",
    "stack",
    "fact",
    "project",
    "instruction",
    "correction"
]
CATEGORIES = list(get_args(CATEGORY_TYPES))

ChatLogs = chat_logs.ChatLogs


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
        self.model              = MEM_MODEL
        self.mem_prompt         = MEM_PROMPT
        self.mem_manual_prompt  = MEM_MANUAL_PROMPT
        self.qry_limit          = RETRIEVE_MEM_ENTRY_LIMIT
        self.chat_logs          = chat_logs
        self._init_memory_db()


    # ============================================================
    # MEMORY HASH
    # ============================================================

    def _hash_memory(self, cont: str) -> str:
        """
        Return a SHA-256 hash of raw text content,
        used for dedupe before embedding.
        """
        app_log.debug("Hashing text from given content")
        return hashlib.sha256(cont.encode("utf-8")).hexdigest()


    def _is_mem_exist(self, new_hash: str) -> bool:
        """
        Check if the same memory was added before (same
        hash, accross sessions). Return True
        if a duplicate exist.
        """
        app_log.debug("Verifing if memory entry already exist")
        self.cur.execute(
            """
            SELECT id FROM memory WHERE content_hash = %s
            """,
            (new_hash,)
        )
        existing = self.cur.fetchone()
        if not existing:
            app_log.debug("Memory entry does not exist")
            return False
        app_log.debug("Memory entry already exist")
        return True


    # ============================================================
    # INITIALISE MEMORY DATABASE
    # ============================================================

    def _init_memory_db(self):
        """Create memory table if missing."""
        app_log.debug("Initialising vector extension for PostgreSQL")
        self.cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            """
        )

        app_log.debug(
            "Initialising table 'memory' with vector embedding dimension of %s",
            EMB_MODEL_DIMENSION[EMBED_MODEL]
        )
        create_mem_tbl = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id              BIGSERIAL PRIMARY KEY,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                embeddings      VECTOR({dimension}),
                prompt_tokens   INTEGER NOT NULL,
                content         TEXT NOT NULL,
                content_hash    TEXT,
                category        TEXT NOT NULL,
                extraction      VARCHAR(20) NOT NULL CHECK (extraction IN ('manual', 'auto'))
            );
            """
        ).format(dimension=sql.SQL(str(int(EMB_MODEL_DIMENSION[EMBED_MODEL]))))
        self.cur.execute(create_mem_tbl)

        # HNSW index - must match the distance operator used in queries
        app_log.debug("Initialising index 'idx_memory_embedding' on table 'memory' using HNSW")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_embedding
            ON memory USING hnsw (embeddings vector_cosine_ops);
            """
        )

        # Index for category lookup
        app_log.debug("Initialising index 'idx_memory_category' on table 'memory'")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_category
            ON memory (category);
            """
        )

        # Unique constraint on content_hash to enforce dedupe at database level
        app_log.debug("Initialising index 'idx_memory_content_hash' on table 'memory'")
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
        embdings: list[float],
        cont_tkns: int,
        cont: str,
        cont_hash: str,
        ctgry: str,
        extraction: Literal["manual", "auto"]
    ) -> str | None:
        """Add new memory entry."""
        app_log.debug("Adding new %s extracted memory", extraction)
        try:
            self.cur.execute(
                """
                INSERT INTO memory (embeddings, prompt_tokens, content, content_hash, category, extraction)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (str(embdings), cont_tkns, cont, cont_hash, ctgry, extraction)
            )
            row = self.cur.fetchone()
            self.conn.commit()

            if not row:
                app_log.warning("Failed to save memory entry to memory")
                return
            mem_id = str(row[0])
            app_log.info("New memory entry (id = %s) saved to memory", mem_id)
            return mem_id

        except Exception as e:
            self.conn.rollback()
            app_log.warning("Database insert error: %s", e)
            return


    def _embed_content_and_add_mem(
        self,
        cont: str,
        ctgry: str,
        extraction: Literal["manual", "auto"]
    ) -> tuple[str, int] | None:
        """Embeds texts and adds to memory logs."""
        cont_hash = self._hash_memory(cont)
        if self._is_mem_exist(cont_hash):
            app_log.debug("Skipping memory upload")
            return

        response = embed.embedding_content(cont)
        if not response:
            return
        cont, embdings, cont_tkns = response

        cont_hash = self._hash_memory(cont)

        mem_id = self._add_mem_embeddings(
            embdings=embdings,
            cont_tkns=cont_tkns,
            cont=cont,
            cont_hash=cont_hash,
            ctgry=ctgry,
            extraction=extraction
        )
        if not mem_id:
            return
        return mem_id, cont_tkns


    def delete_mem(self, ids: list[str]) -> None:
        """Removes a list vector ID reference key directly from database."""
        app_log.info("Deleting %d memory entry(s)", len(ids))
        del_count = 0
        for id in ids:
            del_count += 1
            app_log.debug(
                "[%d/%d] Deleting memory entry (%s)",
                del_count,
                len(ids),
                id
            )
            self.cur.execute(
                """
                DELETE FROM memory
                WHERE id = %s;
                """,
                (id,)
            )
            self.conn.commit()

        count = len(ids) - del_count
        if not count == 0:
            app_log.warning("Failed to delete %d memory entry(s)", count)
            return

        app_log.info("%d memory entry(s) deleted", del_count)
        return


    # ============================================================
    # QUERY MEMORY EMBEDDINGS
    # ============================================================

    def get_mem_content_from_ids(self, ids: list[str]) -> dict[str, str]:
        """Return memory dictionary from a list of ids."""
        app_log.debug("Fetching %d memory entry(s) from memory ids", len(ids))
        mem_dict = {}
        for id in ids:
            self.cur.execute(
                """
                SELECT content, category
                FROM memory
                WHERE id = %s;
                """,
                (id,)
            )
            row = self.cur.fetchone()

            if not row:
                app_log.warning(
                    "Failed to retrieve memory entry '%s': Memory does not exist. Skipping",
                    id
                )
                continue

            mem_dict[str(row[0])] = str(row[1])

        app_log.debug("%d memory entries retrieved", len(mem_dict))
        return mem_dict


    def query_similar_content(
        self,
        qry: str,
        qry_embdings: list[float],
        min_sim: float = 0.65
    ) -> list[dict[str, Any]] | None:
        """Queries database for similar content."""
        app_log.debug(
            "Searching for similar entries in memory: Minimum similarity score = %d",
            min_sim
        )
        self.cur.execute(
            """
            SELECT content, 1 - (embeddings <=> %s) AS cosine_similarity
            FROM memory
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

        if not rows:
            app_log.debug("Failed to retrieve relevant memory: No entry exists in memory or Entry does not reached minimum similarity score")
            return

        app_log.debug("%d memory entry(s) retrieved", len(rows))
        return [
            {
                "content": row[0],
                "similarity": float(row[1])
            } for row in rows
        ]


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
        app_log.debug(
            "Formatting %s extracted memory with category label",
            extraction
        )
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


    def _add_formatted_to_mem(
        self,
        ext_out: str,
        extraction: Literal["manual", "auto"]
    ) -> tuple[list[str], int]:
        """
        Format memory entry(s) and add to memory. Return a list
        of created database ID(s).
        """
        created_ids = []
        tol_tkns = 0
        count = 0

        mem_dict = self._format_extracted_mem(ext_out, extraction)
        app_log.debug(
            "%d memory entry(s) was extracted from user prompt",
            len(mem_dict)
        )

        for cont, ctgry in mem_dict.items():
            count += 1
            app_log.debug("Processing memory entry (%d/%d)", count, len(mem_dict))
            response = self._embed_content_and_add_mem(cont, ctgry, extraction)
            if not response:
                continue
            mem_id, tkn_used = response
            created_ids.append(mem_id)
            tol_tkns += tol_tkns

        app_log.debug("%d tokens used for embedding memory entry(s)", tol_tkns)
        return created_ids, tol_tkns


    def extract_and_store_mem_from_conv(
        self,
        extraction: Literal["manual", "auto"],
        prompt: str | None = None
    ) -> tuple[list[str], int, int, int] | None:
        """
        Depends of the system prompt to decide whether to
        extract memory automatically or manually.
        NOTE: Prompt must be provided for manual extraction.
        """
        # === MANUAL MEMORY EXTRACTION =======================================
        if extraction == "manual":
            app_log.debug(
                "Manual memory extraction using model '%s' is triggered",
                self.model
            )
            system_prompt = self.mem_manual_prompt
            if not prompt:
                app_log.warning("Memory extraction failed: No prompt provided")
                return
            new_convs = [llm.user_message(prompt)]

        # === AUTO MEMORY EXTRACTION =========================================
        else:
            app_log.debug(
                "Auto memory extraction using model '%s' is triggered",
                self.model
            )
            system_prompt = self.mem_prompt
            new_convs = self.chat_logs.get_latest_conv_turn()

        # === FORMAT CONTENT AND EXTRACT MEMORY ==============================
        try:
            old_convs = self.chat_logs.get_old_convs()
            fmt_prompt = format_context.old_and_new_convs(
                old_convs=old_convs, new_convs=new_convs
            )

            response = llm.response_with_new_sys_prompt_and_context(
                model=self.model, sys_prompt=system_prompt, prompt=fmt_prompt
            )
            if not response:
                return
            ext_out, p_tkns, o_tkns = response

            created_ids, total_tkn_used = self._add_formatted_to_mem(ext_out, extraction)
            return created_ids, p_tkns, o_tkns, total_tkn_used

        except Exception as e:
            app_log.warning("Memory extraction synthesis failed: %s", e, exc_info=True)
            return


    # ============================================================
    # AUTO FUNCTIONS
    # ============================================================

    def toggle_auto_retrive_memory_entries(
        self,
        is_auto_mem_rtve: bool,
        prompt: str,
        prompt_embdings: list[float]
    ) -> list[dict[str, Any]] | None:
        """Auto memory entry ability, returns memory entries if its toggled on."""
        if not is_auto_mem_rtve:
            return
        app_log.debug("Auto memory retrieve is currently on")
        return self.query_similar_content(prompt, prompt_embdings)


    # //////////////////////////////////////////////////////////////
    # FUNCTION INCOMPLETE
    def toggle_auto_store_memory_entries(
        self,
        enable_auto_memory_store: bool,
        model_max_tokens: int,
        contxt: list[dict],
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
            created_ids, p_tkns, o_tkns, total_tkn_used = response
    # //////////////////////////////////////////////////////////////
