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

from src import config
from src.agent.chat_logs import ChatLogs
from src.agent.models.llm import LLM
from src.agent.models.embed import Embed
from src.agent.tokens_handler import Tokens
from src.models_database import EMB_MODEL_DIMENSION
from src.logger import get_logger


logger = get_logger(__name__)

CATEGORY_TYPES = Literal["preference", "stack", "fact", "project", "instruction", "correction"]

CATEGORIES = list(get_args(CATEGORY_TYPES))


class Memory:
    def __init__(
        self,
        sess_name: str | None = None,
        project: str | None = None
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
        self.project            = project
        self.model              = config.MODEL
        self.emb_model          = config.EMBED_MODEL
        self.emb_dmsion         = EMB_MODEL_DIMENSION[self.emb_model]
        self.mem_prompt         = config.MEM_PROMPT
        self.mem_manual_prompt  = config.MEM_MANUAL_PROMPT
        self.qry_limit          = config.RETRIEVE_MEM_ENTRY_LIMIT
        self._init_memory_db()

        self.chat_logs          = ChatLogs(sess_name=self.sess_name)
        self.model_tkns         = Tokens(self.model)
        self.emb_model_tkns     = Tokens(self.emb_model)

    def _init_memory_db(self):
        """Create memory table if missing."""
        self.cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            """
        )

        create_mem_emb_tbl = sql.SQL(
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
        ).format(dimension=sql.SQL(str(int(self.emb_dmsion))))
        self.cur.execute(create_mem_emb_tbl)

        # HNSW index - must match the distance operator used in queries
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_embedding
            ON memory USING hnsw (embedding vector_cosine_ops);
            """
        )

        # Index for category lookup
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_category
            ON memory (category);
            """
        )

        # Unique constraint on content_hash to enforce dedupe at database level
        self.cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_content_hash
            ON memory (content_hash);
            """
        )

        self.conn.commit()

    # ============================================================
    # EMBEDDING MEMORY CONTENT
    # ============================================================

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
            return str(row[0]) if row else None

        except Exception as e:
            self.conn.rollback()
            print(f"Database insert error: {e}")
            return

    def _embed_content_and_add_mem(
        self,
        cont: str,
        ctgry: str,
        extraction: Literal["manual", "auto"]
    ) -> str:
        """Embeds texts and adds to memory logs."""
        cont, embeddings = self._embedding_content(cont)
        if not embeddings:
            return "Failed to generate vector embedding: Failed to save entry"

        mem_id = self._add_mem_embeddings(embeddings, cont, ctgry, extraction)
        if not mem_id:
            return "Failed to retrieve memory embedding id: Failed to save entry"
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
            return f"Warning: Unable to delete {count} entry(s)"
        return f"Memory entry(s) deleted"

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
                continue
            mem_dict[str(row[0])] = str(row[1])
        return mem_dict

    def query_similar_content(self, qry: str) -> list[dict[str, Any]] | None:
        """Queries database for similar content."""
        mem_dict = {}

        qry, qry_embeddings = self._embedding_content(qry)
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
        return [
            {
                "content": row[0],
                "similarity": float(row[1])
            } for row in rows
        ] if rows else None

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
    ) -> tuple[list[str], int, int]:
        """
        No prompts needed, process conversation logs,
        extracts standalone atomic facts via LLM, 
        and saved to database.

        Depends of the system prompt to decide whether to
        extract memory automatically or manually.
        """
        if extraction == "manual":
            system_prompt = self.mem_manual_prompt # User ask model with prompt
        else:
            system_prompt = self.mem_prompt # Extract memory automatically

        try:
            # Extract memory
            old_convs = self.chat_logs.get_old_convs()
            new_conv = self.chat_logs.get_latest_conv_turn() if prompt is None else prompt
            fmt_prompt = (
                f"# All previous conversation(s)\n\n"
                f"{old_convs}\n\n"
                f"---\n\n"
                f"# New conversation\n\n"
                f"{new_conv}"
            )
            ext_out, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
                model=self.model,
                system_prompt=system_prompt,
                prompt=fmt_prompt
            )
            created_ids = self._format_and_add_to_mem(ext_out, extraction)
            if created_ids:
                logger.info("Extracted and saved %d memory entry(s)", len(created_ids))
                print(f"Memory saved")
            return created_ids, p_tkns, o_tkns

        except Exception as e:
            logger.error("Memory extraction synthesis failed: error=%s", e, exc_info=True)
            return [], 0, 0

    # ============================================================
    # AUTO FUNCTIONS
    # ============================================================

    def toggle_auto_retrive_memory_entries(
        self,
        is_auto_mem_rtve: bool,
        prompt: str
    ) -> list[dict[str, Any]] | None:
        """Auto memory entry ability, returns memory entries if its toggled on."""
        if is_auto_mem_rtve == True:
            return self.query_similar_content(prompt)
        return

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

        if model_max_tokens < config.AUTO_MEMORY_STORE_TOKENS:
            return

        created_ids, p_tkns, o_tkns = self.extract_and_store_mem_from_conv(extraction= "auto")
