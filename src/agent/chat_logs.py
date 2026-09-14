import psycopg2
import mimetypes
import json
import uuid
from pathlib import Path
from typing import Literal, Any

from src.config import models
from src.config import prompts
from src.config import postgres

from src import format_context
from src.agent.models import llm
from src import logger


app_log = logger.app_logger(f"{__name__}.app")

MODEL           = models.MODEL
SYS_PROMPT      = prompts.SYS_PROMPT
COMPRESS_PROMPT = prompts.COMPRESS_PROMPT


class ChatLogs:
    def __init__(self, conn, sess_name: str | None = None):
        self.conn   = conn
        self.cur    = self.conn.cursor()

        self.model      = MODEL
        self.sys_prompt = SYS_PROMPT
        self.cmp_prompt = COMPRESS_PROMPT

        self.sess_name = sess_name.strip() if sess_name else "default_session"

        self._init_chat_logs_db()


    # =============================================================
    # INITIALISE CHAT LOGS DATABASE
    # =============================================================

    def _init_chat_logs_db(self):
        """Create session lookup and chat logs table if missing."""
        app_log.debug(
            "Initialising table 'chat_sessions' for session '%s'",
            self.sess_name
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_name    VARCHAR(255) NOT NULL UNIQUE,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        app_log.debug(
            "Initialising table 'chat_logs' for session '%s'",
            self.sess_name
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id                  BIGSERIAL PRIMARY KEY,
                session_id          UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                prompt              TEXT NOT NULL,
                response            TEXT NOT NULL,
                state               VARCHAR(20) NOT NULL CHECK (state IN ('external', 'internal')),
                total_prompt_tokens INTEGER NOT NULL,
                total_output_tokens INTEGER NOT NULL,
                is_compressed       BOOL NOT NULL DEFAULT FALSE,
                metadata            JSONB DEFAULT '{}'::jsonb
            );
            """
        )

        # Index for session id lookup
        app_log.debug("Initialising index 'idx_chat_logs_session_id' on table 'chat_logs'")
        self.cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id
            ON chat_logs (session_id);
            """
        )

        self.conn.commit()


    # =============================================================
    # SESSION ID
    # =============================================================

    def get_sess_id(self) -> str | None:
        """Fetch session id from chat_sessions table."""
        app_log.debug("Fetching session id for session '%s'", self.sess_name)
        self.cur.execute(
            """
            SELECT session_id
            FROM chat_sessions
            WHERE LOWER(session_name) = LOWER(%s);
            """,
            (self.sess_name,)
        )
        self.conn.commit()
        row = self.cur.fetchone()

        if not row:
            app_log.info(
                "Session id for session '%s' not found: Session does not exists",
                self.sess_name
            )
            return

        sess_id = str(row[0])
        app_log.debug("Retrieved session id '%s' for session '%s'", sess_id, self.sess_name)
        return sess_id



    def get_all_existing_sess_metadata(self) -> dict | None:
        """Fetch all session names from database."""
        app_log.debug("Fetching all session name(s) in the database")
        sess_dict = {}
        self.cur.execute(
            """
            SELECT session_id, session_name, created_at
            FROM chat_sessions
            """
        )
        self.conn.commit()
        rows = self.cur.fetchall()

        if not rows:
            app_log.debug("No existing session found on table 'chat_sessions' in the database")
            return

        for row in rows:
            sess_dict[str(row[0])] = {
                "session_name": str(row[1]),
                "created_at": str(row[2])
            }
        app_log.debug("%d session(s) found on the table 'chat_sessions' in the database", len(rows))
        return sess_dict


    def create_sess(self) -> str:
        """Create session entry on the chat_sessions table and return its session id."""
        app_log.debug("Creating new session '%s'", self.sess_name)
        self.cur.execute(
            """
            INSERT INTO chat_sessions
            (session_name) VALUES (%s)
            RETURNING session_id;
            """,
            (self.sess_name,)
        )
        self.conn.commit()
        row = self.cur.fetchone()

        if row is None:
            raise RuntimeError("Failed to create session: Database return no ID.")

        app_log.info(
            "New session '%s' created: New session entry is added onto the table 'chat_sessions' in the database", 
            self.sess_name
        )
        return str(row[0])


    def get_or_create_sess_id(self) -> str:
        """
        Fetch session id if session already exists,
        else create new session entry and return
        new generated session id.
        """
        sess_id = self.get_sess_id()

        if not sess_id:
            sess_id = self.create_sess()
        return sess_id


    # =============================================================
    # EDIT CHAT LOGS
    # =============================================================

    def add_conv_turn(
        self,
        prompt: str,
        response: str,
        state: Literal["external", "internal"],
        attchmnts: list[Path] | None = None,
        qry_wth_urls: list[dict[str, list[str]]] | None = None,
        p_tkns: int = 0,
        o_tkns: int = 0
    ):
        """
        Insert new conversation turn including metadata 
        into the specified session database table.

        State:
        - 'internal': Pre-written prompt.
        - 'external': User/model interactions.
        """
        app_log.debug(
            "Uploading the latest conversation turn to session '%s' caht log", self.sess_name
        )
        metadata = self._tool_calls_metadata(attchmnts, qry_wth_urls)

        self.cur.execute(
            """
            INSERT INTO chat_logs (session_id, prompt, response, state, total_prompt_tokens, total_output_tokens, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            (self.get_or_create_sess_id(), prompt, response, state, p_tkns, o_tkns, json.dumps(metadata or {}))
        )
        self.conn.commit()
        app_log.debug("New conversation turn added to session '%s' chat logs", self.sess_name)

        # Resync messages
        self.actv_convs = self.get_actv_convs()
        app_log.debug("Resynced session '%s' conversations", self.sess_name)


    def clear_sess_chat_logs(self) -> bool:
        """Clear all session related chat logs."""
        app_log.debug("Clearing chat logs for session '%s'", self.sess_name)
        self.cur.execute(
            """
            DELETE FROM chat_logs
            WHERE session_id = %s;
            """,
            (self.get_sess_id(),)
        )
        del_count = self.cur.rowcount
        self.conn.commit()

        if del_count == 0:
            app_log.warning(
                "Failed to clear chat logs: Session '%s' does not exists or has no chat logs",
                self.sess_name
            )
            return False

        app_log.info("Cleared all chat logs for session '%s'", self.sess_name)
        self.actv_convs = self.get_actv_convs() # resync messages
        app_log.debug("Resynced session '%s' conversations", self.sess_name)
        return True


    # =============================================================
    # FROM CHAT LOGS
    # =============================================================

    def get_actv_convs(self) -> list[dict]:
        """
        Get all messages in a session with filter options.
        If session does not exists, return system prompt.
        """
        app_log.debug("Fetching all active conversations from session '%s' chat logs", self.sess_name)

        sys_prompt = [{"role": "system", "content": SYS_PROMPT}]

        self.cur.execute(
            """
            SELECT prompt, response
            FROM chat_logs
            WHERE session_id = %s AND state = 'external' AND is_compressed = FALSE
            ORDER BY created_at ASC;
            """,
            (self.get_or_create_sess_id(),)
        )
        self.conn.commit()
        rows = self.cur.fetchall()

        if not rows:
            app_log.debug(
                "No conversation found in session '%s' chat logs. Returning system prompt only",
                self.sess_name
            )
            return sys_prompt

        convs = []
        for row in rows:
            convs.append(llm.user_message(row[0]))
            convs.append(llm.assistant_message(row[1]))
        app_log.debug(
            "%d conversation turns retrieved from session '%s' chat logs",
            len(rows),
            self.sess_name
        )
        return sys_prompt + convs


    def get_chat_history(self, filter: Literal["compressed", "not_compressed", "all"]) -> list[dict] | None:
        """
        Get all messages in a session with filter options.
        If session does not exists, return system prompt.
        """
        app_log.debug(
            "Fetching all conversation turns for session '%s' with the filter: %s", self.sess_name, filter
        )

        if filter == "compressed":
            self.cur.execute(
                """
                SELECT prompt, response
                FROM chat_logs
                WHERE session_id = %s AND state = 'external' AND is_compressed = TRUE
                ORDER BY created_at ASC;
                """,
                (self.get_sess_id(),)
            )
        elif filter == "not_compressed":
            self.cur.execute(
                """
                SELECT prompt, response
                FROM chat_logs
                WHERE session_id = %s AND state = 'external' AND is_compressed = FALSE
                ORDER BY created_at ASC;
                """,
                (self.get_sess_id(),)
            )
        else:
            self.cur.execute(
                """
                SELECT prompt, response
                FROM chat_logs
                WHERE session_id = %s AND state = 'external'
                ORDER BY created_at ASC;
                """,
                (self.get_sess_id(),)
            )

        self.conn.commit()
        rows = self.cur.fetchall()

        if not rows:
            app_log.debug(
                "Failed to retrieve '%s' conversation turns from session '%s' chat logs",
                filter,
                self.sess_name
            )
            return

        convs = []
        for row in rows:
            convs.append({"role": "user", "content": row[0]})
            convs.append({"role": "assistant", "content": row[1]})
        app_log.debug(
            "%d '%s' conversation turns retrieved from session '%s' chat logs",
            len(rows),
            filter,
            self.sess_name
        )
        return convs


    def get_latest_conv_turn(self) -> list[dict] | None:
        """Get the latest external user/assistant conversation turn from the chat log."""
        app_log.debug("Fetching the latest conversation turn from session '%s' chat logs", self.sess_name)
        self.cur.execute(
            """
            SELECT prompt, response
            FROM chat_logs
            WHERE session_id = %s AND state = 'external' AND is_compressed = FALSE
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (self.get_or_create_sess_id(),)
        )

        self.conn.commit()
        row = self.cur.fetchone()

        if not row:
            app_log.warning(
                "No conversation turns found in session '%s' chat logs: Session does not exists or contain no conversations",
                self.sess_name
            )
            return

        app_log.debug(
            "Retrieved latest conversation turn from session '%s' chat logs", self.sess_name
        )
        return [{"role": "user", "content": row[0]}, {"role": "assistant", "content": row[1]}]


    def get_old_convs(self) -> list[dict] | None:
        """
        Get all previous user/assistant conversation turns
        right before the latest external conversation from
        the chat log.
        """
        app_log.debug("Fetching old conversation turns from session '%s' chat logs", self.sess_name)
        self.cur.execute(
            """
            SELECT prompt, response
            FROM chat_logs
            WHERE session_id = %s
                AND id < (
                    SELECT MAX(id)
                    FROM chat_logs
                    WHERE session_id = %s AND state = 'external'
                )
            ORDER BY id ASC
            """,
            (self.get_or_create_sess_id(), self.get_or_create_sess_id())
        )
        self.conn.commit()
        rows = self.cur.fetchall()

        if not rows:
            app_log.debug(
                "No old conversation found in session '%s' chat logs: Session does not exists or contains no conversations",
                self.sess_name
            )
            return

        convs = []
        for row in rows:
            convs.append({"role": "user", "content": row[0]})
            convs.append({"role": "assistant", "content": row[1]})
        app_log.debug(
            "%d old conversation turn(s) retrived form session '%s' chat logs",
            len(rows),
            self.sess_name
        )
        return convs


    # =============================================================
    # METADATA
    # =============================================================

    def _attachments_metadata(
        self,
        attchmnts: list[Path] | None
    ) -> dict[str, dict[str, Any]] | None:
        """
        Add metadata to every filename in the list of filenames.

        {
            "filename": {
                "mime_type": "type",
                "size_bytes": size,
            }
        }
        """
        if not attchmnts:
            app_log.debug("No attachment uploaded. Skipping")
            return

        app_log.debug("Extracting metadata from %d attachment(s)", len(attchmnts))
        attchmnts_dict = {}
        count = 0
        for attchmnt in attchmnts:
            count += 1
            app_log.debug(
                "Extracting metadate from attachment '%s' (%d/%d)",
                attchmnt.name,
                count,
                len(attchmnts)
            )
            mime_type, _ = mimetypes.guess_type(attchmnt)
            attchmnts_dict[attchmnt.name] = {
                "mime_type": mime_type,
                "size_bytes": attchmnt.stat().st_size if attchmnt.exists else 0
            }
            app_log.debug("Extracted metadata from attachment")
        return attchmnts_dict


    def _web_search_metadata(
        self,
        qry_wth_urls: list[dict[str, list[str]]] | None,
    ) -> dict[str, list[str]]:
        """
        Add all URL(s) to every query in the list of queries.

        {
            "query_name": [
                "query_url_1",
                "query_url_2",
                "query_url_3"
            ]
        }
        """
        # /////////////////////////////////////////////
        # MIGHT REQUIRE UPDATE FOR METADATA STRUCTURE
        app_log.debug("Updating web search metadata")
        if qry_wth_urls:
            wb_search_dict = {}
            for qry_dict in qry_wth_urls:
                wb_search_dict.update(qry_dict)
            return wb_search_dict
        # /////////////////////////////////////////////
        return {}


    def _tool_calls_metadata(
        self,
        attchmnts: list[Path] | None = None,
        qry_wth_urls: list[dict[str, list[str]]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Return a dictionary of all tool calls.

        {
            "attachments": ,
            "web_search": 
        }
        """
        app_log.debug("Updating metadata for tool calls for the new conversation turn")
        tool_entries = {}

        if attchmnts:
            tool_entries["attachments"] = self._attachments_metadata(attchmnts)

        if qry_wth_urls:
            tool_entries["web_search"] = self._web_search_metadata(qry_wth_urls)

        return tool_entries


    # =============================================================
    # CHAT COMPRESSION
    # =============================================================

    def compress_active_conv(self, prompt: str, contxt: list[dict] | None = None):
        """Call model to summarise all conversations where 'is_compressed' = FALSE in the database."""
        app_log.info("Compressing session '%s' chat logs", self.sess_name)

        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, cmp_convs=self.get_chat_history("not_compressed")
        )

        smry, p_tkns, o_tkns = llm.response_with_new_sys_prompt_and_context(
            model=MODEL, sys_prompt=COMPRESS_PROMPT, contxt=contxt, prompt=cmbind_prompt,
        )
        app_log.debug("Chat compression complete. Updating chat logs metadata")

        # Update 'is_compress' status for previous conversations
        self.cur.execute(
            """
            UPDATE chat_logs
            SET is_compressed = TRUE
            WHERE session_id = %s AND is_compressed = FALSE;
            """,
            (self.get_sess_id(),)
        )
        self.conn.commit()
        app_log.debug(
            "All previous conversation turns for session '%s' has been listed as 'is_copmpressed = TRUE'",
            self.sess_name
        )

        self.add_conv_turn(
            prompt=prompt,
            response=smry,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )
        app_log.info(
            "Chat compression and chat logs metadata has been updated for session '%s'",
            self.sess_name
        )

        self.actv_convs = self.get_actv_convs() # resync messages


    def auto_compresss_active_conv(self):
        """Auto compress session."""
        app_log.info("Chat compression was triggered for session '%s'", self.sess_name)
        prompt = "Summarise all previous conversations."
        self.compress_active_conv(prompt)
        app_log.info("Auto compression complete. Continuing session")


    # =============================================================
    # Exit
    # =============================================================

    def _close_conn(self):
        """Close connection to database."""
        app_log.info("Closing connection to the database")
        self.cur.close()
        self.conn.close()
        app_log.info("Database connection has closed")
