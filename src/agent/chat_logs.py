import psycopg2
import mimetypes
import json
import uuid
from pathlib import Path
from typing import Literal, Any

from src import config
from src.logger import get_logger


log = get_logger(__name__)


class ChatLogs:
    def __init__(self, sess_name: str | None = None):
        self.conn = psycopg2.connect(
            dbname=config.DBNAME,
            user=config.USER,
            password=config.PASSWORD,
            host=config.HOST,
            port=config.PORT
        )
        self.cur = self.conn.cursor()

        self.sys_prompt     = config.SYS_PROMPT
        self.cmp_prompt   = config.COMPRESS_PROMPT

        self.sess_name      = sess_name.strip() if sess_name else "default_session"

        self._init_chat_logs_db()
        self.sess_id        = self._get_or_create_sess_id()
        self.convs          = self.get_entire_conv()


    # =============================================================
    # INITIALISE CHAT LOGS DATABASE
    # =============================================================

    def _init_chat_logs_db(self):
        """Create session lookup and chat logs table if missing."""
        log.info(
            "Initialising chat_sessions table for session '%s'",
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

        log.info(
            "Initialising chat_logs table for session '%s'",
            self.sess_name
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id          BIGSERIAL PRIMARY KEY,
                session_id  UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                prompt      TEXT NOT NULL,
                response    TEXT NOT NULL,
                state       VARCHAR(20) NOT NULL CHECK (state IN ('external', 'internal')),
                metadata    JSONB DEFAULT '{}'::jsonb
            );
            """
        )

        # Index for session id lookup
        log.info("Initialising index 'idx_chat_logs_session_id' on table 'chat_logs'")
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

        if row:
            sess_id = str(row[0])
            log.info(
                "Retrieved session id '%s' for session '%s'",
                sess_id,
                self.sess_name
            )
            return sess_id
        else:
            log.info(
                "Session id for session '%s' not found: Session does not exists",
                self.sess_name
            )
            return


    def get_all_existing_sess_metadata(self) -> dict:
        """Fetch all session names from database."""
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
            log.info("No existing session found in chat_sessions")
            return {}

        for row in rows:
            sess_dict[str(row[0])] = {
                "session_name": str(row[1]),
                "created_at": str(row[2])
            }
        log.info("%d session(s) found in the chat_sessions", len(rows))
        return sess_dict


    def _create_sess(self) -> str:
        """Create session entry on the chat_sessions table and return its session id."""
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

        log.info(
            "Session '%s' created: Added new session entry to the chat_sessions table", 
            self.sess_name
        )
        return str(row[0])


    def _get_or_create_sess_id(self) -> str:
        """
        Fetch session id if session already exists,
        else create new session entry and return
        new generated session id.
        """
        sess_id = self.get_sess_id()

        if not sess_id:
            log.info("Session '%s' does not exists. Creating new session", self.sess_name)
            sess_id = self._create_sess()
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
        metadata = self._tool_calls_metadata(attchmnts, qry_wth_urls)
        self.cur.execute(
            """
            INSERT INTO chat_logs (session_id, prompt, response, state, metadata)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (self.sess_id, prompt, response, state, json.dumps(metadata or {}))
        )
        self.conn.commit()
        log.info("New conversation turn added to session '%s' chat logs", self.sess_name)

        # Resync messages
        self.convs = self.get_entire_conv()
        log.debug("Resynced session '%s' conversations", self.sess_name)


    def clear_sess_chat_logs(self) -> tuple[str, bool]:
        """Clear all session related chat logs."""
        self.cur.execute(
            """
            DELETE FROM chat_logs
            WHERE session_id = %s;
            """,
            (self.sess_id,)
        )
        del_count = self.cur.rowcount
        self.conn.commit()

        if del_count == 0:
            log.warning(
                "Failed to clear chat logs: Session '%s' does not exists or has no chat logs",
                self.sess_name
            )
            return f"Failed to clear chat logs: Session '{self.sess_name}' does not exists or has no chat logs", False
        log.info("Cleared session '%s' chat logs", self.sess_name)

        self.convs = self.get_entire_conv() # resync messages
        log.debug("Resynced session '%s' conversations", self.sess_name)
        return f"Cleared session '{self.sess_name}' chat logs", True


    # =============================================================
    # FROM CHAT LOGS
    # =============================================================

    def get_entire_conv(self) -> list[dict]:
        """
        Get all messages in a session. 
        If session does not exists, return system prompt.
        """
        sys_prompt = [{"role": "system", "content": self.sys_prompt}]

        self.cur.execute(
            """
            SELECT prompt, response
            FROM chat_logs
            WHERE session_id = %s
            ORDER BY created_at ASC;
            """,
            (self.sess_id,)
        )
        self.conn.commit()
        rows = self.cur.fetchall()

        if rows:
            convs = []
            for row in rows:
                convs.append({"role": "user", "content": row[0]})
                convs.append({"role": "assistant", "content": row[1]})
            log.debug(
                "%d conversation turns retrieved from session '%s'",
                len(rows),
                self.sess_name
            )
            return sys_prompt + convs
        log.debug(
            "Session '%s' conversation does not exists. Returning system prompt only",
            self.sess_name
        )
        return sys_prompt


    def get_latest_conv_turn(self) -> list[dict] | None:
        """Get the latest external user/assistant conversation turn from the chat log."""
        self.cur.execute(
            """
            SELECT prompt, response
            FROM chat_logs
            WHERE session_id = %s AND state = 'external'
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (self.sess_id,)
        )
        self.conn.commit()
        row = self.cur.fetchone()

        if row:
            log.debug(
                "Retrieved latest conversation turn from session '%s' chat logs",
                self.sess_name
            )
            return [
                {"role": "user", "content": row[0]},
                {"role": "assistant", "content": row[1]}
            ]
        else:
            log.debug(
                "No conversation found in session '%s' chat logs: New session or session does not exists",
                self.sess_name
            )
            return


    def get_old_convs(self) -> list[dict] | None:
        """
        Get all previous user/assistant conversation turns
        right before the latest external conversation from
        the chat log.
        """
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
            (self.sess_id, self.sess_id)
        )
        self.conn.commit()
        rows = self.cur.fetchall()

        if rows:
            convs = []
            for row in rows:
                convs.append({"role": "user", "content": row[0]})
                convs.append({"role": "assistant", "content": row[1]})
            log.debug(
                "%d conversation turn(s) retrived form session '%s' chat logs",
                len(rows),
                self.sess_name
            )
            return convs
        else:
            log.debug(
                "No conversation found in session '%s' chat logs: New session or session does not exists",
                self.sess_name
            )
            return


    # =============================================================
    # METADATA
    # =============================================================

    def _attachments_metadata(
        self,
        attchmnts: list[Path] | None
    ) -> dict[str, dict[str, Any]]:
        """
        Add metadata to every filename in the list of filenames.

        {
            "filename": {
                "mime_type": "type",
                "size_bytes": size,
            }
        }
        """
        if attchmnts:
            attchmnts_dict = {}
            for attchmnt in attchmnts:
                mime_type, _ = mimetypes.guess_type(attchmnt)
                attchmnts_dict[attchmnt.name] = {
                    "mime_type": mime_type,
                    "size_bytes": attchmnt.stat().st_size if attchmnt.exists else 0
                }
                log.debug("Added metadata to attachment '%s'", attchmnt)
            return attchmnts_dict
        return {}


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
        tool_entries = {}
        if attchmnts:
            tool_entries["attachments"] = self._attachments_metadata(attchmnts)
        if qry_wth_urls:
            tool_entries["web_search"] = self._web_search_metadata(qry_wth_urls)
        return tool_entries


    # =============================================================
    # Exit
    # =============================================================

    def _close_conn(self):
        """Close connection to database."""
        self.cur.close()
        self.conn.close()
