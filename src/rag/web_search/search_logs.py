import psycopg2
import requests

from src.config import web_search
from src.config import postgres

from src.agent.chat_logs import ChatLogs


DBNAME      = postgres.DBNAME
USER        = postgres.USER
PASSWORD    = postgres.PASSWORD
HOST        = postgres.HOST
PORT        = postgres.PORT


class SearchLogs:
    def __init__(
        self,
        conn,
        sess_name: str | None = None
    ):
        self.conn   = psycopg2.connect(
            dbname=DBNAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        self.cur    = self.conn.cursor()

        self.sess_name = sess_name

        self.chat_logs = ChatLogs(conn = conn, sess_name=self.sess_name)

        self._init_search_logs_db()


    def _init_search_logs_db(self):
        """Create search logs table if missing."""
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS search_logs (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                snippet TEXT NOT NULL
            );
            """
        )
        self.conn.commit()


    def add_search_logs(self, qry: str, results: list[dict]):
        """Add query to search logs."""
        for r in results:
            url = r["url"]
            title = r["title"]
            snippet = r["snippet"]

            self.cur.execute(
                """
                INSERT INTO search_logs (session_id, query, url, title, snippet)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (self.chat_logs.get_sess_id(), qry, url, title, snippet)
            )
            self.conn.commit()


    def clear_sess_search_logs(self) -> str:
        """Clear all session related search logs."""
        self.cur.execute(
            """
            DELETE FROM search_logs
            WHERE session_id = %s;
            """,
            (self.chat_logs.get_sess_id,)
        )
        del_count = self.cur.rowcount
        self.conn.commit()

        if del_count == 0:
            return f"Failed to clear search log(s): Search logs or session '{self.sess_name}' does not exists"
        return f"Cleared session '{self.sess_name}' search log(s)"
