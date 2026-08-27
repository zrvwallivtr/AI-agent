import psycopg2
import requests

from src import config
from src.agent.chat_logs import ChatLogs


class SearchClient:
    def __init__(
        self,
        bs_url: str = config.SEARCH_ENG,
        sess_name: str | None = None
    ):
        self.conn   = psycopg2.connect(
            dbname=config.DBNAME,
            user=config.USER,
            password=config.PASSWORD,
            host=config.HOST,
            port=config.PORT
        )
        self.cur    = self.conn.cursor()

        self.bs_url = bs_url
        self.sess_name = sess_name

        self.chat_logs = ChatLogs(sess_name=self.sess_name)
        self.sess_id = self.chat_logs.sess_id

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

    # ===================================================
    # SEARCH
    # ===================================================

    def get_surface_content(self, qry: str, max_results: int = config.MAX_RESULTS) -> list[dict] | None:
        """Get url, title and snippet from query results."""
        params = {"q": qry, "format": "json", "language": "en", "categories": "general"}

        # Generic user-agent to prevent basic anti-bot blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(
                f"{self.bs_url}/search",
                params=params,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            return [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r.get("content", "")
                } for r in results[:max_results]
            ]

        except Exception as e:
            print(f"Search failed: {e}")
            return

    # ===================================================
    # SEARCH LOGS
    # ===================================================

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
                (self.sess_id, qry, url, title, snippet)
            )
            self.conn.commit()

    def clear_sess_search_logs(self) -> str:
        """Clear all session related search logs."""
        self.cur.execute(
            """
            DELETE FROM search_logs
            WHERE session_id = %s;
            """,
            (self.sess_id,)
        )
        del_count = self.cur.rowcount
        self.conn.commit()

        if del_count == 0:
            return f"Failed to clear search log(s): Search logs or session '{self.sess_name}' does not exists"
        return f"Cleared session '{self.sess_name}' search log(s)"
