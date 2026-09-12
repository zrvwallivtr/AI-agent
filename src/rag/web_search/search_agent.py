import datetime
import requests
import re
import socket

from src.config import models
from src.config import prompts
from src.config import web_search

from src.agent import LLM, Tknizr
from src.rag.web_search.query_search_engine.query_manager import search_or_not, generate_query
from src.rag.web_search.web_scraping import llm_search_arxiv, clean_website_name
from src.logger import app_logger
from src.rag.web_search.firewall import validate_url, SSRFError
from src.rag.web_search.search_logs import SearchLogs


app_log = app_logger(f"{__name__}.app")


MODEL                   = models.MODEL
SEARCH_OR_NOT_PROMPT    = prompts.SEARCH_OR_NOT_PROMPT
QUERY_PROMPT            = prompts.QUERY_PROMPT
SEARCH_ENG              = web_search.SEARCH_ENG
MAX_RESULTS             = web_search.MAX_RESULTS

AVA_WEBS = """You choose exactly one website name from a fixed list. Follow these rules exactly.

AVAILABLE WEBSITES:
arXiv

RULES:
1. Output ONLY the website name — nothing else. No explanation, no punctuation, no quotes,
   no markdown, no extra words before or after it.
2. The output must be an EXACT match to one name from AVAILABLE WEBSITES above, spelled and
   capitalized exactly as shown (e.g. "arXiv", not "Arxiv" or "ARXIV" or "arxiv.org").
3. Choose the single best fit for the user's query based on the conversation context.
4. If no website in the list is a reasonable fit, output exactly: none

EXAMPLES:

User: find papers about black holes
arXiv

User: what's today's weather
none
"""


def is_connected(host="1.1.1.1", port=53, timeout=3) -> bool:
    """
    Returns True if the system can connect to the host/port,
    otherwise returns false.
    Host (Cloudflare DNS):  1.1.1.1
    Port (DNS traffic):     53
    """
    try:
        # Create socket object with connection timeout
        socket.setdefaulttimeout(timeout)

        # Attempt to connect to the host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True

    except (socket.timeout, OSError):
        return False


class SearchAgent:
    def __init__(
        self,
        conn,
        sess_name: str | None = None
    ):
        self.sess_name  = sess_name
        self.model      = MODEL
        # self.tokens     = Tknizr(model=self.model)
        self.s_client   = SearchLogs(conn=conn, sess_name=self.sess_name)


    # ============================================================
    # REQUIRES SEARCH ENGINE
    # ============================================================

    def _get_surface_content(self, qry: str, max_results: int = MAX_RESULTS) -> list[dict] | None:
        """Get url, title and snippet from query results."""
        params = {"q": qry, "format": "json", "language": "en", "categories": "general"}

        # Generic user-agent to prevent basic anti-bot blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(
                f"{SEARCH_ENG}/search",
                params=params,
                headers=headers,
                timeout=10,
            )
            print(response)
            results = response.json().get("results", [])
            print(results)
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


    def query_surface_content(
        self,
        context: list[dict],
        prompt: str
    ) -> tuple[list[dict], int, int] | None:
        """
        Search web, outputing custom max results and
        store results into a temporary file.
        """
        if not is_connected():
            print("Failed to search online content: Internet not connected")
            return

        qry, p_tkns, o_tkns = generate_query(
            model=self.model, context=context, prompt=prompt
        )

        surf_cont = self._get_surface_content(qry=qry)
        if surf_cont:
            self.s_client.add_search_logs(qry=qry, results=surf_cont)
            return surf_cont, p_tkns, o_tkns
        return


    # ============================================================
    # SCRAPING KNOWN WEBSITES
    # ============================================================

    def model_decide_website_to_scrape(self, msgs: list[dict], prompt: str) -> tuple[str, int, int] | None:
        """Call model to decide which know website to scrape."""
        # Pass to suitable known website
        web_response = LLM.response_with_new_sys_prompt_and_context(
            model=self.model, system_prompt=AVA_WEBS, prompt=prompt, context=msgs
        )
        if not web_response:
            return
        web_name, web_p_tkns, web_o_tkns = web_response

        # Sanitize model's output
        web_name = clean_website_name(web_name)
        if web_name is None:
            return

        # Search / scrape website
        sear_response = llm_search_arxiv(model=self.model, context=msgs, prompt=prompt)
        if not sear_response:
            return
        sear_results, sear_p_tkns, sear_o_tkns = sear_response

        # Model interpret results
        ans, ans_p_tkns, ans_o_tkns = LLM.model_response(model=MODEL, msgs=msgs)
        tol_p_tkns = web_p_tkns + sear_p_tkns + ans_p_tkns
        tol_o_tkns = web_o_tkns + sear_o_tkns + ans_o_tkns

        return ans, tol_p_tkns, tol_o_tkns
