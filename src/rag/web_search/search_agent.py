import datetime
import requests
import re
import socket

from src.config import models
from src.config import prompts

from src.agent import LLM, Tknizr
from src.rag.web_search.query_manager import search_or_not, generate_query
from src.logger import app_logger
from src.rag.web_search.firewall import validate_url, SSRFError
from src.rag.web_search.search_client import SearchClient


app_log = app_logger(f"{__name__}.app")


MODEL                   = models.MODEL
SEARCH_OR_NOT_PROMPT    = prompts.SEARCH_OR_NOT_PROMPT
QUERY_PROMPT            = prompts.QUERY_PROMPT


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
    def __init__(self, conn, sess_name: str | None = None):
        self.sess_name  = sess_name
        self.model      = MODEL
        # self.tokens     = Tknizr(model=self.model)
        self.s_client   = SearchClient(conn=conn, sess_name=self.sess_name)


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

        surf_cont = self.s_client.get_surface_content(qry=qry)
        if surf_cont:
            self.s_client.add_search_logs(qry=qry, results=surf_cont)
            return surf_cont, p_tkns, o_tkns
        return
