import datetime
import requests
import re
import socket
import ollama

from src import config
from src.agent.models.llm import LLM
from src.agent.tokens_handler import Tokens
from src.logger import app_logger
from src.tools.web_search.firewall import validate_url, SSRFError
from src.tools.web_search.search_client import SearchClient


app_log = app_logger(f"{__name__}.app")


def is_connected(host="1.1.1.1", port=53, timeout=3):
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


class QueryRouter:
    def __init__(self):
        self.model  = config.MODEL
        self.prompt = config.SEARCH_OR_NOT_PROMPT
        self.tokens = Tokens(model=self.model)


    def search_or_not(self, context: list[dict], prompt: str) -> tuple[bool, int, int]:
        """
        Query model to decide whether a question requires search or not.
        Returns either 'True' or 'False'.
        """
        output, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.prompt,
            context=context,
            prompt=prompt
        )

        if 'true' in output.lower():
            return True, p_tkns, o_tkns
        else:
            return False, p_tkns, o_tkns


class QueryGenerator:
    def __init__(self):
        self.model  = config.MODEL
        self.prompt = config.QUERY_PROMPT
        self.tokens = Tokens(model=self.model)


    def _search_query_check(self, search_query: str) -> str:
        """
        Check format of the search query, remove '"' if it
        exist at the start and end of the query.
        """
        if not search_query:
            return ""

        # Get only the first line of output
        search_query = search_query.strip().split('\n')[0]

        # Remove advanced operators that trigger WAF blocks
        bad_operators = ["inurl:", "site:", "intitle:", "filetype:", "sorted:newest", "sorted:"]
        for operator in bad_operators:
            search_query = search_query.replace(operator, "")

        # Remove quotes, colons and stray punctuation
        search_query = search_query.replace('"', "").replace("'", "").replace(":", "")
        search_query = search_query.strip('`* ')

        # Replace any double space into single space
        search_query = re.sub(r"\s+", " ", search_query)

        return search_query


    def generate_query(self, context: list[dict], prompt: str) -> tuple[str, int, int]:
        """Generate query from user input with dynamic date injection."""
        # Get current date
        current_date = datetime.datetime.now().strftime("%A, %d %B %Y")

        # Update {{current_date}} in 'query_prompt' to actual date
        live_query_prompt = self.prompt.replace("{{current_date}}", current_date)

        query, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=live_query_prompt,
            context=context,
            prompt=prompt
        )

        return self._search_query_check(query), p_tkns, o_tkns


class Search:
    def __init__(self, sess_name: str | None = None):
        self.sess_name  = sess_name
        self.model      = config.MODEL
        self.tokens     = Tokens(model=self.model)
        self.s_client   = SearchClient(sess_name=self.sess_name)
        self.qry_rout   = QueryRouter()
        self.qry_gen    = QueryGenerator()


    def gen_query_and_get_surface_content(
        self,
        context: list[dict],
        prompt: str
    ) -> list[dict] | None:
        """
        Search web, outputing custom max results and
        store results into a temporary file.
        """
        qry, _, _ = self.qry_gen.generate_query(context, prompt)

        surf_cont = self.s_client.get_surface_content(qry=qry)
        if surf_cont:
            self.s_client.add_search_logs(qry=qry, results=surf_cont)
            return surf_cont
        return
