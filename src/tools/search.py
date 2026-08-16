import datetime
from operator import mod
import requests
import tempfile
import os
import re
import socket
import ollama
from bs4 import BeautifulSoup

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


def _get_results_from_query(
    query: str,
    search_eng: str = config.SEARCH_ENG,
    max_results: int = config.MAX_RESULTS,
) -> list[dict]:
    """Get results from given query, search engine and set max results."""
    params = {"q": query, "format": "json", "language": "en", "categories": "general"}

    # Generic user-agent to prevent basic anti-bot blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(search_eng, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])[:max_results]

    except Exception as e:
        print(f"Search failed: {e}")
        return []

def _scrape_url_content(url: str) -> str:
    """
    Downloads entire webpage, removes non-content HTML elements
    and returns clean, readable text.
    """
    # Generic user-agent to prevent basic anti-bot blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Ubuntu) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code != 200:
            return ""

        # Parse the page using the fast lxml parser
        soup = BeautifulSoup(response.text, "lxml")

        # Remove noises (Ads, scripts, stylesheets, sidebars)
        for junk in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            junk.decompose()

        # Get plain text and normalize spacing
        raw_text    = soup.get_text(separator="\n")

        clean_lines = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                # Collapse any internal multiple spaces/tabs into a single space
                line = re.sub(r"\s+", " ", line)
                clean_lines.append(line)

        # Join paragraphs back together with single newlines
        clean_text  = "\n".join(clean_lines)

        # Context safety guardrail: Set max characters per webpage
        return clean_text[:config.MAX_CHAR_PER_PAGE]

    except Exception as e:
        print(f"Scraping failed for {url}: {e}")
        return ""

def _urls_from_results(results: list[dict]) -> list[str]:
    """Get all URLs from the visited website."""
    urls = []

    for i, result in enumerate(results[:config.MAX_RESULTS], start=1):
        url = result.get("url", "")
        urls.append(url)

    return urls

def _store_results_in_tmp_file(results: list[dict]) -> str:
    """Store all results in a specific format temporary file."""
    all_results = ""

    for i, result in enumerate(results[:config.MAX_RESULTS], start=1):
        url     = result.get("url", "")
        title   = result.get("title", "")
        snippet = result.get("content", "")

        print(f"[{i}/{config.MAX_RESULTS}] Scraping: {url}...")

        # Fall back to the search snippet if web scraper gets blocked
        full_page_body  = _scrape_url_content(url)
        final_content   = full_page_body if full_page_body else snippet

        all_results += f"### Source {i}: {title}\n"
        all_results += f"URL: {url}\n"
        all_results += f"Full Page Context:\n{final_content}\n"
        all_results += f"---" + "\n\n"

    # Create and store results
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(all_results)
    tmp.close()
    return tmp.name

def _read_tmp_file(filename: str) -> str:
    """Read all contents in temporary file."""
    with open(filename, "r") as f:
        content = f.read()
    return content

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


class SearchAgent:
    def __init__(self):
        self.model                  = config.MODEL
        self.search_or_not_prompt   = config.SEARCH_OR_NOT_PROMPT
        self.query_prompt           = config.QUERY_PROMPT
        self.tokens                 = Tokens(model=self.model)

    def search_or_not(self, context: list[dict], prompt: str) -> tuple[bool, int, int]:
        """
        Query model to decide whether a question requires search or not.
        Returns either 'True' or 'False'.
        """
        output, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.search_or_not_prompt,
            context=context,
            prompt=prompt
        )

        if 'true' in output.lower():
            return True, p_tkns, o_tkns
        else:
            return False, p_tkns, o_tkns

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

    def generates_query(self, context: list[dict], prompt: str) -> tuple[str, int, int]:
        """Generate query from user input with dynamic date injection."""
        # Get current date
        current_date = datetime.datetime.now().strftime("%A, %d %B %Y")

        # Update {{current_date}} in 'query_prompt' to actual date
        live_query_prompt = self.query_prompt.replace("{{current_date}}", current_date)

        query, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=live_query_prompt,
            context=context,
            prompt=prompt
        )

        return self._search_query_check(query), p_tkns, o_tkns

    def web_search_results(
        self,
        query: str,
    ) -> tuple[str, list[dict[str, list[str]]], str]:
        """
        Search web, outputing custom max results and
        store results into a temporary file.
        """
        print(f"Searching {query} on {config.SEARCH_ENG}...")

        results = _get_results_from_query(query, config.SEARCH_ENG, config.MAX_RESULTS)
        if not results:
            return "No results found.", [], ""

        # Store in temp file
        tmp     = _store_results_in_tmp_file(results)
        urls    = _urls_from_results(results)
        content = _read_tmp_file(tmp)
        query_with_urls = [{f"{query}": urls}]

        # Ensure cleanup even if LLM fails mid-execution
        if os.path.exists(tmp):
            os.unlink(tmp)

        return content, query_with_urls, tmp

    def web_search_and_response(
        self,
        query: str,
        context: list[dict],
        prompt: str,
        max_results=config.MAX_RESULTS
    ) -> tuple[str, list[dict[str, list[str]]], int, int]:
        """
        Answer question from search results.

        Note: Only the initial user's question and model
              generated response will be logged.
        """
        search_results, query_with_urls, tmp = self.web_search_results(query)

        query_message = LLM.user(
            f"# Context from web search\n\n"
            f"{search_results}\n\n"
            f"---\n\n"
            f"# User prompt\n\n"
            f"{prompt}"
        )
        messages = context + [query_message]
        response, p_tkns, o_tkns = LLM.model_response(messages, self.model)
        return response, query_with_urls, p_tkns, o_tkns

    def toggle_auto_web_search(
        self,
        messages: list[dict],
        prompt: str,
        enable_attachments: bool,
        enable_auto_web_search: bool,
        memory_entries: str | None = None,
        file_contents: dict[str, str] | None = None,
        attach_file_data: dict[str, str] | None = None,
    ) -> tuple[str, list[dict[str, list[str]]], str]:
        """
        Auto web search ability, returns search results if its toggled on.

        Model decides from {memory_entries} + {file_content} + {prompt} --> {search_results}
        """
        internet = is_connected() # Check if internet is available

        if internet == False:
            logger.warning("Failed to enable model search: No internet connection")
            return "No results found.", [], ""

        if (
            enable_auto_web_search == False
            or self.tokens.model_max_tokens <= config.AUTO_WEB_SEARCH_TOKENS
        ):
            return "No results found.", [], ""

        # Prepare context
        memory_sect = (
            f"# Retrieved memory entry(s)\n\n"
            f"{memory_entries}\n\n"
            f"---\n\n"
        ) if memory_entries else ""
        dropbox_sect = (
            f"# Previously uploaded files\n\n"
            f"{file_contents}"
            f"---\n\n"
        ) if file_contents else ""
        prompt_sect = (
            f"# User prompt\n\n"
            f"{prompt}\n\n"
            f"---\n\n"
        )
        attach_sect = (
            f"# Attachment(s)\n\n"
            f"{attach_file_data}"
        ) if attach_file_data else ""
        cmbind_prompt = memory_sect + dropbox_sect + prompt_sect + attach_sect

        # Auto web search
        search = self.search_or_not(messages, cmbind_prompt)
        if search == True:
            query, _, _ = self.generates_query(messages, prompt)
            return self.web_search_results(query)
        return "No results found.", [], ""
