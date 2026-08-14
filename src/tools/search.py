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

    def search_or_not(self, context: list[dict], prompt: str) -> bool:
        """
        Query model to decide whether a question requires search or not.
        Returns either 'True' or 'False'.
        """
        output, prompt_tokens, output_tokens = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.search_or_not_prompt,
            context=context,
            prompt=prompt
        )

        if 'true' in output.lower():
            return True
        else:
            return False

    def search_query_check(self, search_query: str) -> str:
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

    def generates_query(self, context: list[dict], prompt: str) -> str:
        """Generate query from user input with dynamic date injection."""
        # Get current date
        current_date = datetime.datetime.now().strftime("%A, %d %B %Y")

        # Update {{current_date}} in 'query_prompt' to actual date
        live_query_prompt = self.query_prompt.replace("{{current_date}}", current_date)

        query, prompt_tokens, output_tokens = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=live_query_prompt,
            context=context,
            prompt=prompt
        )

        return self.search_query_check(query)

    def web(
        self,
        query: str,
        context: list[dict],
        prompt: str,
        max_results=config.MAX_RESULTS
    ) -> tuple[str, list[dict[str, list[str]]], int, int, bool]:
        """
        Answer question from search results. Always returns
        a 4-element tuple to maintain unpack safety.

        Steps:
        1. Search web, outputing custom max results.
        2. Store results into a temporary file.
        3. Model reads file and response.
        
        Note: Only the initial user's question and model
              generated response will be logged.
        """
        print(f"Searching {query} on {config.SEARCH_ENG}...")

        results = _get_results_from_query(query, config.SEARCH_ENG, config.MAX_RESULTS)
        if not results:
            return "Error: No results found.", [{"": []}], 0, 0, False

        # Store in temp file
        tmp     = _store_results_in_tmp_file(results)
        urls    = _urls_from_results(results)

        query_with_urls = [{f"{query}": urls}]

        # Model reads search results
        try:
            print("Reading search results...")
            search_results = _read_tmp_file(tmp)

            # print(search_results)

            query_message = LLM.user(f"# Context from web search\n\n{search_results}\n\n---\n\n# User prompt\n\n{prompt}") # This won't be saved in chat

            messages = context + [query_message]

            response, prompt_tokens, output_tokens = LLM.model_response(messages, self.model)
            return response, query_with_urls, prompt_tokens, output_tokens, True

        finally:
            # Ensure cleanup even if LLM fails mid-execution
            if os.path.exists(tmp):
                os.unlink(tmp)

    def auto_web_search(
        self,
        context: list[dict],
        prompt: str
    ) -> tuple[str, list[dict[str, list[str]]], bool]:
        """Generates, search and answer query based on user prompt."""
        #print("Generating query...")
        query = self.generates_query(
            context=context,
            prompt=prompt
        )

        response, query_with_urls, prompt_tokens, output_tokens, search = self.web(
            query=query,
            context=context,
            prompt=prompt,
            max_results=config.MAX_RESULTS
        )

        if search == False:
            return "Skip.", [{"": []}], False

        response = f"{response}\n{'=' * 40}"

        return response, query_with_urls, True

    def toggle_auto_web_search(
        self,
        messages: list[dict],
        prompt: str,
        memory_entries: str,
        file_contents: str,
        enable_attachments: bool,
        added_file_contents: str,
        enable_auto_web_search: bool,
    ) -> tuple[str, list[dict[str, list[str]]], bool]:
        """
        Auto web search ability, returns search results if its toggled on.

        Model decides from {memory_entries} + {file_content} + {prompt} --> {search_results}
        """
        # Check if internet is available
        internet = is_connected()

        if internet == True and enable_auto_web_search == True and self.tokens.model_max_tokens > config.AUTO_WEB_SEARCH_TOKENS:

            # Enable auto web search
            search_context = messages

            if enable_attachments == True:
                search_context.append({
                    "role": "user",
                    "content": f"# Retrieved memory entries\n\n{memory_entries}\n\n--\n\n# Previously uploaded files\n\n{file_contents}\n\n---\n\n# User input\n\n{prompt}\n\n## Uploaded files\n\n{added_file_contents}"
                })
            else:
                search_context.append({
                    "role": "user",
                    "content": f"# Retrieved memory entries\n\n{memory_entries}\n\n--\n\n# Previously uploaded files\n\n{file_contents}\n\n---\n\n# User input\n\n{prompt}"
                })

            search = self.search_or_not(search_context, prompt)

            if search == True:
                return self.auto_web_search(search_context, prompt)

        return "Skip.", [{"": []}], False

