import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from src.config.files_and_directories import _cfg, CONFIG_FILE


def _check_if_value_is_valid(config_file: Path, field_name: str, value: Any):
    """Check if the value is a number in the field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # logger.critical(
        #     "Invalid setting: field=%s, value=%s",
        #     field_name,
        #     value
        # )
        print(f"Error in {config_file}:")
        print(f"'{field_name}' must be a number, got '{value}' instead.")
        sys.exit(1)


def _url_check(config_file: Path, field_name: str, url: str):
    """Check if url is valid."""
    parsed = urlparse(url)
    if not parsed.scheme in ("http", "https") or not parsed.netloc:
        # logger.critical(
        #     "Invalid URL: field=%s, url=%s",
        #     field_name,
        #     url
        # )
        print(f"Error in {config_file}:")
        print(f"'{field_name}' must be a valid URL, got '{url}' instead.")
        print(f"Example: 'http://localhost:8080/search'")
        sys.exit(1)


SEARCH_ENG          = _cfg["search"]["engine"]
MAX_RESULTS         = _cfg["search"]["max_results"]
MAX_CHAR_PER_PAGE   = _cfg["search"]["max_char_per_page"]
CRAWL4AI_URL        = _cfg["search"]["crawl4ai_url"]

# Safeguard before enabling auto websearch
AUTO_WEB_SEARCH_TOKENS = _cfg["search"]["auto_web_search_enable_at_model_tokens"]
ENABLE_AUTO_WEB_SEARCH = _cfg["search"]["enable_auto_web_search"]


_url_check(CONFIG_FILE, "engine", SEARCH_ENG)
_check_if_value_is_valid(CONFIG_FILE, "max_results", MAX_RESULTS)
