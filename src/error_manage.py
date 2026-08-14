import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

#from src.logger import get_logger


#logger = get_logger(__name__)


# =================================
# Error management
# - Only for 'config.toml' errors.
# =================================

class ErrorManage:
    @staticmethod
    def model_not_found(config_file: Path, model_type: str, model_max_tokens: str):
        """
        Error message for when the model specified in
        'config.toml' does not match any of models on
        MODEL_MAX list in 'src/agent/tokens_handler.py'.
        """
        #logger.critical(
        #    "Unrecognised model: model=%s, missing_parameter=%s",
        #    model_type,
        #    model_max_tokens
        #)
        error_message = (
            f"Error in {config_file}:\n"
            f"User selected '{model_type}' model does not match any in the existing data base, "
            f"please change the model name or add '{model_max_tokens}' value."
        )
        print(error_message)
        sys.exit(1)

    @staticmethod
    def check_if_value_is_valid(config_file: Path, field_name: str, value: Any):
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

    @staticmethod
    def url_check(config_file: Path, field_name: str, url: str):
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

    @staticmethod
    def check_if_system_prompt_exist(config_file: Path, field_name: str, path: Path):
        """Check if the system prompt exist in 'prompts/'."""
        if not path.exists():
            # logger.critical(
            #     "System prompt file does not exists: field=%s, path=%s",
            #     field_name,
            #     path
            # )
            print(f"Error in {config_file}:")
            print(f"'{field_name}' prompt file not found: '{path}'")
            sys.exit(1)


