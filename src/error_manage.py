from pathlib import Path
from urllib.parse import urlparse
import sys


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
        print(f"Error in {config_file}:")
        error_message = (
            f"User selected '{model_type}' model does not match any in the existing data base, "
            f"please change the model name or add '{model_max_tokens}' value."
        )
        print(error_message)
        sys.exit(1)

    @staticmethod
    def check_if_value_is_valid(config_file: Path, field_name: str, value: int | float | None):
        """Check if the value is a number in the field."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            print(f"Error in {config_file}:")
            print(f"'{field_name}' must be a number, got '{value}' instead.")
            sys.exit(1)

    @staticmethod
    def url_check(config_file: Path, field_name: str, url: str):
        """Check if url is valid."""
        parsed = urlparse(url)
        if not parsed.scheme in ("http", "https") or not parsed.netloc:
            print(f"Error in {config_file}:")
            print(f"'{field_name}' must be a valid URL, got '{url}' instead.")
            print(f"Example: 'http://localhost:8080/search'")
            sys.exit(1)

    @staticmethod
    def check_if_system_prompt_exist(config_file: Path, field_name: str, path: Path):
        """Check if the system prompt exist in 'prompts/'."""
        if not path.exists():
            print(f"Error in {config_file}:")
            print(f"'{field_name}' prompt file not found: '{path}'")
            sys.exit(1)


