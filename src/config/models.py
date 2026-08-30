import sys
from typing import Any
from pathlib import Path

from src import models_database
from src.config.files_and_directories import CONFIG_FILE, _cfg


def _model_not_found(config_file: Path, model_type: str, model_max_tokens: str):
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


def _normalise_to_integer(value: Any) -> int | str | None:
    """Converts valid values to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return f"Error: Invalid value '{value}'."


def _set_model_tokens(
    _cfg: dict[str, Any],
    config_file: Path,
    field: str,
    field_name: str
) -> tuple[str, Any]:
    """Verify if 'model_name' is on the list or specified in 'config.toml'."""
    model               = _cfg["models"][field]
    max_tokens          = _cfg["models"].get(field_name, None)
    model_max_tokens    = _normalise_to_integer(max_tokens)

    if model not in models_database.MODEL_MAX:
        if model_max_tokens is None:
            _model_not_found(config_file, field, field_name)
        _check_if_value_is_valid(config_file, field_name, model_max_tokens)

    return model, model_max_tokens


MODEL, MODEL_MAX_TOKENS     = _set_model_tokens(_cfg, CONFIG_FILE, "chat", "chat_max_tokens")
EMBED_MODEL, MEM_MAX_TOKENS = _set_model_tokens(_cfg, CONFIG_FILE, "memory", "memory_max_tokens")
PM_MODEL, PM_MAX_TOKENS     = _set_model_tokens(_cfg, CONFIG_FILE, "project_manager", "pm_max_tokens")

FALLBACK_TOKENIZER = _cfg["models"]["fallback_tokenizer"]
