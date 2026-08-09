import shutil
import sys
from pathlib import Path
from typing import Any

from src import config
from src.error_manage import ErrorManage


# =================================
# Read only static asset
# =================================

def get_bundle_root() -> Path:
    """
    Returns base path for static bundled assets.
    Handles standard execution and PyInstaller bundles.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


# =================================
# User directory (~/.agent_app)
# =================================

def get_user_app_dir(app_name: str = "agent_app") -> Path:
    """
    Returns a writable data directory in the user's home directory.
    Ensures database files and chat logs persist between application runs.
    """
    app_dir = Path.home() / f".{app_name}"
    return app_dir

def get_user_app_data_dir(app_name: str = "agent_app") -> Path:
    """
    Returns a writable data directory in the user's home directory.
    Ensures database files and chat logs persist between application runs.
    """
    app_data_dir = Path.home() / f".{app_name}" / "data"
    return app_data_dir

def create_user_dir(dir_list: list[Path]):
    """Create specified user data directories."""
    for folder in dir_list:
        folder.mkdir(parents=True, exist_ok=True)


# =================================
# User config file
# =================================

def config_toml_handler(config_file: Path, default_config: Path):
    """
    Check if 'config.toml' exists in '~/.agent_app',
    if not copy the default config.toml to user directory.
    """
    if not config_file.exists() and default_config.exists():
        shutil.copy(default_config, config_file) # Copy default 'config.toml' to '~/.agent_app/'

    elif not config_file.exists():
        config_file = default_config # Fallback to default


# =================================
# Model selection
# =================================

def normalise_to_integer(value: Any) -> int | None:
    """Returns .0 to 'int'."""
    if value is not None:
        return int(value)
    else:
        return None

def set_model_tokens(
    _cfg: dict[str, Any],
    config_file: Path,
    field: str,
    field_name: str
) -> tuple[str, int | None]:
    """Verify if 'model_name' is on the list or specified in 'config.toml'."""
    model               = _cfg["models"][field]
    max_tokens          = _cfg["models"].get(field_name, None)
    model_max_tokens    = normalise_to_integer(max_tokens)

    if model not in config.MODEL_MAX:
        if model_max_tokens is None:
            ErrorManage.model_not_found(config_file, field, field_name)
        ErrorManage.check_if_value_is_valid(config_file, field_name, model_max_tokens)

    return model, model_max_tokens


# =================================
# Prompts
# =================================

def prompt_path_handler(
    filename: str,
    config_file: Path,
    dir: Path
) -> tuple[Path, str]:
    """
    Check if file exist, then read file contents
    and return it's path and contents as string.
    """
    prompt_path = dir / filename
    ErrorManage.check_if_system_prompt_exist(config_file, filename, prompt_path)

    prompt = (prompt_path).read_text(encoding="utf-8").strip()
    return prompt_path, prompt

def prioritise_custom_prompt(
    filename: str,
    config_file: Path,
    custom_prompt_dir: Path,
    default_prompt_dir: Path
) -> tuple[Path, str]:
    """Returns active directory and prompt if exists, else, return default directory and prompt."""
    custom_prompt_path = custom_prompt_dir / filename
    path = Path(custom_prompt_path)

    if path.exists():
        return prompt_path_handler(filename, config_file, custom_prompt_dir)

    return prompt_path_handler(filename, config_file, default_prompt_dir)

