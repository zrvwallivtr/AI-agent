import shutil
import tomllib
import sys
from pathlib import Path
from urllib.parse import urlparse

from src.agent.tokens_handler import MODEL_MAX, Tokens


# ========================================================
# Read-only static assets
# ========================================================

def get_bundle_root() -> Path:
    """
    Returns base path for static bundled assets.
    Handles standard execution and PyInstaller bundles.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent

BUNDLE_ROOT     = get_bundle_root()
PROMPT_DIR      = BUNDLE_ROOT / "src" / "prompts"
DEFAULT_CONFIG  = BUNDLE_ROOT / "config.toml"


# ========================================================
# Writable persistent data (stored in ~/.agent_app)
# ========================================================

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

DATA_DIR        = get_user_app_dir() # ~/.agent_app
APP_DATA_DIR    = get_user_app_data_dir() # ~/.agent_app/data
CUS_PROMPT_DIR  = APP_DATA_DIR / "prompts"
CHATS_DIR       = APP_DATA_DIR / "chats"
CHROMADB_DIR    = APP_DATA_DIR / "chroma"
PROJECTS_DIR    = APP_DATA_DIR / "projects"
DROPBOX_DIR     = APP_DATA_DIR / "dropbox"

# Defaults chat session paths
DEFAULT_PATH                = CHATS_DIR / "default_session" / "chat.json"
DEFAULT_CHAT_HISTORY_PATH   = CHATS_DIR / "default_session" / "chat_history.json"

# Auto-create user data directories on startup
for folder in [DATA_DIR, APP_DATA_DIR, CUS_PROMPT_DIR, CHATS_DIR, CHROMADB_DIR, PROJECTS_DIR, DROPBOX_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ========================================================
# Load user config file (config.toml)
# ========================================================

config_file = DATA_DIR / "config.toml"

def config_toml_handler(config_file: Path):
    """
    Check if 'config.toml' exists in '~/.agent_app',
    if not copy the default config.toml to user directory.
    """
    if not config_file.exists() and DEFAULT_CONFIG.exists():
        shutil.copy(DEFAULT_CONFIG, config_file) # Copy default 'config.toml' to '~/.agent_app/'

    elif not config_file.exists():
        config_file = DEFAULT_CONFIG # Fallback to default

config_toml_handler(config_file)

_cfg = tomllib.loads(config_file.read_text(encoding="utf-8"))


# ========================================================
# Error check class:
#
# - Only accounts for input errors in `config.toml`.
# ========================================================

class ErrorManage:
    @staticmethod
    def _model_not_found(model_type: str, model_max_tokens: str):
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
    def _check_if_value_is_valid(field_name: str, value: int | float | None):
        """Check if the value is a number in the field."""
        if not isinstance(value, (int, float)):
            print(f"Error in {config_file}:")
            print(f"'{field_name}' must be a number, got '{value}' instead.")
            sys.exit(1)

    @staticmethod
    def _url_check(field_name: str, url: str):
        """Check if url is valid."""
        parsed = urlparse(url)
        if not parsed.scheme in ("http", "https") or not parsed.netloc:
            print(f"Error in {config_file}:")
            print(f"'{field_name}' must be a valid URL, got '{url}' instead.")
            print(f"Example: 'http://localhost:8080/search'")
            sys.exit(1)

    @staticmethod
    def _check_if_system_prompt_exist(field_name: str, path: Path):
        """Check if the system prompt exist in 'prompts/'."""
        if not path.exists():
            print(f"Error in {config_file}:")
            print(f"'{field_name}' prompt file not found: '{path}'")
            sys.exit(1)


# ========================================================
# Model selection
# ========================================================

def _normalise_to_integer(value) -> int | None:
    """Returns .0 to 'int'."""
    if value is not None:
        return int(value)
    else:
        return None

def _set_model_tokens(field: str, field_name: str) -> tuple[str, int | None]:
    """Verify if 'model_name' is on the list or specified in 'config.toml'."""
    model               = _cfg["models"][field]
    max_tokens          = _cfg["models"].get(field_name, None)
    model_max_tokens    = _normalise_to_integer(max_tokens)

    if model not in MODEL_MAX:
        if model_max_tokens is None:
            ErrorManage._model_not_found(field, field_name)
        ErrorManage._check_if_value_is_valid(field_name, model_max_tokens)

    return model, model_max_tokens

MODEL, MODEL_MAX_TOKENS     = _set_model_tokens("chat", "chat_max_tokens")
EMBED_MODEL, MEM_MAX_TOKENS = _set_model_tokens("memory", "memory_max_tokens")
PM_MODEL, PM_MAX_TOKENS     = _set_model_tokens("project_manager", "pm_max_tokens")


# ========================================================
# Prompts
# ========================================================

def _prompt_path_handler(filename: str, dir: Path) -> tuple[Path, str]:
    # Check if file exist
    prompt_path = dir / filename
    ErrorManage._check_if_system_prompt_exist(filename, prompt_path)

    prompt = (prompt_path).read_text(encoding="utf-8").strip()
    return prompt_path, prompt

def _prioritise_custom_prompt(filename: str, custom_prompt_dir: Path, default_prompt_dir: Path) -> tuple[Path, str]:
    """Returns active directory and prompt if exists, else, return default directory and prompt."""
    custom_prompt_path = custom_prompt_dir / filename
    path = Path(custom_prompt_path)

    if path.exists():
        return _prompt_path_handler(filename, custom_prompt_dir)

    return _prompt_path_handler(filename, default_prompt_dir)

# System
SYS_PROMPT_DIR              = PROMPT_DIR / "system"
CUS_SYS_PROMPT_DIR          = CUS_PROMPT_DIR / "system"
SYS_PROMPT_PATH, SYS_PROMPT = _prioritise_custom_prompt("standard", CUS_SYS_PROMPT_DIR, SYS_PROMPT_DIR)

# Memory
MEM_PROMPT_DIR                              = PROMPT_DIR / "memory"
CUS_MEM_PROMPT_DIR                          = CUS_PROMPT_DIR / "memory"
MEM_PROMPT_PATH, MEM_PROMPT                 = _prioritise_custom_prompt("memory_agent", CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR)
MEM_MANUAL_PROMPT_PATH, MEM_MANUAL_PROMPT   = _prioritise_custom_prompt("manual_memory_extraction", CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR)

# Project manager
PM_PROMPT_DIR               = PROMPT_DIR / "project_manager"
CUS_PM_PROMPT_DIR           = CUS_PROMPT_DIR / "project_manager"
PM_PROMPT_PATH, PM_PROMPT   = _prioritise_custom_prompt("manager", CUS_PM_PROMPT_DIR, PM_PROMPT_DIR)

# Search agent
SEARCH_AGENT_PROMPT_DIR                         = PROMPT_DIR / "search_agent"
CUS_SEARCH_AGENT_PROMPT_DIR                     = CUS_PROMPT_DIR / "search_agent"
SEARCH_OR_NOT_PROMPT_PATH, SEARCH_OR_NOT_PROMPT = _prioritise_custom_prompt("search_or_not", CUS_SEARCH_AGENT_PROMPT_DIR, SEARCH_AGENT_PROMPT_DIR)
QUERY_PROMPT_PATH, QUERY_PROMPT                 = _prioritise_custom_prompt("query_generator", CUS_SEARCH_AGENT_PROMPT_DIR, SEARCH_AGENT_PROMPT_DIR)

# File reader
FILE_READER_PROMPT_DIR                          = PROMPT_DIR / "file_reader"
CUS_FILE_READER_PROMPT_DIR                      = CUS_PROMPT_DIR / "file_reader"
FILE_OR_NOT_PROMPT_PATH, FILE_OR_NOT_PROMPT     = _prioritise_custom_prompt("file_or_not", CUS_FILE_READER_PROMPT_DIR, FILE_READER_PROMPT_DIR)
GET_FILE_LIST_PROMPT_PATH, GET_FILE_LIST_PROMPT = _prioritise_custom_prompt("get_file_list", CUS_FILE_READER_PROMPT_DIR, FILE_READER_PROMPT_DIR)


# ========================================================
# Memory
# ========================================================

RETRIEVE_MEM_ENTRY_LIMIT = _cfg["memory"]["retrieve_entry_limit"]


# ========================================================
# File reader
# ========================================================

# Safeguard before enabling auto read files from dropbox
AUTO_READ_DROPBOX_TOKENS = _cfg["file_reader"]["model_max_tokens"]


# ========================================================
# Web search
# ========================================================

SEARCH_ENG          = _cfg["search"]["engine"]
MAX_RESULTS         = _cfg["search"]["max_results"]
MAX_CHAR_PER_PAGE   = _cfg["search"]["max_char_per_page"]

# Safeguard before enabling auto websearch
AUTO_WEBSEARCH_TOKENS = _cfg["search"]["model_max_tokens"]


ErrorManage._url_check("engine", SEARCH_ENG)
ErrorManage._check_if_value_is_valid("max_results", MAX_RESULTS)
