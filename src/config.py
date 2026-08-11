import tomllib

from src.agent.tokens_handler import MODEL_MAX, Tokens
from src import setup
from src.error_manage import ErrorManage


# ========================================================
# Read-only static assets
# ========================================================

BUNDLE_ROOT     = setup.get_bundle_root()
PROMPT_DIR      = BUNDLE_ROOT / "src" / "prompts"
DEFAULT_CONFIG  = BUNDLE_ROOT / "config.toml"


# ========================================================
# Writable persistent data (stored in ~/.agent_app)
# ========================================================

DATA_DIR        = setup.get_user_app_dir() # ~/.agent_app
APP_DATA_DIR    = setup.get_user_app_data_dir() # ~/.agent_app/data
CUS_PROMPT_DIR  = APP_DATA_DIR / "prompts"
CHATS_DIR       = APP_DATA_DIR / "chats"
CHROMADB_DIR    = APP_DATA_DIR / "chroma"
PROJECTS_DIR    = APP_DATA_DIR / "projects"
DROPBOX_DIR     = APP_DATA_DIR / "dropbox"

# Defaults chat session paths
DEFAULT_PATH                = CHATS_DIR / "default_session" / "chat.json"
DEFAULT_CHAT_HISTORY_PATH   = CHATS_DIR / "default_session" / "chat_history.json"

setup.create_user_dir(
    [
        DATA_DIR,
        APP_DATA_DIR,
        CUS_PROMPT_DIR,
        CHATS_DIR,
        CHROMADB_DIR,
        PROJECTS_DIR,
        DROPBOX_DIR
    ]
)


# ========================================================
# Load user config file (config.toml)
# ========================================================

config_file = DATA_DIR / "config.toml"

setup.config_toml_handler(config_file, DEFAULT_CONFIG)

_cfg = tomllib.loads(config_file.read_text(encoding="utf-8"))


# ========================================================
# Model selection
# ========================================================

MODEL, MODEL_MAX_TOKENS     = setup.set_model_tokens(_cfg, config_file, "chat", "chat_max_tokens")
EMBED_MODEL, MEM_MAX_TOKENS = setup.set_model_tokens(_cfg, config_file, "memory", "memory_max_tokens")
PM_MODEL, PM_MAX_TOKENS     = setup.set_model_tokens(_cfg, config_file, "project_manager", "pm_max_tokens")


# ========================================================
# Prompts
# ========================================================

# System
SYS_PROMPT_DIR              = PROMPT_DIR / "system"
CUS_SYS_PROMPT_DIR          = CUS_PROMPT_DIR / "system"
SYS_PROMPT_PATH, SYS_PROMPT = setup.prioritise_custom_prompt("standard", config_file, CUS_SYS_PROMPT_DIR, SYS_PROMPT_DIR)

# Compression
COMPRESS_PROMPT_DIR                     = PROMPT_DIR / "chat_compression"
CUS_COMPRESS_PROMPT_DIR                 = CUS_PROMPT_DIR / "chat_compression"
COMPRESS_PROMPT_PATH, COMPRESS_PROMPT   = setup.prioritise_custom_prompt("instructions", config_file, CUS_COMPRESS_PROMPT_DIR, COMPRESS_PROMPT_DIR)

# Memory
MEM_PROMPT_DIR                              = PROMPT_DIR / "memory"
CUS_MEM_PROMPT_DIR                          = CUS_PROMPT_DIR / "memory"
MEM_PROMPT_PATH, MEM_PROMPT                 = setup.prioritise_custom_prompt("memory_agent", config_file, CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR)
MEM_MANUAL_PROMPT_PATH, MEM_MANUAL_PROMPT   = setup.prioritise_custom_prompt("manual_memory_extraction", config_file, CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR)

# Project manager
PM_PROMPT_DIR               = PROMPT_DIR / "project_manager"
CUS_PM_PROMPT_DIR           = CUS_PROMPT_DIR / "project_manager"
PM_PROMPT_PATH, PM_PROMPT   = setup.prioritise_custom_prompt("manager", config_file, CUS_PM_PROMPT_DIR, PM_PROMPT_DIR)

# Search agent
SEARCH_AGENT_PROMPT_DIR                         = PROMPT_DIR / "search_agent"
CUS_SEARCH_AGENT_PROMPT_DIR                     = CUS_PROMPT_DIR / "search_agent"
SEARCH_OR_NOT_PROMPT_PATH, SEARCH_OR_NOT_PROMPT = setup.prioritise_custom_prompt("search_or_not", config_file, CUS_SEARCH_AGENT_PROMPT_DIR, SEARCH_AGENT_PROMPT_DIR)
QUERY_PROMPT_PATH, QUERY_PROMPT                 = setup.prioritise_custom_prompt("query_generator", config_file, CUS_SEARCH_AGENT_PROMPT_DIR, SEARCH_AGENT_PROMPT_DIR)

# File reader
FILE_READER_PROMPT_DIR                          = PROMPT_DIR / "file_reader"
CUS_FILE_READER_PROMPT_DIR                      = CUS_PROMPT_DIR / "file_reader"
GEN_SUMMARY_PROMPT_PATH, GEN_SUMMARY_PROMPT     = setup.prioritise_custom_prompt("generate_summary", config_file, CUS_FILE_READER_PROMPT_DIR, FILE_READER_PROMPT_DIR)
FILE_OR_NOT_PROMPT_PATH, FILE_OR_NOT_PROMPT     = setup.prioritise_custom_prompt("file_or_not", config_file, CUS_FILE_READER_PROMPT_DIR, FILE_READER_PROMPT_DIR)
GET_FILE_LIST_PROMPT_PATH, GET_FILE_LIST_PROMPT = setup.prioritise_custom_prompt("get_file_list", config_file, CUS_FILE_READER_PROMPT_DIR, FILE_READER_PROMPT_DIR)


# ========================================================
# Memory
# ========================================================

RETRIEVE_MEM_ENTRY_LIMIT = _cfg["memory"]["retrieve_entry_limit"]
AUTO_MEMORY_STORE_TOKENS = _cfg["memory"]["auto_memory_store_enable_at_model_tokens"]


# ========================================================
# File reader
# ========================================================

# Safeguard before enabling auto read files from dropbox
AUTO_READ_DROPBOX_TOKENS = _cfg["file_reader"]["auto_read_dropbox_enable_at_model_tokens"]


# ========================================================
# Web search
# ========================================================

SEARCH_ENG          = _cfg["search"]["engine"]
MAX_RESULTS         = _cfg["search"]["max_results"]
MAX_CHAR_PER_PAGE   = _cfg["search"]["max_char_per_page"]

# Safeguard before enabling auto websearch
AUTO_WEBSEARCH_TOKENS = _cfg["search"]["auto_web_search_enable_at_model_tokens"]


ErrorManage.url_check(config_file, "engine", SEARCH_ENG)
ErrorManage.check_if_value_is_valid(config_file, "max_results", MAX_RESULTS)
