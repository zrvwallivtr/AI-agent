import sys
from pathlib import Path

from src.config.files_and_directories import PROMPT_DIR, CUS_PROMPT_DIR, CONFIG_FILE


def _prompt_path_handler(
    filename: str, config_file: Path, dir: Path
) -> tuple[Path, str]:
    """
    Check if file exist, then read file contents
    and return it's path and contents as string.
    """
    path = dir / filename
    if not path.exists():
        # logger.critical(
        #     "System prompt file does not exists: field=%s, path=%s",
        #     field_name,
        #     path
        # )
        print(f"Error in {config_file}:")
        print(f"'{filename}' prompt file not found: '{path}'")
        sys.exit(1)

    prompt = (path).read_text(encoding="utf-8").strip()
    return path, prompt


def _prioritise_custom_prompt(
    filename: str, config_file: Path, custom_prompt_dir: Path, default_prompt_dir: Path
) -> tuple[Path, str]:
    """Returns active directory and prompt if exists, else, return default directory and prompt."""
    custom_prompt_path = custom_prompt_dir / filename
    path = Path(custom_prompt_path)

    if path.exists():
        return _prompt_path_handler(filename, config_file, custom_prompt_dir)

    return _prompt_path_handler(filename, config_file, default_prompt_dir)


# ===========================================================
# SYSTEM PROMPT
# ===========================================================

SYS_PROMPT_DIR              = PROMPT_DIR / "system"
CUS_SYS_PROMPT_DIR          = CUS_PROMPT_DIR / "system"

SYS_PROMPT_PATH, SYS_PROMPT = _prioritise_custom_prompt(
    "standard", CONFIG_FILE, CUS_SYS_PROMPT_DIR, SYS_PROMPT_DIR
)


# ===========================================================
# COMPRESSION PROMPT
# ===========================================================

COMPRESS_PROMPT_DIR     = PROMPT_DIR / "chat_compression"
CUS_COMPRESS_PROMPT_DIR = CUS_PROMPT_DIR / "chat_compression"

COMPRESS_PROMPT_PATH, COMPRESS_PROMPT = _prioritise_custom_prompt(
    "instructions", CONFIG_FILE, CUS_COMPRESS_PROMPT_DIR, COMPRESS_PROMPT_DIR
)


# ===========================================================
# MEMORY PROMPT
# ===========================================================

MEM_PROMPT_DIR              = PROMPT_DIR / "memory"
CUS_MEM_PROMPT_DIR          = CUS_PROMPT_DIR / "memory"

# Auto memory extraction
MEM_PROMPT_PATH, MEM_PROMPT = _prioritise_custom_prompt(
    "memory_agent", CONFIG_FILE, CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR
)

# Manual memory extraction
MEM_MANUAL_PROMPT_PATH, MEM_MANUAL_PROMPT = _prioritise_custom_prompt(
    "manual_memory_extraction", CONFIG_FILE, CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR
)

# Interpret retrieved memory entries
MEM_RECALL_INTERPRET_PATH, MEM_RECALL_INTERPRET_PROMPT = _prioritise_custom_prompt(
    "memory_recall_interpreter", CONFIG_FILE, CUS_MEM_PROMPT_DIR, MEM_PROMPT_DIR
)

# ===========================================================
# PROJECT MANAGER PROMPT
# ===========================================================

PM_PROMPT_DIR       = PROMPT_DIR / "project_manager"
CUS_PM_PROMPT_DIR   = CUS_PROMPT_DIR / "project_manager"

PM_PROMPT_PATH, PM_PROMPT = _prioritise_custom_prompt(
    "manager", CONFIG_FILE, CUS_PM_PROMPT_DIR, PM_PROMPT_DIR
)


# ===========================================================
# SEARCH AGENT PROMPT
# ===========================================================

SEARCH_AGENT_PROMPT_DIR     = PROMPT_DIR / "search_agent"
CUS_SEARCH_AGENT_PROMPT_DIR = CUS_PROMPT_DIR / "search_agent"

# Search or not
SEARCH_OR_NOT_PROMPT_PATH, SEARCH_OR_NOT_PROMPT = _prioritise_custom_prompt(
    "search_or_not", CONFIG_FILE, CUS_SEARCH_AGENT_PROMPT_DIR, SEARCH_AGENT_PROMPT_DIR
)

# Query generation
QUERY_PROMPT_PATH, QUERY_PROMPT = _prioritise_custom_prompt(
    "query_generator", CONFIG_FILE, CUS_SEARCH_AGENT_PROMPT_DIR, SEARCH_AGENT_PROMPT_DIR
)

