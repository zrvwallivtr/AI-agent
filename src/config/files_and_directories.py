import sys
import shutil
import tomllib
from pathlib import Path


# ========================================================
# STATIC ASSETS
# ========================================================

MAIN_DIR        = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(__file__).resolve().parent.parent.parent
PROMPT_DIR      = MAIN_DIR / "src" / "prompts"
DEFAULT_CONFIG  = MAIN_DIR / "config.toml"


# ========================================================
# ~/.agent_app
# ========================================================

APP_NAME = "agent_app"

APP_DIR         = Path.home() / f".{APP_NAME}"
CONFIG_FILE     = APP_DIR / "config.toml"
UPLOAD_DIR      = APP_DIR / "uploads"
ENV_PATH        = APP_DIR / ".env"
APP_DATA_DIR    = APP_DIR / "data"
CUS_PROMPT_DIR  = APP_DATA_DIR / "prompts"
# PROJECTS_DIR    = APP_DATA_DIR / "projects"
LOG_DIR         = APP_DIR / "logs"
LOG_FILE        = LOG_DIR / "agent.log"


def _create_user_dir(dir_list: list[Path]):
    """Create specified user data directories."""
    for folder in dir_list:
        folder.mkdir(parents=True, exist_ok=True)


_create_user_dir(
    [
        APP_DIR,
        APP_DATA_DIR,
        CUS_PROMPT_DIR,
        # PROJECTS_DIR,
        LOG_DIR
    ]
)


if not CONFIG_FILE.exists() and DEFAULT_CONFIG.exists():
    shutil.copy(DEFAULT_CONFIG, CONFIG_FILE) # Copy default 'config.toml' to '~/.agent_app/'
elif not CONFIG_FILE.exists():
    config_file = DEFAULT_CONFIG # Fallback to default

_cfg = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
