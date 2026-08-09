import tomllib
import pytest
from _pytest.python import path_matches_patterns
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import src.config as config_module
import src.setup as setup_module
import src.agent.chat as chat_module
import src.agent.memory as memory_module
import src.tools.file_reader as file_reader_module
import src.tools.project_manager as project_manager_module
import src.tools.search as search_module
from src import config


# ============================================================
# Variables
# ============================================================

@dataclass
class TestConfig:
    tmp_dir: Path

    data_dir: Path
    app_data_dir: Path
    cus_prompt_dir: Path
    chats_dir: Path
    chromadb_dir: Path
    projects_dir: Path
    dropbox_dir: Path

    default_path: Path
    default_chat_history_path: Path

    config_file: Path

    _cfg: dict[str, Any]

    model: str
    embed_model: str
    pm_model: str

    # model_max_tokens: int
    # mem_max_tokens: int
    # pm_max_tokens: int

    # sys_prompt_dir: Path
    # cus_sys_prompt_dir: Path

@pytest.fixture(scope="session")
def test_config(tmp_path_factory) -> TestConfig:
    tmp_dir = tmp_path_factory.mktemp("agent_app")

    data_dir        = tmp_dir / ".agent_app"
    app_data_dir    = data_dir / "test_data"
    cus_prompt_dir  = app_data_dir / "prompts"
    chats_dir       = app_data_dir / "chats"
    chromadb_dir    = app_data_dir / "chroma"
    projects_dir    = app_data_dir / "projects"
    dropbox_dir     = app_data_dir / "dropbox_dir"

    data_dir.mkdir(parents=True, exist_ok=True)
    app_data_dir.mkdir(parents=True, exist_ok=True)
    cus_prompt_dir.mkdir(parents=True, exist_ok=True)
    chats_dir.mkdir(parents=True, exist_ok=True)
    chromadb_dir.mkdir(parents=True, exist_ok=True)
    projects_dir.mkdir(parents=True, exist_ok=True)
    dropbox_dir.mkdir(parents=True, exist_ok=True)

    default_path                = chats_dir / "default_session" / "chat.json"
    default_chat_history_path   = chats_dir / "default_session" / "chat_history.json"

    config_file = data_dir / "config.toml"

    config_file.write_text("""
[models]
#chat                = "gemma3:1b"
chat                = "ministral_3b:latest"
memory              = "nomic-embed-text"
project_manager     = "gemma3:1b"

# chat_max_tokens     =
# pm_max_tokens       =

[load_system_prompt]
chat            = "system"
# memory          = "memory"
# project_manager = "project_manager"

# [ollama]
# url = "http://localhost:11434"

[memory]
retrieve_entry_limit = 3

[file_reader]
model_max_tokens    = 128000

[search]
engine              = "http://localhost:8080/search"
max_results         = 3
max_char_per_page   = 3000
model_max_tokens    = 128000
    """)

    _cfg = tomllib.loads(config_file.read_text(encoding="utf-8"))

    model       = "mock_model"
    embed_model = "mock_embed_model"
    pm_model    = "mock_pm_model"

    return TestConfig(
        tmp_dir=tmp_dir,

        data_dir=data_dir,
        app_data_dir=app_data_dir,
        cus_prompt_dir=cus_prompt_dir,
        chats_dir=chats_dir,
        chromadb_dir=chromadb_dir,
        projects_dir=projects_dir,
        dropbox_dir=dropbox_dir,

        default_path=default_path,
        default_chat_history_path=default_chat_history_path,

        config_file=config_file,

        _cfg=_cfg,

        model=model,
        embed_model=embed_model,
        pm_model=pm_model,
    )


# ============================================================
# Config
# ============================================================

@pytest.fixture(autouse=True)
def mock_config(monkeypatch: pytest.MonkeyPatch, test_config: TestConfig):

    # ====================================
    # Writable persistent data
    # (stored in ~/tmp_path/.agent_app)
    # ====================================

    # tmp_dir / .agent_app
    monkeypatch.setattr(config_module, "DATA_DIR", test_config.data_dir)

    # tmp_dir / .agent_app / data
    monkeypatch.setattr(config_module, "APP_DATA_DIR", test_config.app_data_dir)

    monkeypatch.setattr(config_module, "APP_DATA_DIR", test_config.app_data_dir)
    monkeypatch.setattr(config_module, "CUS_PROMPT_DIR", test_config.cus_prompt_dir)
    monkeypatch.setattr(config_module, "CHATS_DIR", test_config.chats_dir)
    monkeypatch.setattr(config_module, "CHROMADB_DIR", test_config.chromadb_dir)
    monkeypatch.setattr(config_module, "PROJECTS_DIR", test_config.projects_dir)
    monkeypatch.setattr(config_module, "DROPBOX_DIR", test_config.dropbox_dir)

    monkeypatch.setattr(config_module, "DEFAULT_PATH", test_config.default_path)
    monkeypatch.setattr(config_module, "DEFAULT_CHAT_HISTORY_PATH", test_config.default_chat_history_path)

    # ====================================
    # Load user config file (config.toml)
    # ====================================

    monkeypatch.setattr(config_module, "config_file", test_config.config_file)

    def _config_toml_handler(config_file: Path):
        return None

    monkeypatch.setattr(setup_module, "config_toml_handler", _config_toml_handler)

    monkeypatch.setattr(config_module, "_cfg", test_config._cfg)

    # ====================================
    # Model selection
    # ====================================

    monkeypatch.setattr(config_module, "MODEL", test_config.model)
    monkeypatch.setattr(config_module, "MODEL_MAX_TOKENS", 4000)

    monkeypatch.setattr(config_module, "EMBED_MODEL", test_config.embed_model)
    monkeypatch.setattr(config_module, "MEM_MAX_TOKENS", 2000)

    monkeypatch.setattr(config_module, "PM_MODEL", test_config.pm_model)
    monkeypatch.setattr(config_module, "PM_MAX_TOKENS", 4000)

    # ====================================
    # Prompts
    # ====================================

    # ====================================
    # Limits
    # ====================================

    monkeypatch.setattr(config_module, "AUTO_READ_DROPBOX_TOKENS", 128000)
    monkeypatch.setattr(config_module, "AUTO_WEBSEARCH_TOKENS", 128000)


# ============================================================
# Model list
# ============================================================

@pytest.fixture(autouse=True)
def mock_model_list(monkeypatch: pytest.MonkeyPatch):
    model_max = {
        "mock_model":       4000,
        "custom_model":     4000,
        "mock_embed_model": 2000,
    }
    monkeypatch.setattr(config_module, "MODEL_MAX", model_max)
    monkeypatch.setattr("src.agent.tokens_handler.MODEL_MAX", model_max)


# ============================================================
# Ollama API
# ============================================================

@pytest.fixture(autouse=True)
def mock_ollama(monkeypatch: pytest.MonkeyPatch):
    fake_models = {
        "models": [
            {"model": "mock_model"},
            {"model": "custom_model"},
            {"model": "mock_embed_model"},
        ]
    }
    monkeypatch.setattr("ollama.list", lambda: fake_models)


# ============================================================
# Chat.__init__
# ============================================================

# @pytest.fixture(autouse=True)
# def _session_paths(monkeypatch: pytest.MonkeyPatch, test_config: TestConfig):
#     monkeypatch.setattr(config_module, "DEFAULT_PATH", test_config.default_path)
#     monkeypatch.setattr(config_module, "DEFAULT_CHAT_HISTORY_PATH", test_config.default_chat_history_path)

@pytest.fixture(autouse=True)
def mock_chat_class(monkeypatch: pytest.MonkeyPatch, test_config: TestConfig):

    def _mock_session_path(session: str | None = None) -> tuple[Path, Path]:
        if session is not None:
            return test_config.chats_dir / f"{session}"/ "chat.json", test_config.chats_dir / f"{session}"/ "chat_history.json"
        else:
            return test_config.default_path, test_config.default_chat_history_path

    monkeypatch.setattr(chat_module, "_session_path", _mock_session_path)


# ============================================================
# FileReader.__init__
# ============================================================

@pytest.fixture(autouse=True)
def mock_file_reader_class(monkeypatch: pytest.MonkeyPatch, test_config: TestConfig):

    def _mock_session_path(session: str | None = None) -> Path:
        if session is not None:
            return test_config.dropbox_dir / session
        return test_config.dropbox_dir / "chat"

    monkeypatch.setattr(file_reader_module, "_session_path", _mock_session_path)
