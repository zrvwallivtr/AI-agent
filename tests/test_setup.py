import sys
import shutil
from pathlib import Path
from unittest.mock import patch

from _pytest.monkeypatch import monkeypatch
import pytest

import src.setup as setup


# =============================================================
# get_bundle_root
# =============================================================

def test_get_bundle_root_standard():
    if hasattr(sys, "_MEIPASS"):
        delattr(sys, "_MEIPASS")

    root = setup.get_bundle_root()

    assert isinstance(root, Path)

def test_get_bundle_root_pyinstaller(monkeypatch, test_config):
    fake_meipass = test_config.tmp_dir / "meipass_bundle"
    fake_meipass.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)

    root = setup.get_bundle_root()

    assert root == Path(fake_meipass)


# =============================================================
# get_user_app_dir
# =============================================================

def test_get_user_app_dir(test_config):
    with patch.object(Path, "home", return_value=test_config.tmp_dir):
        app_dir = setup.get_user_app_dir("test_agent_app")

        assert app_dir == test_config.tmp_dir / ".test_agent_app"


# =============================================================
# get_user_app_data_dir
# =============================================================

def test_get_user_app_data_dir(test_config):
    with patch.object(Path, "home", return_value=test_config.tmp_dir):
        data_dir = setup.get_user_app_data_dir("test_agent_app")

        assert data_dir == test_config.tmp_dir / ".test_agent_app" / "data"


# =============================================================
# create_user_dir
# =============================================================

def test_create_user_dir(test_config):
    dir1 = test_config.tmp_dir / "new_dir_1"
    dir2 = test_config.tmp_dir / "nested" / "new_dir_2"

    setup.create_user_dir([dir1, dir2])

    assert dir1.exists() and dir1.is_dir()
    assert dir2.exists() and dir2.is_dir()


# =============================================================
# config_toml_handler
# =============================================================

def _config_toml_handler_implementation(config_file: Path, default_config: Path):
    if not config_file.exists() and default_config.exists():
        shutil.copy(default_config, config_file)

    elif not config_file.exists():
        config_file = default_config

def test_config_toml_handler_copies_default_when_missing(monkeypatch, test_config):
    monkeypatch.setattr(setup, "config_toml_handler", _config_toml_handler_implementation)
    target_config = test_config.tmp_dir / "user_config.toml"
    default_config = test_config.tmp_dir / "default_config.toml"
    default_config.write_text("[models]\nchat = 'mock_model'", encoding="utf-8")

    setup.config_toml_handler(target_config, default_config)

    assert target_config.exists()
    assert target_config.read_text(encoding="utf-8") == default_config.read_text(encoding="utf-8")

def test_config_toml_handler_preserves_existing_config(monkeypatch, test_config):
    monkeypatch.setattr(setup, "config_toml_handler", _config_toml_handler_implementation)
    target_config = test_config.tmp_dir / "existing_user_config.toml"
    default_config = test_config.tmp_dir / "default_config.toml"

    target_config.write_text("custom_user_setting = true", encoding="utf-8")
    default_config.write_text("default_setting = false", encoding="utf-8")

    setup.config_toml_handler(target_config, default_config)

    assert target_config.read_text(encoding="utf-8") == "custom_user_setting = true"


# =============================================================
# normalise_to_integer
# =============================================================

@pytest.mark.parametrize("val, expected", [
    (100, 100),
    (100.0, 100),
    ("4000", 4000),
    (None, None),
])
def test_normalise_to_integer(val, expected):
    assert setup._normalise_to_integer(val) == expected

@pytest.mark.parametrize("val, expected", [
    ("invalid_string", "Error: Invalid value 'invalid_string'."),
])
def test_normalise_to_integer_error(val, expected):
    assert setup._normalise_to_integer(val) == expected

# =============================================================
# set_model_tokens
# =============================================================

def test_set_model_tokens_known_model(test_config):
    cfg = {
        "models": {
            "chat": "mock_model",
            "chat_max_tokens": 4000
        }
    }
    model, tokens = setup.set_model_tokens(cfg, test_config.config_file, "chat", "chat_max_tokens")

    assert model == "mock_model"
    assert tokens == 4000

@patch("src.setup.ErrorManage")
def test_set_model_tokens_unknown_model_missing_max_tokens(mock_error_manage, test_config):
    cfg = {
        "models": {
            "chat": "unlisted_model"
        }
    }

    setup.set_model_tokens(cfg, test_config.config_file, "chat", "chat_max_tokens")

    mock_error_manage.model_not_found.assert_called_once_with(
        test_config.config_file, "chat", "chat_max_tokens"
    )

@patch("src.setup.ErrorManage")
def test_set_model_tokens_unknown_model_validates_tokens(mock_error_manage, test_config):
    cfg = {
        "models": {
            "chat": "unlisted_model",
            "chat_max_tokens": 8000
        }
    }

    model, tokens = setup.set_model_tokens(cfg, test_config.config_file, "chat", "chat_max_tokens")

    assert model == "unlisted_model"
    assert tokens == 8000
    mock_error_manage.check_if_value_is_valid.assert_called_once_with(
        test_config.config_file, "chat_max_tokens", 8000
    )


# =============================================================
# prompt_path_handler
# =============================================================

@patch("src.setup.ErrorManage")
def test_prompt_path_handler(mock_error_manage, test_config):
    prompt_dir = test_config.tmp_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    prompt_file = prompt_dir / "system.txt"
    prompt_file.write_text("   You are a helpful assistant.   \n", encoding="utf-8")

    path, content = setup._prompt_path_handler("system.txt", test_config.config_file, prompt_dir)

    assert path == prompt_file
    assert content == "You are a helpful assistant."
    mock_error_manage.check_if_system_prompt_exist.assert_called_once_with(
        test_config.config_file, "system.txt", prompt_file
    )


# =============================================================
# prioritise_custom_prompt
# =============================================================

def test_prioritise_custom_prompt_uses_custom(test_config, tmp_path):
    custom_dir = tmp_path / "cus_prompts"
    default_dir = tmp_path / "def_prompts"
    custom_dir.mkdir(exist_ok=True)
    default_dir.mkdir(exist_ok=True)

    (custom_dir / "system.txt").write_text("Custom system prompt", encoding="utf-8")
    (default_dir / "system.txt").write_text("Default system prompt", encoding="utf-8")

    with patch("src.setup.ErrorManage"):
        path, content = setup.prioritise_custom_prompt(
            "system.txt", test_config.config_file, custom_dir, default_dir
        )

        assert path == custom_dir / "system.txt"
        assert content == "Custom system prompt"

def test_prioritise_custom_prompt_falls_back_to_default(test_config, tmp_path):
    custom_dir = tmp_path / "cus_prompts"
    default_dir = tmp_path / "def_prompts"
    custom_dir.mkdir(exist_ok=True)
    default_dir.mkdir(exist_ok=True)

    (default_dir / "system.txt").write_text("Default system prompt", encoding="utf-8")

    with patch("src.setup.ErrorManage"):
        path, content = setup.prioritise_custom_prompt(
            "system.txt", test_config.config_file, custom_dir, default_dir
        )

        assert path == default_dir / "system.txt"
        assert content == "Default system prompt"
