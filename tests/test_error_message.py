import pytest
from unittest import mock
from pathlib import Path

from src.error_manage import ErrorManage


# =========================================================
# model_not_found
# =========================================================

def test_model_not_found(test_config, capsys):
    config_file = test_config.tmp_dir / "config.toml"
    
    with pytest.raises(SystemExit) as exc_info:
        ErrorManage.model_not_found(
            config_file=config_file,
            model_type="mock_model",
            model_max_tokens="CUSTOM_MODEL_TOKENS"
        )

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert f"Error in {config_file}:" in captured.out
    assert "User selected 'mock_model' model does not match" in captured.out
    assert "add 'CUSTOM_MODEL_TOKENS' value." in captured.out


# =========================================================
# check_if_value_is_valid
# =========================================================

@pytest.mark.parametrize("valid_value", [100, 3.14, 0, -5.5])
def test_check_if_value_is_valid_with_correct_values(test_config, valid_value):
    config_file = test_config.tmp_dir / "config.toml"

    ErrorManage.check_if_value_is_valid(config_file, "max_tokens", valid_value)

@pytest.mark.parametrize("invalid_value", ["100", None, [], {}, True])
def test_check_if_value_is_valid_failure(test_config, invalid_value, capsys):
    config_file = test_config.tmp_dir / "config.toml"

    with pytest.raises(SystemExit) as exc_info:
        ErrorManage.check_if_value_is_valid(config_file, "temperature", invalid_value)

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert f"Error in {config_file}:" in captured.out
    assert f"'temperature' must be a number, got '{invalid_value}' instead." in captured.out


# =========================================================
# url_check
# =========================================================

@pytest.mark.parametrize("valid_url", [
    "http://localhost:8080/search",
    "https://api.openai.com/v1",
    "http://127.0.0.1:5000",
])
def test_url_check_valid(test_config, valid_url):
    config_file = test_config.tmp_dir / "config.toml"

    ErrorManage.url_check(config_file, "search_api_url", valid_url)


@pytest.mark.parametrize("invalid_url", [
    "ftp://localhost:8080",
    "localhost:8080",
    "http://",
    "not_a_url",
    "https://",
])
def test_url_check_invalid(test_config, invalid_url, capsys):
    config_file = test_config.tmp_dir / "config.toml"

    with pytest.raises(SystemExit) as exc_info:
        ErrorManage.url_check(config_file, "endpoint_url", invalid_url)

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert f"Error in {config_file}:" in captured.out
    assert f"'endpoint_url' must be a valid URL, got '{invalid_url}' instead." in captured.out


# =========================================================
# check_if_system_prompt_exist
# =========================================================

def test_check_if_system_prompt_exist_success(test_config):
    config_file = test_config.tmp_dir / "config.toml"
    prompt_file = test_config.tmp_dir / "system_prompt.txt"
    prompt_file.write_text("You are a helpful assistant.", encoding="utf-8")

    ErrorManage.check_if_system_prompt_exist(config_file, "system_prompt", prompt_file)


def test_check_if_system_prompt_exist_failure(test_config, capsys):
    config_file = test_config.tmp_dir / "config.toml"
    missing_prompt_file = test_config.tmp_dir / "non_existent_prompt.txt"

    with pytest.raises(SystemExit) as exc_info:
        ErrorManage.check_if_system_prompt_exist(config_file, "system_prompt", missing_prompt_file)

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert f"Error in {config_file}:" in captured.out
    assert f"'system_prompt' prompt file not found: '{missing_prompt_file}'" in captured.out
