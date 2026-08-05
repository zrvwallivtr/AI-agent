import json
from _pytest.monkeypatch import monkeypatch
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import src.agent.chat as chat_module
from src.agent.chat import Chat, _session_path
from src.agent.llm import LLM
from src.config import MODEL, CHAT_DIR, DEFAULT_PATH, DEFAULT_CHAT_HISTORY_PATH


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    mock_chat_dir = tmp_path / "data" / "chat"
    mock_chat_dir.mkdir(parents=True, exist_ok=True)

    # Override paths
    monkeypatch.setattr(chat_module, "CHAT_DIR", mock_chat_dir)
    monkeypatch.setattr(chat_module, "DEFAULT_PATH", mock_chat_dir / "chat.json")
    monkeypatch.setattr(chat_module, "DEFAULT_CHAT_HISTORY_PATH", mock_chat_dir / "chat_history.json")

    monkeypatch.setattr(chat_module, "CHAT_PROMPT", "You are a helpful assistant.")

    return mock_chat_dir


# =============================================
# Path routing tests
# =============================================

def test_session_path_with_custom_name(setup_test_env):
    active, history         = _session_path("test")
    assert active.name      == "test.json"
    assert history.name     == "test_chat_history.json"
    assert active.parent    == setup_test_env

def test_session_path_fall_back_defaults():
    active, history = _session_path()
    assert active   == chat_module.DEFAULT_PATH
    assert history  == chat_module.DEFAULT_CHAT_HISTORY_PATH

def test_chat_initialise():
    chat = Chat(session="new_session")

    # Check initiation elements
    # - inital system prompt should exists
    assert len(chat.all()) == 1
    assert chat.all()[0]["role"] == "system"
    assert chat.all()[0]["state"] == "internal"

def test_chat_load_existing_history_file(setup_test_env):
    session_path = setup_test_env / "existing_session.json"
    session_data = [
        {"role": "system", "content": "You are a helpful assistant.", "state": "internal"},
        {"role": "user", "content": "What is Python?", "state": "external"}
    ]

    # Write session file
    session_path.write_text(json.dumps(session_data))

    # Initialise
    chat = Chat(session="existing_session")

    # Ensure session file saved to the correct location
    assert len(chat.all())          == 2
    assert chat.all()[1]["content"] == "What is Python?"

def test_chat_auto_update_modified_system_prompt(setup_test_env, monkeypatch):
    # 1. Save session file with old system prompt config
    session_path    = setup_test_env / "updated_sys_prompt_session.json"
    historical_data = [
        {"role": "system", "content": "Old system prompt", "state": "internal"},
        {"role": "user", "content": "What is Python?", "state": "external"}
    ]
    session_path.write_text(json.dumps(historical_data))

    # 2. Modify system prompt after session is created
    monkeypatch.setattr(chat_module, "CHAT_PROMPT", "New system prompt")

    # 3. Initialise
    chat = Chat(session="updated_sys_prompt_session")

    # Ensure chat session file is synchronized
    assert chat.all()[0]["content"] == "New system prompt"

    # Ensure messages is kept
    assert chat.all()[1]["content"] == "What is Python?"

def test_to_llm_trims_metadata():
    chat = Chat(session="trim_test")
    chat.append_message(role="user", content="Hello", state="external")
    chat.append_message(role="assistant", content="Hi", state="external", prompt_tokens=10, output_tokens=5)

    trimmed_messages = chat.to_llm()

    # Ensure messages are trimmed into proper format
    assert "state" not in trimmed_messages[1]
    assert "prompt_tokens" not in trimmed_messages[2]
    assert trimmed_messages[1] == {"role": "user", "content": "Hello"}

# =============================================
# Check if session cleared properly
# =============================================

def test_clear_and_file_deletion(setup_test_env):
    chat = Chat(session="deletion_test")
    chat.append_message(role="user", content="Test deletion.", state="external")

    # Verify unlinking components
    active_path     = setup_test_env / "deletion_test.json"
    history_path    = setup_test_env / "deletion_test_chat_history.json"

    assert active_path.exists()
    assert history_path.exists()

    # Deletion
    response = chat.clear_active_conv()
    assert "Deleted session" in response
    assert not active_path.exists()

    # Test error code
    error_response = chat.clear_active_conv()
    assert "Error: Session" in error_response

# =============================================
# Ensure correct compression steps
# =============================================

def test_compression(monkeypatch):
    chat = Chat(session="compression_test")
    chat.append_message(role="user", content="Message 1", state="external")
    chat.append_message(role="assistant", content="Message 2", state="external")

    # Mock LLM response
    mock_response = ("This is condensed context summary statement.", 50, 20)
    monkeypatch.setattr(LLM, "model_response", MagicMock(return_value=mock_response))

    chat.compression(model="mock_model")

    records = chat.all()
    assert len(records) == 3
    assert records[1]["content"] == "Summarise context from previous conversations."
    assert "[CONVERSATION SUMMARY]" in records[2]["content"]
    assert "This is condensed context summary statement." in records[2]["content"]

def test_compression_failure(monkeypatch):
    chat = Chat(session="compression_fail_test")
    chat.append_message(role="user", content="Message 1", state="external")

    # Mock failed compression attempt
    mock_fail_response = ("", 10, 0)
    monkeypatch.setattr(LLM, "model_response", MagicMock(return_value=mock_fail_response))

    chat.compression(model="mock_model")

    # Ensure content unchanged when compression failed
    assert len(chat.all()) == 2
    assert chat.all()[1]["content"] == "Message 1"
