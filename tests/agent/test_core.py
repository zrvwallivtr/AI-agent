import json

from unittest import mock
from unittest.mock import patch
import pytest
from unittest.mock import MagicMock, call

import src.agent.core as core_module
import src.agent.chat as chat_module
from src.agent.core import Agent, _last_token_usage, _detect_cmd
from src.agent.llm import LLM


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    # Mock memory directory
    mock_memory_dir = tmp_path / "data" / "chroma"
    mock_memory_dir.mkdir(parents=True, exist_ok=True)

    # Mock chat directory
    mock_chat_dir = tmp_path / "data" / "chat"
    mock_chat_dir.mkdir(parents=True, exist_ok=True)

    # Mock paths
    mock_chat_file      = mock_chat_dir / "chat.json"

    # Override paths
    monkeypatch.setattr(core_module, "CHROMADB_DIR", mock_memory_dir)
    monkeypatch.setattr(chat_module, "CHAT_DIR", mock_chat_dir)
    monkeypatch.setattr(chat_module, "DEFAULT_PATH", mock_chat_file)

    # Override configs
    monkeypatch.setattr(core_module, "MEM_MANUAL_PROMPT", "Manual memory retrieve prompt")
    monkeypatch.setattr(core_module, "MAX_RESULTS", 1)

    return mock_chat_file

@pytest.fixture
def mock_dependentcies(monkeypatch):
    mocks = {
            "chat":         MagicMock(),
            "memory":       MagicMock(),
            "tokens":       MagicMock(),
            "search_agent": MagicMock(),
            "file_reader":  MagicMock(),
    }

    # Overwrites all the actual classes
    monkeypatch.setattr(core_module, "Chat", MagicMock(return_value=mocks["chat"]))
    monkeypatch.setattr(core_module, "Memory", MagicMock(return_value=mocks["memory"]))
    monkeypatch.setattr(core_module, "Tokens", MagicMock(return_value=mocks["tokens"]))
    monkeypatch.setattr(core_module, "SearchAgent", MagicMock(return_value=mocks["search_agent"]))
    monkeypatch.setattr(core_module, "FileReader", MagicMock(return_value=mocks["file_reader"]))

    # Set standard value
    monkeypatch.setattr(core_module, "MODEL_MAX_TOKENS", 4000)

    return mocks


# ========================================================================
# Tokens
# ========================================================================

def test_last_token_values_when_msg_final_is_assistant():
    messages = [
        {"role": "system", "content": "You are a helpful assistant." ,"state": "internal"},
        {"role": "user", "content": "Message 1", "state": "external"},
        {"role": "assistant", "content": "Message 2", "state": "external", "prompt_tokens": 10, "output_tokens": 30}
    ]

    # Get latest token values
    prompt_tokens, output_tokens = _last_token_usage(messages)

    assert prompt_tokens == 10
    assert output_tokens == 30

def test_last_token_values_when_msg_final_is_usr():
    messages = [
        {"role": "system", "content": "You are a helpful assistant." ,"state": "internal"},
        {"role": "user", "content": "Message 1", "state": "external"},
        {"role": "assistant", "content": "Message 2", "state": "external", "prompt_tokens": 10, "output_tokens": 30},
        {"role": "user", "content": "Message 3", "state": "external"},
    ]

    # Get latest token values
    prompt_tokens, output_tokens = _last_token_usage(messages)

    # Ensure function return the latest assistant token usage
    assert prompt_tokens == 10
    assert output_tokens == 30

# ========================================================================
# Utility test
# ========================================================================

def test_parsing_with_prompt():
    cmd, prompt = _detect_cmd("/forget This is a message.")

    assert cmd      == "/forget"
    assert prompt   == "This is a message."

def test_parsing_without_prompt():
    cmd, prompt = _detect_cmd("/recall  ")

    assert cmd      == "/recall"
    assert prompt   == ""

def test_parsing_without_command():
    cmd, prompt = _detect_cmd("This message contains no commands.")

    assert cmd is None
    assert prompt   == "This message contains no commands."

# ========================================================================
# Chat compression
# ========================================================================

@patch("src.agent.core.Agent._validate_model")
def test_manage_token_budget_auto_compression_trigger(mock_validate, mock_dependentcies):
    mocks = mock_dependentcies
    agent = Agent(model="mock_model")

    mocks["tokens"].max_tokens                          = 4000

    # Mock token usage
    mocks["tokens"].count_history_tokens.return_value   = 35000
    agent._manage_token_budget(prompt="This is message will exceed max token usage threshold.")

    mocks["chat"].compression.assert_called_once_with("mock_model")

# ========================================================================
# Command
# ========================================================================

def test_command_forget_cancel(mock_dependentcies, monkeypatch):
    mocks = mock_dependentcies
    agent = Agent()

    # Bypass not found error in '_cmd_forget' function
    mocks["memory"].get_exact_match.return_value = ("mem_id_123", "This is a memory.")

    # Mock user press 'c' to cancel '/forget' command
    monkeypatch.setattr("builtins.input", lambda _: "c")

    output_msg = agent._cmd_forget(prompt="This message does not matter in this test.")

    # Ensure memory entry not deleted from database
    assert output_msg == "Deletion cancelled."
    mocks["memory"].delete_from_db.assert_not_called()

def test_command_forget_confirm(mock_dependentcies, monkeypatch):
    mocks = mock_dependentcies
    agent = Agent()

    # Bypass not found error in '_cmd_forget' function
    mocks["memory"].get_exact_match.return_value = ("mem_id_1234", "This is a memory.")

    # Mock user press enter to continue '/forget' command
    monkeypatch.setattr("builtins.input", lambda _:"")

    output_msg = agent._cmd_forget(prompt="This message does not matter in this test.")

    # Ensure memory entry deleted from database
    assert output_msg == "Entry deleted."
    mocks["memory"].delete_from_db.assert_called_once_with(["mem_id_1234"])

def test_command_memorise_undo(mock_dependentcies, monkeypatch):
    mocks = mock_dependentcies
    agent = Agent()

    # Simulate an entry is saved to the database
    mocks["memory"].extract_to_db.return_value = ["entry_id"]
    mocks["memory"].get_entries_by_ids.return_value = [{"category": "fact", "content": "Memory entry"}]

    # Mock user press 'u' to undo '/memorise' command
    monkeypatch.setattr("builtins.input", lambda _: "u")

    output_msg = agent._cmd_memorise(prompt="This message does not matter in this test.")

    # Ensure memory entry deleted from database
    assert output_msg == "Entry deleted."
    mocks["memory"].delete_from_db.assert_called_once_with(["entry_id"])

def test_command_recall_formatting(mock_dependentcies):
    mocks = mock_dependentcies
    agent = Agent()

    # Simulate two entries are retrieved from the database
    mocks["memory"].retrieve_relevant_entry.return_value = [
        {"category": "stack", "content": "Message 1"},
        {"category": "preference", "content": "Message 2"}
    ]

    output = agent._cmd_recall(prompt="User setup data")

    # Ensure all retrieved memory entires returned
    assert "Found matching entries in long-term memory:" in output
    assert "- [stack] Message 1" in output
    assert "- [preference] Message 2" in output
    mocks["chat"].append_message.assert_any_call("assistant", output, "external")

# ========================================================================
# Standard workflows
# ========================================================================

@patch("src.agent.core.Agent._validate_model")
def test_ask_standard_llm_response(mock_validate, setup_test_env, mock_dependentcies, monkeypatch):
    """
    Only testing by asking the model a question.

    The 'MODEL_MAX_TOKENS' in 'mock_dependentcies'
    will not run any of the auto functions:
    - _toggle_auto_web_search
    """
    mocks = mock_dependentcies
    agent = Agent(model="mock_model")

    # Setup chat history
    messages = [
        {"role": "system", "content": "You are a helpful assistant." ,"state": "internal"},
        {"role": "user", "content": "Message 1", "state": "external"},
        {"role": "assistant", "content": "Message 2", "state": "external", "prompt_tokens": 10, "output_tokens": 30}
    ]

    mocks["chat"].to_llm.return_value = [dict(m) for m in messages]

    mocks["memory"].add_memory_entries.return_value = ""
    mocks["tokens"].max_tokens = 4000
    mocks["tokens"].count_history_tokens.return_value = 100

    # Mock LLM response
    answer              = "This is a model response answer."
    prompt_tokens       = 45
    output_tokens       = 12
    mock_llm_response   = MagicMock(return_value=(answer, prompt_tokens, output_tokens))
    monkeypatch.setattr(LLM, "model_response", mock_llm_response)

    # Start
    prompt = "This is a message."
    agent.ask(prompt=prompt)

    # Verifications

    mock_llm_response.assert_called_once()

    called_messages, called_model_kwargs = mock_llm_response.call_args
    passed_messages = called_messages[0]

    assert passed_messages[-1] == {
        "role": "user",
        "content": f"\nUser input:{prompt}"
    }

    mocks["chat"].append_message.assert_has_calls([
        call("user", prompt, "external"),
        call("assistant", answer, "external", prompt_tokens, output_tokens)
    ], any_order=False)


# ========================================================================
# Test commands in workflows
# ========================================================================
# "/forget Forget request."
# "/memorise Memorise request."
# "/recall Recall request."
# "/search Search request."


# ========================================================================
# Test autofunctions
# ------------------
#
# Test if the messages is formatted correctly:
# - Starts with 'system prompt', alternating between 'user'
#   and 'assistant'.
# - All the contexts are combined into a single entry and
#   send to the model along with the prompt.
# - The context are labled clearly.
# ========================================================================
# 1. Auto add memory (on), Auto read dropbox (on), Auto web search (on)
# 2. Auto add memory (on), Auto read dropbox (off), Auto web search (off)
# 3. Auto add memory (off), Auto read dropbox (on), Auto web search (off)
# 4. Auto add memory (off), Auto read dropbox (off), Auto web search (on)
