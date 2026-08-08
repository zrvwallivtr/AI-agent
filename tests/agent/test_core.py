import json
from re import search

import pytest
from unittest import mock
from unittest.mock import patch
from unittest.mock import MagicMock, call

import src.agent.core as core_module
import src.agent.chat as chat_module
import src.agent.memory as memory_module
from src.agent.core import Agent, _last_token_usage, _detect_cmd
from src import config
from tools.search import SearchAgent


AVAILABLE_CMDS  = ["/forget", "/memorise", "/recall", "/search"]



# ========================================================================
# _last_token_usage
# ========================================================================

messages_assistant_final = [
    {"role": "system", "content": "You are a helpful assistant." ,"state": "internal"},
    {"role": "user", "content": "Message 1", "state": "external"},
    {"role": "assistant", "content": "Message 2", "state": "external", "prompt_tokens": 10, "output_tokens": 30}
]

messages_user_final = [
    {"role": "system", "content": "You are a helpful assistant." ,"state": "internal"},
    {"role": "user", "content": "Message 1", "state": "external"},
    {"role": "assistant", "content": "Message 2", "state": "external", "prompt_tokens": 10, "output_tokens": 30},
    {"role": "user", "content": "Message 3", "state": "external"},
]

def test_last_token_values_when_msg_final_is_assistant():
    # Get latest token values
    prompt_tokens, output_tokens = _last_token_usage(messages_assistant_final)

    assert prompt_tokens == 10
    assert output_tokens == 30

def test_last_token_values_when_msg_final_is_usr():
    # Get latest token values
    prompt_tokens, output_tokens = _last_token_usage(messages_user_final)

    # Ensure function return the latest assistant token usage
    assert prompt_tokens == 10
    assert output_tokens == 30


# ========================================================================
# _detect_cmd
# ========================================================================

cmd_only    = f"{AVAILABLE_CMDS[0]} "
prompt_only = "This message contains no commands."

cmd_and_prompt  = f"{cmd_only} {prompt_only}"
without_prompt  = f"{cmd_only} "
without_cmd     = f"{prompt_only}"

def test_parsing_with_prompt():
    cmd, prompt = _detect_cmd(cmd_and_prompt)

    assert cmd      == f"{AVAILABLE_CMDS[0]}"
    assert prompt   == prompt_only

def test_parsing_without_prompt():
    cmd, prompt = _detect_cmd(without_prompt)

    assert cmd      == f"{AVAILABLE_CMDS[0]}"
    assert prompt   == ""

def test_parsing_without_command():
    cmd, prompt = _detect_cmd(without_cmd)

    assert cmd is None
    assert prompt   == prompt_only


# ========================================================================
# Agent.__init__
# ========================================================================

def test_agent_init_default(test_config):
    agent = Agent()

    chat_path       = test_config.tmp_dir / ".agent_app" / "test_data" / "chats" / "default_session" / "chat.json"
    history_path    = test_config.tmp_dir / ".agent_app" / "test_data" / "chats" / "default_session" / "chat_history.json"
    chroma_path     = test_config.tmp_dir / ".agent_app" / "test_data" / "chroma"

    assert agent.model                  == "mock_model"
    assert agent.chat.active_conv_path  == chat_path
    assert agent.chat.chat_history_path == history_path
    assert agent.memory.path            == chroma_path

def test_agent_init_custom_model(test_config):
    agent = Agent(model="custom_model")

    chat_path       = test_config.tmp_dir / ".agent_app" / "test_data" / "chats" / "default_session" / "chat.json"
    history_path    = test_config.tmp_dir / ".agent_app" / "test_data" / "chats" / "default_session" / "chat_history.json"
    chroma_path     = test_config.tmp_dir / ".agent_app" / "test_data" / "chroma"

    assert agent.model                  == "custom_model"
    assert agent.chat.active_conv_path  == chat_path
    assert agent.chat.chat_history_path == history_path
    assert agent.memory.path            == chroma_path

def test_agent_init_custom_session(test_config):
    agent = Agent(session="custom")

    chat_path       = test_config.tmp_dir / ".agent_app" / "test_data" / "chats" / "custom" / "chat.json"
    history_path    = test_config.tmp_dir / ".agent_app" / "test_data" / "chats" / "custom" / "chat_history.json"
    chroma_path     = test_config.tmp_dir / ".agent_app" / "test_data" / "chroma"

    assert agent.chat.active_conv_path  == chat_path
    assert agent.chat.chat_history_path == history_path
    assert agent.memory.path            == chroma_path


# ========================================================================
# _cmd_forget
# ========================================================================

def test_cmd_forget(monkeypatch):
    agent           = Agent()
    agent.memory    = MagicMock()

    agent.memory.get_exact_match.return_value   = ("mem_123", "This is a memory entry.")
    monkeypatch.setattr("builtins.input", lambda _: "") # User select [Enter]

    result = agent._cmd_forget("This is a message.")

    agent.memory.get_exact_match.assert_called_once_with("This is a message.")
    agent.memory.delete_from_db.assert_called_once_with(["mem_123"])
    assert result == "Entry deleted."

def test_cmd_forget_cancel(monkeypatch):
    agent           = Agent()
    agent.memory    = MagicMock()

    agent.memory.get_exact_match.return_value = ("mem_123", "This is a memory entry.")
    monkeypatch.setattr("builtins.input", lambda _: "c") # User select [c]

    result = agent._cmd_forget("This is a message.")

    agent.memory.get_exact_match.assert_called_once_with("This is a message.")
    agent.memory.delete_from_db.assert_not_called()
    assert result == "Deletion cancelled."

def test_cmd_forget_no_prompt_error(monkeypatch):
    agent           = Agent()
    agent.memory    = MagicMock()

    result = agent._cmd_forget("")

    assert result == "Please specify what to forget."

def test_cmd_forget_no_match_error(monkeypatch):
    agent           = Agent()
    agent.memory    = MagicMock()

    agent.memory.get_exact_match.return_value = ()

    result = agent._cmd_forget("This is a message.")

    assert result == "Error: No matching memory found."


# ========================================================================
# _cmd_memorise
# ========================================================================

def test_cmd_memorise(monkeypatch, capsys):
    agent           = Agent()
    agent.memory    = MagicMock()
    agent.chat      = MagicMock()

    agent.memory.extract_to_db.return_value         = ("mem_123", 40, 60)
    agent.memory.get_entries_by_ids.return_value    = (
        {
            "category": "fact",
            "content": "Memory content"
        },
    )
    monkeypatch.setattr("builtins.input", lambda _: "") # User select [Enter]

    prompt = "This is a message."
    result = agent._cmd_memorise(prompt)

    captured = capsys.readouterr()

    agent.memory.extract_to_db.assert_called_once_with(
        context=agent.chat.to_llm(),
        prompt=prompt,
        source="explicit",
        manual=True
    )
    agent.chat.append_user_message_with_metadata.assert_called_once_with(
        content=prompt,
        state="external"
    )
    agent.chat.append_assistant_message_with_metadata.assert_called_once_with(
        content=result, 
        state="internal",
        prompt_tokens=40,
        output_tokens=60
    )
    assert "Extracting content from user's input..." in captured.out
    assert "Content saved to database:" in captured.out
    assert "- [fact] Memory content" in captured.out
    assert result == "Memory saved to database."

def test_cmd_memorise_undo(monkeypatch, capsys):
    agent           = Agent()
    agent.memory    = MagicMock()
    agent.chat      = MagicMock()

    agent.memory.extract_to_db.return_value         = ("mem_123", 40, 60)
    agent.memory.get_entries_by_ids.return_value    = (
        {
            "category": "fact",
            "content": "Memory content"
        },
    )
    monkeypatch.setattr("builtins.input", lambda _: "u") # User select [u]

    prompt = "This is a message."
    result = agent._cmd_memorise(prompt)

    captured = capsys.readouterr()

    agent.memory.extract_to_db.assert_called_once_with(
        context=agent.chat.to_llm(),
        prompt=prompt,
        source="explicit",
        manual=True
    )
    agent.chat.append_user_message_with_metadata.assert_not_called()
    agent.chat.append_assistant_message_with_metadata.assert_not_called()
    assert "Extracting content from user's input..." in captured.out
    assert "Content saved to database:" in captured.out
    assert "- [fact] Memory content" in captured.out
    assert result == "Entry deleted."

def test_cmd_memorise_no_prompt_error(monkeypatch, capsys):
    agent = Agent()

    prompt = ""
    result = agent._cmd_memorise(prompt)

    assert result == "Please specify what to memorize."

def test_cmd_memorise_no_data_error(monkeypatch, capsys):
    agent           = Agent()
    agent.memory    = MagicMock()

    agent.memory.extract_to_db.return_value         = ("mem_123", 40, 60)
    agent.memory.get_entries_by_ids.return_value    = ()

    prompt = "This is a message."
    result = agent._cmd_memorise(prompt)

    captured = capsys.readouterr()

    assert "Extracting content from user's input..." in captured.out
    assert result == "Error: No data was extracted by the model."


# ========================================================================
# _cmd_recall
# ========================================================================

def test_cmd_recall(monkeypatch):
    agent           = Agent()
    agent.memory    = MagicMock()
    agent.chat      = MagicMock()

    agent.memory.retrieve_relevant_entry.return_value = (
        {
            "category": "fact",
            "content": "Memory content"
        },
    )

    prompt = "This is a message."
    result = agent._cmd_recall(prompt)

    agent.chat.append_user_message_with_metadata.assert_called_once_with(
        content=prompt,
        state="external"
    )
    agent.chat.append_assistant_message_with_metadata.assert_called_once_with(
        content=result, 
        state="external",
    )
    assert result == (
        "Found matching entries in long-term memory:\n"
        "- [fact] Memory content"
    )

def test_cmd_recall_no_prompt_error(monkeypatch):
    agent = Agent()

    prompt = ""
    result = agent._cmd_recall(prompt)

    assert result == "Please specify what to recall."

def test_cmd_recall_no_memory_found_error(monkeypatch):
    agent           = Agent()
    agent.memory    = MagicMock()
    agent.chat      = MagicMock()

    agent.memory.retrieve_relevant_entry.return_value = ()

    prompt = "This is a message."
    result = agent._cmd_recall(prompt)

    agent.chat.append_user_message_with_metadata.assert_not_called()
    agent.chat.append_assistant_message_with_metadata.assert_not_called()
    assert result == "Error: No matching memories found."


# ========================================================================
# _cmd_search
# ========================================================================

query_with_urls = (
    {"Search query": "url"}
)

def test_cmd_search(monkeypatch, capsys):
    agent               = Agent()
    agent.search_agent  = MagicMock()
    agent.chat          = MagicMock()

    agent.search_agent.generates_query.return_value = ("Search query")
    agent.search_agent.web.return_value             = (
        "Agent answers from web search.",
        query_with_urls,
        40,
        60,
        True
    )

    prompt = "This is a message."
    result = agent._cmd_search(prompt)
    agent.chat.append_user_message_with_metadata.assert_called_once_with(
        content=prompt,
        state="external"
    )
    agent.chat.append_assistant_message_with_metadata.assert_called_once_with(
        content="Agent answers from web search.",
        state="external",
        query_with_urls=query_with_urls,
        search=True,
        prompt_tokens=40,
        output_tokens=60
    )
    assert result == None

def test_cmd_search_no_prompt_error():
    agent       = Agent()
    agent.chat  = MagicMock()

    prompt = ""
    result = agent._cmd_search(prompt)

    agent.chat.append_user_message_with_metadata.assert_not_called()
    agent.chat.append_assistant_message_with_metadata.assert_not_called()
    assert result == "Please specify what to search."


# ========================================================================
# _file_context
# ========================================================================

context = [
    {"role": "system", "content": "You are a helpful assistant." ,"state": "internal"},
    {"role": "user", "content": "Message 1", "state": "external"},
    {"role": "assistant", "content": "Message 2", "state": "external", "prompt_tokens": 10, "output_tokens": 30}
]

found_files = ["found_file1", "found_file2", "found_file3"]
not_found_files = ["not_found1", "not_found2", "not_found3"]

def test_file_context_required():
    agent               = Agent()
    agent.file_reader   = MagicMock()

    agent.file_reader.require_file_or_not.return_value = True
    agent.file_reader.get_filenames.return_value = (found_files, not_found_files)

    prompt = "This is a message."
    result = agent._file_context(context, prompt)

    agent.file_reader.require_file_or_not.assert_called_once_with(context, prompt)
    agent.file_reader.get_filenames.assert_called_once_with(context, prompt)
    assert result == (found_files, not_found_files)

def test_file_context_not_required():
    agent               = Agent()
    agent.file_reader   = MagicMock()

    agent.file_reader.require_file_or_not.return_value = False

    prompt = "This is a message."
    result = agent._file_context(context, prompt)

    agent.file_reader.require_file_or_not.assert_called_once_with(context, prompt)
    agent.file_reader.get_filenames.assert_not_called()
    assert result == None


# ========================================================================
# ask
# ========================================================================
#agent._cmd_forget.return_value = "Entry deleted."
#agent._cmd_memorise.return_value = "Memory saved to database."
#agent._cmd_recall.return_value = "- [fact] Memory content"
#agent._cmd_search.return_value = None

to_llm_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Message 2"}
]

def test_ask(monkeypatch):
    agent               = Agent()
    agent.memory        = MagicMock()
    agent.file_reader   = MagicMock()
    agent.search_agent  = MagicMock()
    agent.chat          = MagicMock()

    prompt = "This is a message."

    agent._manage_token_budget = MagicMock()

    mock_detect_cmd = MagicMock(return_value=(None, prompt))
    monkeypatch.setattr("src.agent.core._detect_cmd", mock_detect_cmd)

    mock_model_response = MagicMock(return_value=("Answer", 10, 30))
    monkeypatch.setattr("src.agent.core.LLM.model_response", mock_model_response)
    
    agent.chat.to_llm.return_value                          = [to_llm_messages]
    agent.memory.toggle_auto_add_memory_entry.return_value  = "- [fact] Memory content"
    agent.file_reader.toggle_auto_read_dropbox.return_value = (
        "File contents",
        ["path1", "path2", "path3"],
        True
    )
    query_with_urls = (
        {"query": "url"}
    )
    agent.search_agent.toggle_auto_web_search.return_value = (
        "Search results",
        query_with_urls,
        True
    )

    agent.ask(prompt)

    agent._manage_token_budget.assert_called_once_with(prompt)
    mock_detect_cmd.assert_called_once_with(prompt)
    agent.chat.to_llm.assert_called_once()
    agent.memory.toggle_auto_add_memory_entry.assert_called_once()
    agent.file_reader.toggle_auto_read_dropbox.assert_called_once()
    agent.search_agent.toggle_auto_web_search.assert_called_once()
    mock_model_response.assert_called_once()
    agent.chat.append_user_message_with_metadata.assert_called_once_with(
        content=prompt,
        state="external"
    )
    agent.chat.append_assistant_message_with_metadata.assert_called_once_with(
        content="Answer",
        state="external",
        attachments=["path1", "path2", "path3"],
        query_with_urls=query_with_urls,
        prompt_tokens=10,
        output_tokens=30
    )
