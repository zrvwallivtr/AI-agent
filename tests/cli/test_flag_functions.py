import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.cli.flag_functions as flag_module
from src.cli.flag_functions import General, Session, _delete_session_files


# ==============================================
# Delete session files
# ==============================================

@patch("src.cli.flag_functions.FileReader")
@patch("src.cli.flag_functions.Chat")
def test_delete_session_files_execution(mock_chat_class, mock_file_reader_class):
    mock_chat           = mock_chat_class.return_value
    mock_file_reader    = mock_file_reader_class.return_value
    
    # Execute the cleanup routine
    _delete_session_files(session="test_session")
    
    # Check that managers were initialized with correct session tracking strings
    mock_chat_class.assert_called_once_with("test_session")
    mock_file_reader_class.assert_called_once_with("test_session")
    
    # Confirm downstream wipe methods were triggered
    mock_chat.clear_active_conv.assert_called_once()
    mock_chat.clear_chat_history.assert_called_once()
    mock_file_reader.clear_session_dropbox.assert_called_once_with("test_session")


# ==============================================
# General class
# ==============================================

@patch("src.cli.flag_functions.Agent")
def test_general_question(mock_agent_class):
    mock_agent_instance = mock_agent_class.return_value
    
    General.question(model="mock_model", prompt="Hello Agent", session="sess_123", project="proj_abc")
    
    mock_agent_class.assert_called_once_with(model="mock_model", session="sess_123", project="proj_abc")
    mock_agent_instance.ask.assert_called_once_with(prompt="Hello Agent")

@patch("src.cli.flag_functions._delete_session_files")
def test_general_reset_default(mock_delete_fn):
    General.reset_default()
    mock_delete_fn.assert_called_once_with()

@patch("src.cli.flag_functions.ollama.list")
@patch("builtins.print")
def test_general_installed_models_printing(mock_print, mock_ollama_list):
    mock_ollama_list.return_value = {
        "models": [
            {"model": "llama3:latest"},
            {"model": "mistral:7b"}
        ]
    }
    
    General.installed_models()
    
    # Ensure printed
    mock_print.assert_any_call("Installed models:")
    mock_print.assert_any_call("- llama3:latest")
    mock_print.assert_any_call("- mistral:7b")


# ==============================================
# Session class
# ==============================================

@patch("src.cli.flag_functions.Chat")
@patch("builtins.print")
def test_create_session_already_exists_guardrail(mock_print, mock_chat_class):
    mock_chat = mock_chat_class.return_value
    mock_chat.active_conv_path.exists.return_value = True  # Emulate conflict
    
    session_manager = Session(session="duplicate_session")
    session_manager.create_session(model="mock_model", prompt=None)
    
    mock_print.assert_called_once_with("Error: Session duplicate_session already exist.")
    mock_chat.save.assert_not_called()

@patch("src.cli.flag_functions.Chat")
@patch("builtins.print")
def test_create_session_empty_prompt(mock_print, mock_chat_class):
    """Validates that a fresh session shell is saved when no immediate prompt is attached."""
    mock_chat = mock_chat_class.return_value
    mock_chat.active_conv_path.exists.return_value = False
    mock_chat.chat_history_path.exists.return_value = False
    
    session_manager = Session(session="fresh_session")
    session_manager.create_session(model="mock_model", prompt=None)
    
    mock_chat.save.assert_called_once()
    mock_print.assert_called_once_with("Created new session: fresh_session")

@patch("src.cli.flag_functions.Chat")
@patch("src.cli.flag_functions.General.question")
def test_create_session_with_inline_prompt(mock_question_fn, mock_chat_class):
    mock_chat = mock_chat_class.return_value
    mock_chat.active_conv_path.exists.return_value = False
    mock_chat.chat_history_path.exists.return_value = False
    
    session_manager = Session(session="interactive_session")
    session_manager.create_session(model="mock_model", prompt="This message is irrelevant in this test.")
    
    mock_question_fn.assert_called_once_with(
        prompt="This message is irrelevant in this test.", 
        model="mock_model", 
        session="interactive_session"
    )

@patch("src.cli.flag_functions.Chat")
@patch("src.cli.flag_functions._delete_session_files")
def test_session_deletion_routing(mock_delete_fn, mock_chat_class):
    session_manager = Session(session="kill_me")
    session_manager.delete_session()
    
    mock_delete_fn.assert_called_once_with("kill_me")

@patch("src.cli.flag_functions.CHAT_DIR")
@patch("src.cli.flag_functions.DEFAULT_PATH")
@patch("src.cli.flag_functions.DEFAULT_CHAT_HISTORY_PATH")
@patch("builtins.print")
def test_list_session_filtering_logic(mock_print, mock_def_hist, mock_def_path, mock_chat_dir):
    # Set naming signatures for the standard global ignores
    mock_def_path.name = "default_chat.json"
    mock_def_hist.name = "default_chat_history.json"
    
    # Mock paths
    session_ok_1        = MagicMock(spec=Path)
    session_ok_1.name   = "alpha.json"
    session_ok_2        = MagicMock(spec=Path)
    session_ok_2.name   = "beta.json"
    
    session_history         = MagicMock(spec=Path)
    session_history.name    = "alpha_chat_history.json" # Hidden by filter rule
    
    session_system_default      = MagicMock(spec=Path)
    session_system_default.name = "default_chat.json" # Hidden by exclusion matching
    
    # All file list
    mock_chat_dir.glob.return_value = [
        session_ok_1, 
        session_ok_2, 
        session_history, 
        session_system_default
    ]
    
    Session.list_session()
    
    # Compare file list to actual printed list in 'list_session'
    printed_values = [call.args[0] for call in mock_print.call_args_list]
    
    # Ensure all relevant values are printed
    assert "alpha" in printed_values
    assert "beta" in printed_values

    # Ensure all irrelevant values are not printed
    assert "alpha_chat_history" not in printed_values
    assert "default_chat" not in printed_values
