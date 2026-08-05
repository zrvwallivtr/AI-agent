import pytest
import sys
from unittest.mock import MagicMock, patch
from argparse import Namespace

import src.cli.flags as cli_module
from src.cli.flags import main


@pytest.fixture(autouse=True)
def mock_agent_class():
    """Globally blocks the real Agent class from initializing token handlers."""
    with patch("src.cli.flags.Agent") as mock_agent:
        yield mock_agent


# ===========================================================================================
# General flags
#
# agent --reset-default
# agent --installed-models
# ===========================================================================================

@patch("sys.argv", ["agent", "-rd"])
@patch("src.cli.flags.General.reset_default")
def test_main_model_routing(mock_reset):
    # Verifies --reset-default triggers the General reset method
    main()
    mock_reset.assert_called_once()

@patch("sys.argv", ["agent", "-rd"])
@patch("src.cli.flags.General.reset_default")
def test_main_reset_default_routing(mock_reset):
    # Verifies --reset-default triggers the General reset method
    main()
    mock_reset.assert_called_once()

@patch("sys.argv", ["agent", "-i"])
@patch("src.cli.flags.General.installed_models")
def test_main_installed_models_routing(mock_installed):
    # Verifies --installed-models calls the system listing mechanism
    main()
    mock_installed.assert_called_once()


# ===========================================================================================
# Session flags
#
# agent --list-session
# agent --new-session new_session "This message is irrelevant in this test."
# agent --delete-session old_session
# ===========================================================================================

@patch("sys.argv", ["agent", "-ls"])
@patch("src.cli.flags.Session.list_session")
def test_main_list_sessions_routing(mock_list_session):
    # Verifies --list-session maps to the static session viewer
    main()
    mock_list_session.assert_called_once()

@patch("sys.argv", ["agent", "-ns", "new_session", "This message is irrelevant in this test."])
@patch("src.cli.flags.Session")
def test_main_new_session_routing(mock_session_class):
    mock_session_instance = mock_session_class.return_value
    
    main()
    
    # Verifies the default context setup matches instantiation requirements
    mock_session_class.assert_called_once_with(None)
    mock_session_instance.create_session.assert_called_once_with(
        model=cli_module.MODEL, 
        prompt="This message is irrelevant in this test."
    )

@patch("sys.argv", ["agent", "-d", "old_session"])
@patch("src.cli.flags.Session")
def test_main_delete_session_routing(mock_session_class):
    # Verifies session termination routes to the matching cleanup hook
    mock_session_instance = mock_session_class.return_value
    
    main()
    
    mock_session_instance.delete_session.assert_called_once()


# ===========================================================================================
# Read flag
#
# agent --r data.csv
# ===========================================================================================

@patch("sys.argv", ["agent", "-r", "data.csv"])
@patch("builtins.print")
def test_main_read_missing_question_error(mock_print):
    # Validates guardrails catch missing text parameters during file operations
    main()
    mock_print.assert_called_once_with("Error: question required")


# ===========================================================================================
# Project manager
#
# agent --project-summary test_project test_session
# agent --project-task test_project test_session
# agent --project-project-dec test_project test_session
# ===========================================================================================

@patch("argparse.ArgumentParser.parse_args")
@patch("src.cli.flags.ProjectManager")
def test_main_new_project_workspace(mock_pm_class, mock_parse_args):
    # Ensures project initialization executes workspace scaffolders
    mock_parse_args.return_value = Namespace(
        model="mock", session=None, question=None, reset_default=False,
        installed_models=False, new_session=None, delete_session=None,
        list_session=False, read=None, new_project="workspace_alpha",
        create_project="workspace_alpha", project_summary=None,
        project_task=None, project_dec=None
    )
    mock_pm_instance = mock_pm_class.return_value
    
    main()
    
    mock_pm_class.assert_called_once_with("workspace_alpha")
    mock_pm_instance.create_project_workspace.assert_called_once()

@patch("sys.argv", ["agent", "-ps", "test_project", "test_session", "Summarize this"])
@patch("src.cli.flags.ProjectManager")
def test_main_project_summary_edit(mock_pm_class):
    # Validates parameter slicing and delivery to the project summarization engine
    mock_pm_instance = mock_pm_class.return_value
    
    main()
    
    mock_pm_class.assert_called_once_with("test_project")
    mock_pm_instance.summarize_project.assert_called_once_with("Summarize this", session="test_session")

@patch("sys.argv", ["agent", "-pt", "test_project", "test_session"])
@patch("src.cli.flags.ProjectManager")
def test_main_project_tasklist_generation(mock_pm_class):
    # Ensures checklist generation requests pass down verified workspace keys
    mock_pm_instance = mock_pm_class.return_value
    
    main()
    
    mock_pm_class.assert_called_once_with("test_project")
    mock_pm_instance.generate_tasklist.assert_called_once_with(session="test_session")

@patch("sys.argv", ["agent", "-pd", "test_project", "test_session", "New decision log entry"])
@patch("src.cli.flags.ProjectManager")
def test_main_project_decision_logging(mock_pm_class):
    # Verifies project log managers handle multi-argument decision tracking updates
    mock_pm_instance = mock_pm_class.return_value
    
    main()
    
    mock_pm_class.assert_called_once_with("test_project")
    mock_pm_instance.add_decisions.assert_called_once_with("New decision log entry", session="test_session")


# ===========================================================================================
# Question routing and fallback
#
# agent "This message is irrelevant in this test."
# agent -help
# agnet
# ===========================================================================================

@patch("sys.argv", ["agent", "This message is irrelevant in this test."])
@patch("src.cli.flags.General.question")
def test_main_plain_question_routing(mock_question_fn):
    # Verifies that unflagged text inputs route to the default assistant pipeline
    main()
    mock_question_fn.assert_called_once_with(
        prompt="This message is irrelevant in this test.",
        model=cli_module.MODEL,
        session=None
    )

@patch("sys.argv", ["agent", "-h"])
@patch("argparse.ArgumentParser.print_help")
def test_main_help_flag(mock_print_help):
    # Ensures the interface details are printed with the --help flag
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    mock_print_help.assert_called_once()

@patch("sys.argv", ["agent"])
@patch("argparse.ArgumentParser.print_help")
def test_main_empty_input_fallback_help(mock_print_help):
    # Ensures the interface details are printed if no operation tasks are provided
    main()
    mock_print_help.assert_called_once()


# ===========================================================================================
# Combinations
#
# agent --session test_session --model test_model "This message is irrelevant in this test"
# ===========================================================================================

@patch("sys.argv", ["agent", "-s", "test_session", "-m", "test_model", "This message is irrelevant in this test."])
@patch("src.cli.flags.General.question")
def test_session_and_model_flags_with_question(mock_question_fn):
    # Verifies that unflagged text inputs route to the default assistant pipeline
    main()
    mock_question_fn.assert_called_once_with(
        prompt="This message is irrelevant in this test.",
        model="test_model",
        session="test_session"
    )
