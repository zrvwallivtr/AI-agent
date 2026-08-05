import pytest
from unittest.mock import MagicMock, mock_open, patch
from pathlib import Path

import src.tools.project_manager as pm_module
from src.tools.project_manager import ProjectManager
from src.agent.llm import LLM
from src.agent.chat import Chat


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(pm_module, "PROJECTS_DIR", tmp_path)

    # Mock config
    monkeypatch.setattr(pm_module, "PM_MODEL", "mock_pm_model")
    monkeypatch.setattr(pm_module, "PM_PROMPT", "Mock System Instruction")

    return tmp_path


# ===================================
# Workspace
# ===================================

def test_create_project_workspace(isolated_workspace):
    pm = ProjectManager(project_name="mock_project")

    pm.create_project_workspace()

    assert pm.path.exists()
    assert (pm.path / "Summary.md").read_text() == "# Summary\n"
    assert (pm.path / "Tasks.md").read_text() == "# Tasks\n"
    assert (pm.path / "Decisions.md").read_text() == "# Decisions\n"

def test_create_already_exists_project_workspace(isolated_workspace, capsys):
    pm = ProjectManager(project_name="mock_project")

    # Creates project folder, simulating that the project has already been created
    pm.path.mkdir(parents=True, exist_ok=True)

    # Mock create existing project
    pm.create_project_workspace()

    # Captures print() into the terminal
    captured = capsys.readouterr()
    assert "already exist" in captured.out

def test_generate_tasklist(isolated_workspace, monkeypatch):
    pm = ProjectManager(project_name="mock_project")
    pm.path.mkdir(parents=True, exist_ok=True)

    # Mock existing project
    (pm.path / "Summary.md").write_text("# Summary\nInitial Project Summary Info")
    (pm.path / "Tasks.md").write_text("# Task\n- [ ] Old Task")

    # Mock model response in 'generate_tasklist' function
    mock_llm = MagicMock(return_value="- [ ] New High Priority Task\n- [ ] Old Task")
    monkeypatch.setattr(LLM, "model_response", mock_llm)

    response = pm.generate_tasklist(session="session_123")

    assert response == "- [ ] New High Priority Task\n- [ ] Old Task"
    assert (pm.path / "Tasks.md").read_text() == "- [ ] New High Priority Task\n- [ ] Old Task"
