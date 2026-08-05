import pytest

from pathlib import Path
from unittest.mock import MagicMock

from rich import text

from agent import memory
import src.agent.memory as memory_module
import src.agent.tokens_handler as tokens_module

from src.agent.memory import Memory, CATEGORIES, CATEGORY_TYPES
from src.agent.tokens_handler import Tokens
from src.agent.llm import LLM
from src.config import MODEL, CHAT_DIR, DEFAULT_PATH, DEFAULT_CHAT_HISTORY_PATH


@pytest.fixture(autouse=True)
def setup_memory_env(tmp_path, monkeypatch):
    mock_db_dir = tmp_path / "data" / "chroma"
    mock_db_dir.mkdir(parents=True, exist_ok=True)

    # Override configs
    monkeypatch.setattr(memory_module, "MODEL", "mock_model")
    monkeypatch.setattr(memory_module, "EMBED_MODEL", "mock_embed_model")
    monkeypatch.setattr(memory_module, "MEM_PROMPT", "Memory retrieve prompt")
    monkeypatch.setattr(memory_module, "MEM_MANUAL_PROMPT", "Manual memory retrieve prompt")

    # Override tokens handler
    monkeypatch.setitem(tokens_module.MODEL_MAX, "mock_model", 4000)
    monkeypatch.setitem(tokens_module.MODEL_MAX, "mock_embed_model", 4000)
    monkeypatch.setattr(Tokens, "check_fit", MagicMock(return_value=True))

    # Override paths
    monkeypatch.setattr(memory_module, "CHROMADB_DIR", mock_db_dir)

    # Mock embedding
    mock_vector = {"embedding": [0.1, 0.5, -0.2]}
    monkeypatch.setattr("ollama.embeddings", MagicMock(return_value=mock_vector))

    return mock_db_dir

@pytest.fixture
def mock_chat():
    return MagicMock()


# ======================================
# Initialise
# ======================================

def test_memory_init_without_project(mock_chat):
    global_mem = Memory(chat=mock_chat, project=None)
    assert global_mem.vector_db.name == "global_memory"

def test_memory_init_with_project(mock_chat):
    project_mem = Memory(chat=mock_chat, project="name")
    assert project_mem.vector_db.name == "project_name"


# ======================================
# Append to database
# ======================================

def test_append_to_db(mock_chat):
    memory = Memory(chat=mock_chat)

    entry_id = memory._append_to_db(
        content="Memory",
        category="stack",
        source="explicit"
    )

    assert entry_id.startswith("mem_")
    assert memory.vector_db.count() == 1

def test_append_to_db_aborts_on_empty_string(mock_chat, monkeypatch):
    memory = Memory(chat=mock_chat)

    blank_entry_id = memory._append_to_db(
        content="    ",
        category="stack",
        source="explicit"
    )
    assert blank_entry_id is None

def test_append_to_db_aborts_on_oversized_tokens(mock_chat, monkeypatch):
    memory = Memory(chat=mock_chat)

    # Mock token count exceed limit
    monkeypatch.setattr(Tokens, "check_fit", MagicMock(return_value=False))

    oversized_entry_id = memory._append_to_db(
        content="Oversized memory",
        category="stack",
        source="explicit"
    )
    assert oversized_entry_id is None

def test_append_to_db_with_duplicated_entry(mock_chat):
    memory  = Memory(chat=mock_chat)
    text    = "This is the duplicated message."

    old_id = memory._append_to_db(content=text, category="instruction", source="explicit")
    assert memory.vector_db.count() == 1

    # Test duplicated message but with its content in upper case
    new_id = memory._append_to_db(content=text.upper(), category="instruction", source="explicit")

    # Ensure duplicated memory will be overwritten
    assert old_id == new_id
    assert memory.vector_db.count() == 1

# ======================================
# Memory structuring
# ======================================

def test_appended_memory_content_structure(mock_chat):
    memory = Memory(chat=mock_chat)

    markdown = (
        "- [preference] This should also be stored into the database.\n"
        "* • [STACK] This should be stored into the database.\n"
        "[UNKNOWN_TAG] This should drop into the fallback default category.\n"
        "- [fact] \n" # Emty body segment: should be ignored
    )

    created_ids = memory._format_and_append_to_db(markdown, source="extracted")

    # Expecting three extractions
    assert len(created_ids) == 3

    entries = memory.get_entries_by_ids(created_ids)
    assert entries[0]["category"] == "preference"
    assert entries[1]["category"] == "stack"
    assert entries[2]["category"] == "fact"

# ======================================
#  LLM integration
# ======================================

def test_correct_system_prompt_routing(mock_chat, monkeypatch):
    memory = Memory(chat=mock_chat)

    # Complex structured response
    mock_response                   = MagicMock()
    mock_response.message.content   = "[fact] Memory."
    mock_llm_call                   = MagicMock(return_value=mock_response)
    monkeypatch.setattr(LLM, "response_with_new_sys_prompt_and_context", mock_llm_call)

    # Ensure the auto extract memory prompt is correctly implemented
    memory.extract_to_db(context=[], prompt="User query", source="extracted", manual=False)
    assert mock_llm_call.call_args[1]["system_prompt"] == memory.mem_prompt

    # Ensure the manual extract memory prompt is correctly implemented
    memory.extract_to_db(context=[], prompt="Save this fact", source="extracted", manual=True)
    assert mock_llm_call.call_args[1]["system_prompt"] == memory.mem_manual_prompt

# ======================================
# Data retrieval
# ======================================

def test_add_memory_to_messages(mock_chat):
    memory = Memory(chat=mock_chat)

    old_messages    = [
        {"role": "system", "content": "System prompt", "state": "internal"},
        {"role": "user", "content": "Question?", "state": "external"},
        {"role": "assistant", "content": "Answer.", "state": "external"}
    ]
    messages        = old_messages

    # Store entry to db for retrieval
    memory._append_to_db("Content stored in database.", "instruction", "explicit")

    # Retrieve from database
    memory_entries = memory.add_memory_entries(prompt="This message should not be returned.", messages=messages)

    assert old_messages == messages
    assert "Relevant context from memory:" in memory_entries
    assert "This message should not be returned." not in memory_entries

def test_add_empty_memory_to_messages(mock_chat):
    memory = Memory(chat=mock_chat)

    old_messages    = [
        {"role": "system", "content": "System prompt", "state": "internal"},
        {"role": "user", "content": "Question?", "state": "external"},
        {"role": "assistant", "content": "Answer.", "state": "external"}
    ]
    messages        = old_messages

    # Retrieve from empty database
    empty_memory_entries = memory.add_memory_entries(prompt="This message should not be returned.", messages=messages)

    assert old_messages == messages
    assert empty_memory_entries == ""

def test_get_exact_match(mock_chat):
    memory = Memory(chat=mock_chat)

    # Ensure memory is empty
    assert memory.get_exact_match("This message does not exist in memory.") is None

    # Add memory to database
    string = "Content"
    entry_id = memory._append_to_db(string, "fact", "explicit")

    # Verify exact memory match
    match_result = memory.get_exact_match(string)
    assert match_result is not None
    assert match_result[0] == entry_id
    assert match_result[1] == string

    # Ensure deletion
    memory.delete_from_db(ids=[entry_id])
    assert memory.vector_db.count() == 0
