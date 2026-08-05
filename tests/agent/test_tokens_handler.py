import pytest
from src.agent.tokens_handler import Tokens
from src.config import MODEL


def test_from_config():
    """Test case: Import configurations from 'config.toml'."""
    tokens          = Tokens(model=MODEL)
    messages        = [{"role": "user", "content": "Hello world."}]
    tokens_count    = tokens.count_history_tokens(messages)

    assert isinstance(tokens_count, int)
    assert tokens_count > 0

def test_manual_config():
    """Model name not on pre written list."""
    tokens          = Tokens(model="mock_model", max_tokens=4096)
    messages        = [{"role": "user", "content": "Hello world."}]
    tokens_count    = tokens.count_history_tokens(messages)

    assert isinstance(tokens_count, int)
    assert tokens_count > 0

def test_error_handle():
    """Check error handling."""
    with pytest.raises(ValueError) as exc_info:
        tokens          = Tokens(model="mock_model")
