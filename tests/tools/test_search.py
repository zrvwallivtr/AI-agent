import pytest
import datetime
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import requests

import src.tools.search as search_module
from src.tools.search import (
    SearchAgent,
    _get_results_from_query,
    _scrape_url_content,
    _store_results_in_tmp_file,
    _read_tmp_file
)
from src.agent.llm import LLM


@pytest.fixture(autouse=True)
def mock_search_config(monkeypatch):
    monkeypatch.setattr(search_module, "MODEL", "mock_search_llm")
    monkeypatch.setattr(search_module, "SEARCH_ENG", "https://mock.search.engine/search")
    monkeypatch.setattr(search_module, "MAX_RESULTS", 2)
    monkeypatch.setattr(search_module, "MAX_CHAR_PER_PAGE", 100)
    monkeypatch.setattr(search_module, "SEARCH_OR_NOT_PROMPT", "System decision prompt context")
    monkeypatch.setattr(search_module, "QUERY_PROMPT", "Current Date: {{current_date}}. Generate query for: ")


# ========================================
# Low-level helper function tests
# ========================================

@patch("requests.get")
def test_get_results_from_query(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "Res 1", "url": "http://1.com", "content": "Snippet 1"},
            {"title": "Res 2", "url": "http://2.com", "content": "Snippet 2"},
            {"title": "Res 3", "url": "http://3.com", "content": "Snippet 3"},
        ]
    }
    mock_get.return_value = mock_response

    results = _get_results_from_query("testing query", "https://mock.engine", 2)
    
    # Ensure only two results are returned
    assert len(results) == 2
    assert results[0]["title"] == "Res 1"
    assert results[1]["title"] == "Res 2"
    mock_get.assert_called_once()

@patch("requests.get")
def test_get_results_from_query_failure(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out.")
    
    results = _get_results_from_query("broken query", "https://mock.engine", 2)
    assert results == []

@patch("requests.get")
def test_scrape_url_content(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <header>Skip this header section</header>
            <nav>Skip menu links</nav>
            <main>
                <h1>Heading Content</h1>
                <p>Hello   World!   This   has   too   many   spaces.</p>
                <script>console.log('strip me');</script>
            </main>
            <footer>Skip copyright</footer>
        </body>
    </html>
    """
    mock_get.return_value = mock_response

    trimmed_text = _scrape_url_content("https://sample.target.site")
    
    # Ensure noise tags are decomposed and structural spaces are flattened
    assert "Skip this header section" not in trimmed_text
    assert "console.log" not in trimmed_text
    assert "Heading Content" in trimmed_text
    assert "Hello World! This has too many spaces." in trimmed_text

@patch("requests.get")
def test_scrape_url_content_http_error(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_get.return_value = mock_response

    # Ensure nothing is returned
    assert _scrape_url_content("https://forbidden.site") == ""

def test_store_and_read_tmp_file():
    mock_results = [
        {"title": "Page Title A", "url": "http://a.com", "content": "Fallback snippet body text"}
    ]
    
    with patch("src.tools.search._scrape_url_content", return_value="Scraped Body Context Material"):
        tmp_filename = _store_results_in_tmp_file(mock_results)
        
        try:
            # Ensure results stored in temporary file
            assert os.path.exists(tmp_filename)
            content = _read_tmp_file(tmp_filename)
            
            # Ensure proper format is stored in the temporary file
            assert "### Source 1: Page Title A" in content
            assert "URL: http://a.com" in content
            assert "Full Page Context:\nScraped Body Context Material" in content

        finally:
            # Remove test file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)


# ========================================
# SearchAgent object method tests
# ========================================

@pytest.mark.parametrize(
    "llm_output, expected_boolean",
    [
        ("True", True),
        ("The decision path is definitely true.", True),
        ("FALSE", False),
        ("unrelated narrative content block response", False),
    ]
)
def test_search_or_not_evaluations(monkeypatch, llm_output, expected_boolean):
    agent = SearchAgent()
    
    mock_response                   = MagicMock()
    mock_response.message.content   = llm_output
    monkeypatch.setattr(LLM, "response_with_new_sys_prompt_and_context", MagicMock(return_value=mock_response))

    assert agent.search_or_not([], "This message is irrelavant in this test.") == expected_boolean

@pytest.mark.parametrize(
    "input_query, expected_clean",
    [
        ('inurl:news "artificial intelligence"', "news artificial intelligence"),
        ('site:github.com sorted:newest "ai_agent"', "github.com ai_agent"),
        ('`*stray punctuation check*`', "stray punctuation check"),
        ('First Line Only\nSecond Line', "First Line Only")
    ]
)
def test_search_query_check_format(input_query, expected_clean):
    agent = SearchAgent()

    assert agent.search_query_check(input_query) == expected_clean


def test_generates_query_date(monkeypatch):
    agent = SearchAgent()

    # Freeze system time (Saturday, 13th June 2026)
    fixed_time = datetime.datetime(2026, 6, 13, 12, 0, 0)
    class MockDateTime:
        @classmethod
        def now(cls):
            return fixed_time
    monkeypatch.setattr(datetime, "datetime", MockDateTime)

    mock_response                   = MagicMock()
    mock_response.message.content   = "Model generated query"
    mock_llm_call                   = MagicMock(return_value=mock_response)
    monkeypatch.setattr(LLM, "response_with_new_sys_prompt_and_context", mock_llm_call)

    result = agent.generates_query(context=[], prompt="This message is irrelavant in this test.")

    # Verify formatting expectations
    mock_llm_call.assert_called_once_with(
        model="mock_search_llm",
        system_prompt="Current Date: Saturday, 13 June 2026. Generate query for: ",
        context=[],
        prompt="This message is irrelavant in this test."
    )
    assert result == "Model generated query"


# ========================================
# Execution
# ========================================

def test_web(monkeypatch):
    agent = SearchAgent()
    
    # Mock result from query
    mock_results = [{"title": "Doc", "url": "http://doc.org", "content": "Text context snippets"}]
    monkeypatch.setattr(search_module, "_get_results_from_query", MagicMock(return_value=mock_results))

    # Mock downstream temp path generator
    monkeypatch.setattr(search_module, "_store_results_in_tmp_file", MagicMock(return_value="fake_tmp_file.txt"))
    
    # Mock reading utility execution context return payloads
    monkeypatch.setattr(search_module, "_read_tmp_file", MagicMock(return_value="Bundled Text File Context Body Content"))
    
    # Mock file clean up as 'fake_tmp_file.txt' does not exist
    mock_unlink = MagicMock()
    monkeypatch.setattr(os, "unlink", mock_unlink)
    monkeypatch.setattr(os.path, "exists", MagicMock(return_value=True))

    # Mock user message
    test_user_message = {"role": "user", "content": "This message is irrelavant in this test."}
    monkeypatch.setattr(LLM, "user", MagicMock(return_value=test_user_message))

    # Mock LLM response
    monkeypatch.setattr(LLM, "model_response", MagicMock(return_value=("Model reads context and responses.", 100, 50)))

    # Execute function
    response, prompt_tokens, out_tokens = agent.web(
        query="sanitized search context string",
        context=[{"role": "system", "content": "Init"}],
        prompt="User question about search content."
    )

    # Ensure correct returned values
    assert response == "Model reads context and responses."
    assert prompt_tokens == 100
    assert out_tokens == 50

    # Ensure context assembly was constructed correctly before delivery
    LLM.user.assert_called_once_with(
        "Context from web search:\n\nBundled Text File Context Body Content\n\nUser question: User question about search content."
    )
    
    # Verify the 'finally' block executed cleanup for the tmp file
    mock_unlink.assert_called_once_with("fake_tmp_file.txt")

def test_web_empty_results_fallback(monkeypatch):
    agent = SearchAgent()
    # Mock not result from '_get_results_from_query' in 'web'
    monkeypatch.setattr(search_module, "_get_results_from_query", MagicMock(return_value=[]))

    # Execute function
    result = agent.web("query string", [], "prompt string")

    # Ensure error message returned
    assert result == "Error: No web search results could be retrieved."
