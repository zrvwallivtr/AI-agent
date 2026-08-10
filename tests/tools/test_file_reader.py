import json
import pytest
from os.path import exists
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.llm import LLM
import src.tools.file_reader as file_reader_module
from src.tools.file_reader import FileReader, _session_path, _is_dir_empty, _read_file
from tools import file_reader
from src import config


# =============================================================
# _session_path
# =============================================================

def test_session_path(test_config):
    session_path = _session_path()
    path = test_config.tmp_dir / ".agent_app" / "test_data" / "dropbox" / "chat"

    assert session_path == path

def test_session_path_custom_session(test_config):
    session_path = _session_path(session="custom_session")
    path = test_config.tmp_dir / ".agent_app" / "test_data" / "dropbox" / "custom_session"

    assert session_path == path


# =============================================================
# _is_dir_empty
# =============================================================

def test_is_dir_empty(test_config):
    empty_dir = test_config.tmp_dir / "empty_dir"
    empty_dir.mkdir()

    assert _is_dir_empty(empty_dir) is True

def test_is_dir_empty_with_files(test_config):
    contents_dir = test_config.tmp_dir / "contents"
    contents_dir.mkdir()
    (contents_dir / "content.txt").write_text("Hello world")

    assert _is_dir_empty(contents_dir) is False

def test_is_dir_empty_non_exists_dir(test_config):
    non_exist_path = test_config.tmp_dir / "not_exists"

    assert _is_dir_empty(non_exist_path) is False


# =============================================================
# _read_file
# =============================================================

def test_read_file(test_config):
    file_dir = test_config.tmp_dir / "text_file"
    file_path = file_dir / "content.txt"
    file_dir.mkdir(parents=True, exist_ok=True)
    (file_dir / "content.txt").write_text("Hello world")

    content = _read_file(file_path)

    assert content == "Hello world"


# =============================================================
# FileReader.__init__
# =============================================================

def test_file_reader_init_default(test_config):
    test_dir = test_config.tmp_dir / ".agent_app" / "test_data" / "dropbox" / "chat"
    test_dir.mkdir(parents=True, exist_ok=True)

    file_reader = FileReader()

    assert file_reader.model == "mock_model"
    assert file_reader.dropbox_dir == test_dir
    assert file_reader.file_or_not_prompt == config.FILE_OR_NOT_PROMPT
    assert file_reader.file_list_prompt == config.GET_FILE_LIST_PROMPT
    assert file_reader.metadata_path == test_dir / "file_metadata.json"
    assert file_reader.file_metadata == {}
    assert file_reader.gen_summary_prompt == config.GEN_SUMMARY_PROMPT

def test_file_reader_init_custom_session(test_config):
    test_dir = test_config.tmp_dir / ".agent_app" / "test_data" / "dropbox" / "custom_session"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "file_metadata.json").write_text('"Test"')

    file_reader = FileReader(session="custom_session")

    assert file_reader.model == "mock_model"
    assert file_reader.dropbox_dir == test_dir
    assert file_reader.file_or_not_prompt == config.FILE_OR_NOT_PROMPT
    assert file_reader.file_list_prompt == config.GET_FILE_LIST_PROMPT
    assert file_reader.metadata_path == test_dir / "file_metadata.json"
    assert file_reader.file_metadata == "Test"
    assert file_reader.gen_summary_prompt == config.GEN_SUMMARY_PROMPT


# =============================================================
# _read_txt
# =============================================================

def test_read_txt(test_config):
    file_reader = FileReader()
    txt_file = test_config.tmp_dir / "test.txt"
    txt_file.write_text("Hello world\nTesting UTF-8", encoding="utf-8")

    result = file_reader._read_txt(txt_file)
    assert result == "Hello world\nTesting UTF-8"

def test_read_txt_error_handling(tmp_path):
    file_reader = FileReader()
    missing_file = tmp_path / "does_not_exist.txt"
    
    with patch.object(Path, "read_text", side_effect=OSError("File read error")):
        result = file_reader._read_txt(missing_file, file_type="txt")
        assert "Error reading txt file" in result
        assert "File read error" in result


# =============================================================
# _read_csv
# =============================================================

def test_read_csv(tmp_path):
    file_reader = FileReader()
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,age\nAlice,30", encoding="utf-8")

    result = file_reader._read_csv(csv_file)
    assert result == "name,age\nAlice,30"


# ==============================================================================
# _read_xlsx
# ==============================================================================

@patch("openpyxl.load_workbook")
def test_read_xlsx_success(mock_load_workbook):
    file_reader = FileReader()
    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    
    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = [
        ("Header 1", "Header 2"),
        (None, None),  # Empty row, should be skipped
        ("Value 1", 100),
    ]
    mock_wb.__getitem__.return_value = mock_ws
    mock_load_workbook.return_value = mock_wb

    result = file_reader._read_xlsx(Path("test.xlsx"))
    
    assert "=== Sheet: Sheet1 ===" in result
    assert "Header 1 | Header 2" in result
    assert "Value 1 | 100" in result

@patch("openpyxl.load_workbook", side_effect=Exception("Corrupted sheet"))
def test_read_xlsx_exception(mock_load_workbook):
    file_reader = FileReader()
    result = file_reader._read_xlsx(Path("bad.xlsx"))
    assert "Error reading xlsx file 'bad.xlsx': Corrupted sheet" in result


# =============================================================
# _read_yaml
# =============================================================

def test_read_yaml(tmp_path):
    file_reader = FileReader()
    yaml_file = tmp_path / "config.yaml"
    
    content = "version: '1.0'\napp: test"
    yaml_file.write_text(content, encoding="utf-8")

    assert file_reader._read_yaml(yaml_file) == content


# =============================================================
# _read_yml
# =============================================================

def test_read_yml(tmp_path):
    file_reader = FileReader()
    yml_file = tmp_path / "config.yml"
    
    content = "version: '1.0'\napp: test"
    yml_file.write_text(content, encoding="utf-8")

    assert file_reader._read_yml(yml_file) == content


# =============================================================
# _read_toml
# =============================================================

def test_read_toml(tmp_path):
    file_reader = FileReader()
    toml_file = tmp_path / "config.toml"
    content = "[tool.pytest]\nminversion = '6.0'"
    toml_file.write_text(content, encoding="utf-8")

    assert file_reader._read_toml(toml_file) == content


# =============================================================
# _read_xml
# =============================================================

def test_read_xml(tmp_path):
    file_reader = FileReader()
    xml_file = tmp_path / "data.xml"
    content = "<root><data>test</data></root>"
    xml_file.write_text(content, encoding="utf-8")

    assert file_reader._read_xml(xml_file) == content

# ==============================================================================
# _read_pdf
# ==============================================================================

@patch("pdfplumber.open")
def test_read_pdf_success(mock_pdf_open):
    file_reader = FileReader()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 Content"
    
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = None  # Empty page
    
    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = "Page 3 Content"

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    result = file_reader._read_pdf(Path("doc.pdf"))
    assert result == "Page 1 Content\n\nPage 3 Content"

@patch("pdfplumber.open", side_effect=Exception("Encrypted PDF"))
def test_read_pdf_exception(mock_pdf_open):
    file_reader = FileReader()
    result = file_reader._read_pdf(Path("protected.pdf"))
    assert "Error reading PDF file 'protected.pdf': Encrypted PDF" in result


# ==============================================================================
# _read_docx
# ==============================================================================

@patch("docx.Document")
def test_read_docx_success(mock_docx):
    file_reader = FileReader()
    mock_p1 = MagicMock()
    mock_p1.text = "   Paragraph One   "
    
    mock_p2 = MagicMock()
    mock_p2.text = "      "  # Whitespace paragraph, should be skipped
    
    mock_p3 = MagicMock()
    mock_p3.text = "Paragraph Two"

    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_p1, mock_p2, mock_p3]
    mock_docx.return_value = mock_doc

    result = file_reader._read_docx(Path("doc.docx"))
    assert result == "Paragraph One\nParagraph Two"

@patch("docx.Document", side_effect=Exception("Invalid docx format"))
def test_read_docx_exception(mock_docx):
    file_reader = FileReader()
    result = file_reader._read_docx(Path("broken.docx"))
    assert "Error reading docx file 'broken.docx': Invalid docx format" in result


# ==============================================================================
# _read_epub
# ==============================================================================

@patch("ebooklib.epub.read_epub")
def test_read_epub_success(mock_read_epub):
    file_reader = FileReader()
    mock_item = MagicMock()
    mock_item.get_type.return_value = 9  # ITEM_DOCUMENT in ebooklib
    mock_item.get_content.return_value = (
        b"<html><body>"
        b"<style>body {color: red;}</style>"
        b"<h1>Chapter 1</h1><p>Text content here.</p>"
        b"<script>console.log('test');</script>"
        b"</body></html>"
    )

    mock_book = MagicMock()
    mock_book.get_items.return_value = [mock_item]
    mock_read_epub.return_value = mock_book

    with patch("ebooklib.ITEM_DOCUMENT", 9):
        result = file_reader._read_epub(Path("book.epub"))
        
        # Ensures script and style tags were removed while extracting content
        assert "Chapter 1" in result
        assert "Text content here." in result
        assert "color: red" not in result
        assert "console.log" not in result

@patch("ebooklib.epub.read_epub", side_effect=Exception("Invalid EPUB zip"))
def test_read_epub_exception(mock_read_epub):
    file_reader = FileReader()
    result = file_reader._read_epub(Path("corrupt.epub"))
    assert "Error reading EPUB file 'corrupt.epub': Invalid EPUB zip" in result


# =============================================================
# _read_code
# =============================================================

def test_read_code(tmp_path):
    file_reader = FileReader()
    py_file = tmp_path / "script.py"
    py_file.write_text("print('hello')", encoding="utf-8")

    result = file_reader._read_code(py_file)
    expected = "```py\nprint('hello')\n```"
    assert result == expected


# =============================================================
# _add_file_metadata
# =============================================================

def test_add_file_metadata():
    file_reader = FileReader()

    filename = "test_document.txt"
    file_path = file_reader.dropbox_dir / filename
    content = "Hello world, unit test content"
    file_path.write_text(content, encoding="utf-8")

    summary = "This is a test summary."
    file_reader._add_file_metadata(filename, summary)

    assert filename in file_reader.file_metadata
    metadata = file_reader.file_metadata[filename]
    assert metadata["summary"] == summary
    assert metadata["path"] == str(file_path)
    assert metadata["mime_type"] == "text/plain"
    assert metadata["size_bytes"] == len(content.encode("utf-8"))

    assert file_reader.metadata_path.exists()
    saved_json = json.loads(file_reader.metadata_path.read_text(encoding="utf-8"))
    assert saved_json == file_reader.file_metadata

def test_add_file_metadata_multi_entries():
    file_reader = FileReader(session="test_multi_entries")

    file1 = file_reader.dropbox_dir / "doc1.txt"
    file2 = file_reader.dropbox_dir / "doc2.txt"
    file3 = file_reader.dropbox_dir / "doc3.txt"
    file1.write_text("Content 1", encoding="utf-8")
    file2.write_text("Content 2", encoding="utf-8")
    file3.write_text("Content 3", encoding="utf-8")

    file_reader._add_file_metadata("doc1.txt", "Summary 1")
    file_reader._add_file_metadata("doc2.txt", "Summary 2")
    file_reader._add_file_metadata("doc3.txt", "Summary 3")

    assert "doc1.txt" in file_reader.file_metadata
    assert "doc2.txt" in file_reader.file_metadata
    assert "doc3.txt" in file_reader.file_metadata

    saved_json = json.loads(file_reader.metadata_path.read_text(encoding="utf-8"))
    assert len(saved_json) == 3
    assert saved_json["doc1.txt"]["summary"] == "Summary 1"
    assert saved_json["doc2.txt"]["summary"] == "Summary 2"
    assert saved_json["doc3.txt"]["summary"] == "Summary 3"


# =============================================================
# _get_filenames_from_metadata
# =============================================================

def test_get_filenames_from_metadata():
    file_reader = FileReader(session="test_get_filenames")

    file1 = file_reader.dropbox_dir / "script1.py"
    file2 = file_reader.dropbox_dir / "script2.sh"
    file3 = file_reader.dropbox_dir / "script3.txt"
    file1.write_text("Content 1", encoding="utf-8")
    file2.write_text("Content 2", encoding="utf-8")
    file3.write_text("Content 3", encoding="utf-8")

    file_reader._add_file_metadata("script1.py", "Summary 1")
    file_reader._add_file_metadata("script2.sh", "Summary 2")
    file_reader._add_file_metadata("script3.txt", "Summary 3")

    filenames = file_reader._get_filenames_from_metadata()

    assert filenames == ["script1.py", "script2.sh", "script3.txt"]


# =============================================================
# _files_not_in_file_metadata
# =============================================================

def test_files_not_in_file_metadata():
    file_reader = FileReader(session="test_files_not_in_metadata")

    file1 = file_reader.dropbox_dir / "script1.py"
    file2 = file_reader.dropbox_dir / "script2.sh"
    file3 = file_reader.dropbox_dir / "script3.txt"
    file1.write_text("Content 1", encoding="utf-8")
    file2.write_text("Content 2", encoding="utf-8")
    file3.write_text("Content 3", encoding="utf-8")

    file_reader._add_file_metadata("script3.txt", "Summary 3")

    filenames = file_reader._files_not_in_file_metadata()

    assert "script1.py" in filenames
    assert "script2.sh" in filenames
    assert "script3.txt" not in filenames

def test_files_not_in_file_metadata_return_none():
    file_reader = FileReader(session="test_files_in_metadata")

    file1 = file_reader.dropbox_dir / "script1.py"
    file2 = file_reader.dropbox_dir / "script2.sh"
    file3 = file_reader.dropbox_dir / "script3.txt"
    file1.write_text("Content 1", encoding="utf-8")
    file2.write_text("Content 2", encoding="utf-8")
    file3.write_text("Content 3", encoding="utf-8")

    file_reader._add_file_metadata("script1.py", "Summary 1")
    file_reader._add_file_metadata("script2.sh", "Summary 2")
    file_reader._add_file_metadata("script3.txt", "Summary 3")

    filenames = file_reader._files_not_in_file_metadata()

    assert "script1.py" not in filenames
    assert "script2.sh" not in filenames
    assert "script3.txt" not in filenames


# =============================================================
# _list_available_files
# =============================================================

def test_list_available_files():
    file_reader = FileReader(session="available_files")

    file1 = file_reader.dropbox_dir / "script1.py"
    file2 = file_reader.dropbox_dir / "script2.sh"
    file3 = file_reader.dropbox_dir / "script3.txt"
    ignore_file = file_reader.dropbox_dir / "file_metadata.json"
    file1.write_text("Content 1", encoding="utf-8")
    file2.write_text("Content 2", encoding="utf-8")
    file3.write_text("Content 3", encoding="utf-8")
    ignore_file.write_text("'Content'", encoding="utf-8")

    filenames = file_reader._list_available_files()

    assert "script1.py" in filenames
    assert "script2.sh" in filenames
    assert "script3.txt" in filenames
    assert "file_metadata.json" not in filenames


# =============================================================
# _store_file_in_dropbox
# =============================================================

def test_store_file_in_dropbox():
    file_reader = FileReader(session="store_file")

    content = "Content"
    filename = "file.txt"
    response = file_reader._store_file_in_dropbox(content, filename)

    path = Path(file_reader.dropbox_dir / filename)
    test = path.read_text()

    assert path.exists()
    assert test == "Content"


# =============================================================
# _clear_session_dropbox
# =============================================================

def test_clear_session_dropbox():
    file_reader = FileReader(session="clear_session")

    content = "Content"
    filename = "file.txt"
    response = file_reader._store_file_in_dropbox(content, filename)

    path = Path(file_reader.dropbox_dir / filename)
    test = path.read_text()
    assert path.exists()

    response = file_reader.clear_session_dropbox()
    assert not path.exists()
    assert response == f"Cleared all files in '{file_reader.dropbox_dir}'."


# =============================================================
# _load_file_content
# =============================================================

def test_load_file_content():
    file_reader = FileReader(session="load_file_content")
    filename = "doc.txt"
    file_path = file_reader.dropbox_dir / filename
    file_path.write_text("Sample file content", encoding="utf-8")

    mock_parser = MagicMock(return_value="Parsed content from mock")
    file_reader.formats[".txt"] = mock_parser

    content, path, read_file = file_reader._load_file_content(filename)

    assert read_file is True
    assert path == file_path
    assert content == "Parsed content from mock"
    mock_parser.assert_called_once_with(file_path)

def test_load_file_content_not_found():
    file_reader = FileReader(session="file_not_found")
    filename = "non_existing.txt"
    file_path = file_reader.dropbox_dir / filename

    content, path, read_file = file_reader._load_file_content(filename)

    assert read_file is False
    assert path == file_path
    assert content == f"Error: File {file_path} not found."

def test_load_file_content_uppercase():
    file_reader = FileReader(session="load_file_uppercase")
    filename = "DATA.CSV"
    file_path = file_reader.dropbox_dir / filename
    file_path.write_text("col1,col2,col3\n1,2,3", encoding="utf-8")

    mock_parser = MagicMock(return_value="Parsed csv data")
    file_reader.formats[".csv"] = mock_parser

    content, path, read_file = file_reader._load_file_content(filename)

    assert read_file is True
    assert path == file_path
    assert content == "Parsed csv data"
    mock_parser.assert_called_once_with(file_path)

def test_load_file_content_unknown_fallback():
    file_reader = FileReader(session="unknown_file")
    filename = "file.unknown"
    file_path = file_reader.dropbox_dir / filename
    file_path.write_text("Sample file content", encoding="utf-8")

    with patch.object(file_reader, "_read_txt", return_value="Sample file content") as mock_read_txt:
        content, path, read_file = file_reader._load_file_content(filename)

        assert read_file is True
        assert path == file_path
        assert content == "Sample file content"
        mock_read_txt.assert_called_once_with(file_path)


# =============================================================
# _load_contents_from_file_list
# =============================================================

def test_load_contents_from_file_list():
    file_reader = FileReader(session="load_file_content")
    filenames = ["doc1.txt", "doc2.txt", "doc3.txt"]

    mock_responses = {
        "doc1.txt": ("Content of doc1", file_reader.dropbox_dir / "doc1.txt", True),
        "doc2.txt": ("Content of doc2", file_reader.dropbox_dir / "doc2.txt", True),
        "doc3.txt": ("Content of doc3", file_reader.dropbox_dir / "doc3.txt", True)
    }

    def mock_load_content(filename):
        return mock_responses[filename]

    with patch.object(file_reader, "_load_file_content", side_effect=mock_load_content):
        result = file_reader._load_contents_from_file_list(filenames)

        expected_block_1 = (
            "Context from file:\n"
            "Filename = doc1.txt\n"
            "Content of doc1\n"
            "========================================"
        )
        expected_block_2 = (
            "Context from file:\n"
            "Filename = doc2.txt\n"
            "Content of doc2\n"
            "========================================"
        )
        expected_block_3 = (
            "Context from file:\n"
            "Filename = doc3.txt\n"
            "Content of doc3\n"
            "========================================"
        )
        expected_result = f"{expected_block_1}\n\n{expected_block_2}\n\n{expected_block_3}"

        assert result == expected_result

def test_load_contents_from_file_list_from_empty_list():
    file_reader = FileReader(session="empty_file_list")

    result = file_reader._load_contents_from_file_list([])

    assert result == "No valid file context found."

def test_load_contents_from_file_list_print_missing(capsys):
    file_reader = FileReader(session="list_all missing")
    filenames = ["missing1.txt", "missing2.txt", "exists.txt"]

    missing_path_1 = file_reader.dropbox_dir / "missing1.txt"
    missing_path_2 = file_reader.dropbox_dir / "missing2.txt"
    existing_path_3 = file_reader.dropbox_dir / "exists.txt"

    mock_responses = {
        "missing1.txt": ("Error: Not found", missing_path_1, False),
        "missing2.txt": ("Error: Not found", missing_path_2, False),
        "exists.txt": ("Valid file content", existing_path_3, True),
    }

    with patch.object(file_reader, "_load_file_content", side_effect=lambda f: mock_responses[f]):
        result = file_reader._load_contents_from_file_list(filenames)

        assert "Filename = missing1.txt" not in result
        assert "Filename = missing2.txt" not in result
        assert "Filename = exists.txt" in result
        assert "Valid file content" in result

        captured = capsys.readouterr()
        assert f"Warning: Error reading file {missing_path_1}" in captured.out
        assert f"Warning: Error reading file {missing_path_2}" in captured.out


# =============================================================
# _generate_short_summary
# =============================================================

@patch("src.tools.file_reader.LLM")
def test_generate_short_summary(mock_llm):
    file_reader = FileReader(session="summary")
    filename = "doc.txt"
    file_path = file_reader.dropbox_dir / filename

    mock_response = MagicMock()
    mock_response.message.content = "Summary of document"
    mock_llm.response_with_new_sys_prompt_and_context.return_value = mock_response

    with patch.object(file_reader, "_load_file_content", return_value=("File content here", file_path, True)):
        result = file_reader._generate_short_summary(filename)

        assert result == "Summary of document"
        mock_llm.response_with_new_sys_prompt_and_context.assert_called_once()


# =============================================================
# _add_metadata_and_summary
# =============================================================

def test_add_metadata_and_summary():
    file_reader = FileReader(session="add_metadata_and_summary")
    filenames = ["file1.txt", "file2.txt", "file3.txt"]

    with patch.object(file_reader, "_generate_short_summary", return_value="Test summary") as mock_gen_summary, \
        patch.object(file_reader, "_add_file_metadata") as mock_add_meta:

        file_reader._add_metadata_and_summary(filenames)

        assert mock_gen_summary.call_count == 3
        assert mock_add_meta.call_count == 3
        mock_add_meta.assert_any_call("file1.txt", "Test summary")
        mock_add_meta.assert_any_call("file2.txt", "Test summary")
        mock_add_meta.assert_any_call("file3.txt", "Test summary")


# =============================================================
# _structured_file_string
# =============================================================

def test_structured_file_string():
    file_reader = FileReader(session="structured_file_string")

    result = file_reader._structured_file_string("data.csv", "Contains metrics")

    expected_result = (
        "Filename = data.csv\n"
        "Summary = Contains metrics\n"
        "========================================\n"
    )
    assert result == expected_result


# =============================================================
# _require_file_or_not
# =============================================================

@patch("src.tools.file_reader.LLM")
def test_require_file_or_not(mock_llm):
    file_reader = FileReader(session="require_or_not")
    mock_response = MagicMock()
    mock_response.message.content = "TRUE"
    mock_llm.response_with_new_sys_prompt_and_context.return_value = mock_response

    result = file_reader._require_file_or_not(context=[], prompt="This is a message.")

    assert result is True


# =============================================================
# get_filenames
# =============================================================

@patch("src.tools.file_reader.LLM")
def test_get_filenames(mock_llm):
    file_reader = FileReader(session="get_filenames")

    existing_file = "found.txt"
    (file_reader.dropbox_dir / existing_file).write_text("exists")

    mock_response = MagicMock()
    mock_response.message.content = "- 'found.txt'\n* \"missing.txt\""
    mock_llm.response_with_new_sys_prompt_and_context.return_value = mock_response

    found, not_found = file_reader.get_filenames(context=[], available_files="found.txt, missing.txt")

    assert found == ["found.txt"]
    assert not_found == ["missing.txt"]


# =============================================================
# read_files_with_context_prompt
# =============================================================

@patch("src.tools.file_reader.LLM")
def test_read_files_with_context_prompt(mock_llm):
    file_reader = FileReader()
    file_path = file_reader.dropbox_dir / "doc.txt"

    mock_llm.user.return_value = {"role": "user", "content": "formatted prompt"}
    mock_llm.model_response.return_value = ("Model answer", 100, 50)

    with patch.object(file_reader, "_load_file_content", return_value=("Content body", file_path, True)):
        response, p_tokens, o_tokens = file_reader.read_files_with_context_prompt(
            context=[], context_prompt="Use context:", filename_list=["doc.txt"], prompt="Summarize"
        )

        assert response == "Model answer"
        assert p_tokens == 100
        assert o_tokens == 50

def test_read_files_with_context_prompt_no_files():
    file_reader = FileReader()

    response, prompt_tokens, output_tokens = file_reader.read_files_with_context_prompt(
        context=[],
        context_prompt="Context:",
        filename_list=None,
        prompt="Summarize"
    )

    assert response == "Error: No files provided"
    assert prompt_tokens == 0
    assert output_tokens == 0

def test_read_files_with_context_prompt_file_missing():
    file_reader = FileReader()
    missing_path = file_reader.dropbox_dir / "missing.txt"

    with patch.object(file_reader, "_load_file_content", return_value=("Error", missing_path, False)):
        response, p_tokens, o_tokens = file_reader.read_files_with_context_prompt(
            context=[], context_prompt="Context:", filename_list=["missing.txt"], prompt="Summarize"
        )

        assert response == f"Error: File {missing_path} not found."
        assert p_tokens == 0
        assert o_tokens == 0


# =============================================================
# _file_context
# =============================================================

def test_file_context():
    file_reader = FileReader()
    test_metadata = {
        "file1.txt": {"summary": "Summary 1"}
    }
    file_reader.file_metadata = test_metadata

    with patch.object(file_reader, "_files_not_in_file_metadata", return_value=["file1.txt"]), \
         patch.object(file_reader, "_add_metadata_and_summary"), \
         patch.object(file_reader, "_load_file_metadata", return_value=test_metadata), \
         patch.object(file_reader, "_list_available_files", return_value=["file1.txt"]):

        result = file_reader._file_context()

        assert "All available files:" in result
        assert "Filename = file1.txt" in result
        assert "Summary = Summary 1" in result


# =============================================================
# toggle_auto_read_dropbox
# =============================================================

def test_toggle_auto_read_dropbox_disabled():
    file_reader = FileReader()

    content, files, active = file_reader.toggle_auto_read_dropbox(
        max_tokens=10000, auto_read_dropbox=False, messages=[], prompt="test"
    )
    assert active is False
    assert files == []
    assert content == ""

@patch("src.tools.file_reader._is_dir_empty", return_value=True)
def test_toggle_auto_read_dropbox_empty_dir(mock_is_dir_empty):
    file_reader = FileReader()

    with patch("src.config.AUTO_READ_DROPBOX_TOKENS", 100):
        content, files, active = file_reader.toggle_auto_read_dropbox(
            max_tokens=500, auto_read_dropbox=True, messages=[], prompt="test"
        )

        assert active is False
        assert files == []
        assert content == ""

def test_toggle_auto_read_dropbox_active_and_files_found():
    file_reader = FileReader()

    with patch("src.config.AUTO_READ_DROPBOX_TOKENS", 100), \
         patch("src.tools.file_reader._is_dir_empty", return_value=False), \
         patch.object(file_reader, "_require_file_or_not", return_value=True), \
         patch.object(file_reader, "_file_context", return_value="Context string"), \
         patch.object(file_reader, "get_filenames", return_value=(["found.txt"], [])), \
         patch.object(file_reader, "_load_contents_from_file_list", return_value="Loaded content"):

        content, files, active = file_reader.toggle_auto_read_dropbox(
            max_tokens=500, auto_read_dropbox=True, messages=[], prompt="test"
        )

        assert active is True
        assert files == ["found.txt"]
        assert content == "Loaded content"
