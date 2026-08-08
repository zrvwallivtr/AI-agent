import os
import re
from typing import Optional
import pdfplumber
from pathlib import Path
import openpyxl
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings


from src.agent.llm import LLM
from src.agent.chat import Chat
from src import config


def _session_path(session: str | None = None) -> Path:
    if session is not None:
        return config.DROPBOX_DIR / session
    return config.DROPBOX_DIR / "chat"

def _read_file_in_dropbox(path) -> str:
    """Read file contents in dropbox."""
    with open(path, "r") as f:
        content = f.read()
    return content

def _is_dir_empty(path: Path) -> bool:
    """Return True if directory is not empty, otherwise 'False'."""
    with os.scandir(path) as entries:
        return next(entries, None) is not None


class FileReader:
    def __init__(self, session: str | None = None):
        self.model              = config.MODEL
        self.file_or_not_prompt = config.FILE_OR_NOT_PROMPT
        self.file_list_prompt   = config.GET_FILE_LIST_PROMPT
        self.chat               = Chat(session=session)
        self.store_file_path    = _session_path(session)

        # Map formats to their parsing methods
        self.formats = {
            # Plain text
            ".txt": self._read_txt,

            # Data & configuration formats
            ".csv": self._read_csv,
            ".xlsx": self._read_xlsx,
            ".yaml": self._read_yaml,
            ".yml": self._read_yml,
            ".toml": self._read_toml,
            ".xml": self._read_xml,
            
            # Text documents
            ".pdf": self._read_pdf,
            ".docx": self._read_docx,
            ".epub": self._read_epub,
            
            # Programming
            ".py": self._read_code,
            ".js": self._read_code,
            ".ts": self._read_code,
            ".tsx": self._read_code,
            ".json": self._read_code,
            ".md": self._read_code,
            ".sh": self._read_code,
            ".html": self._read_code,
            ".css": self._read_code,
            ".rs": self._read_code,
            ".go": self._read_code,
        }

    # ==================================================
    # Parsers
    #
    # - Extracts file contents line by line into plain
    #   text.
    # ==================================================

    def _read_txt(self, path: Path, file_type: str = "txt") -> str:
        """Reads plain text files using UTF-8 validation."""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"Error reading {file_type} file '{path}': {str(e)}"

    def _read_csv(self, path: Path) -> str:
        """Reads csv as plain text files."""
        return self._read_txt(path, "csv")

    def _read_xlsx(self, path: Path) -> str:
        """Extracts text content from spreadsheet row by row."""
        try:
            wb          = openpyxl.load_workbook(path, data_only=True)
            excel_text  = []

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                excel_text.append(f"=== Sheet: {sheet_name} ===")

                for row in ws.iter_rows(values_only=True):
                    # Only process lines that contain actual data elements
                    if any(cell is not None for cell in row):
                        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                        excel_text.append(" | ".join(clean_row))

            return "\n".join(excel_text)

        except Exception as e:
            return f"Error reading xlsx file '{path}': {str(e)}"

    def _read_yaml(self, path: Path) -> str:
        return self._read_txt(path, "yaml")

    def _read_yml(self, path: Path) -> str:
        return self._read_txt(path, "yml")

    def _read_toml(self, path: Path) -> str:
        return self._read_txt(path, "toml")

    def _read_xml(self, path: Path) -> str:
        return self._read_txt(path, "xml")

    def _read_pdf(self, path: Path) -> str:
        """Extracts pdf text content layout page by page."""
        try:
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
                pages = [p for p in pages if p]
                return "\n\n".join(pages)

        except Exception as e:
            return f"Error reading PDF file '{path}': {str(e)}"
    
    def _read_docx(self, path: Path) -> str:
        """Extracts structural text elements line by line from document."""
        try:
            doc = docx.Document(path)
            paragraphs = []

            for para in doc.paragraphs:
                clean_text = para.text.strip()
                if clean_text:
                    paragraphs.append(clean_text)

            return "\n".join(paragraphs)

        except Exception as e:
            return f"Error reading docx file '{path}': {str(e)}"

    def _read_epub(self, path: Path) -> str:
        """Extracts plain text blocks from internal EPUB XHTML document payloads."""
        try:
            book            = epub.read_epub(path)
            chapters_text   = []

            for item in book.get_items():
                # EPUB structural content is split into internal document items
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "lxml")

                    # Remove visual layout nodes that shouldn't feed context windows
                    for junk in soup(["script", "style"]):
                        junk.decompose()

                    plain_text = soup.get_text(separator="\n").strip()
                    if plain_text:
                        chapters_text.append(plain_text)

            return "\n\n".join(chapters_text)
        except Exception as e:
            return f"Error reading EPUB file '{path}': {str(e)}"

    def _read_code(self, path: Path) -> str:
        """Reads code files and wraps them in markdown code fences."""
        lang    = path.suffix.lstrip(".")
        content = self._read_txt(path, f"{lang}")

        return f"```{lang}\n{content}\n```"

    # ==================================================
    # Storage uploaded files in dropbox
    # ==================================================
    
    # add a short summary for every file in dropbox
    def _store_content_in_dropbox(self, content: str, filename: str) -> None | str:
        """Store file contents in the dropbox."""
        try:
            self.store_file_path.mkdir(parents=True, exist_ok=True)
            path = self.store_file_path / filename
            path.write_text(content, encoding="utf-8")

        except Exception as e:
            return f"Failed to cache context in workspace storage: {e}"

    # list filenames in dropbox along side with the short summary
    def list_available_files(self) -> list[str]:
        """Return an index of files currently in the directory."""
        if not self.store_file_path.exists():
            return []

        return [f.name for f in self.store_file_path.iterdir () if f.is_file()]

    def clear_session_dropbox(self, session: str | None = None) -> str:
        """Remove session dropbox directory when cleaning history."""
        if self.store_file_path.exists():
            for child in self.store_file_path.iterdir():
                if child.is_file():
                    child.unlink()

            self.store_file_path.rmdir()
            return f"Cleared all files in session {session}'s dropbox."

        return f"Error: Session {session}'s dropbox not found."

    # ==================================================
    # Load files
    # ==================================================

    def load_file_contents(self, filenames: list[str]) -> str:
        """From a list of paths, returns contents in path as a string."""
        blocks = []
        paths = [Path(config.DROPBOX_DIR / filename) for filename in filenames]

        for path in paths:

            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()

                # Format
                block = (
                    f"Context from file:"
                    f"Filename = {path.name}\n"
                    f"{content}\n"
                    "="*40
                )
                blocks.append(block)

            except Exception as e:
                print(f"Error reading file {path}: {e}")

        return "\n\n".join(blocks)

    # ==================================================
    # Model integration
    # ==================================================

    def require_file_or_not(self, context: list[dict], prompt: str) -> bool:
        """Query model to decide if file context is needed, return 'True' or 'False' only."""
        response = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.file_or_not_prompt,
            context=context,
            prompt=prompt
        )

        output = response.message.content

        if 'true' in output.lower():
            return True
        else:
            return False

    def get_filenames(self, context: list[dict], prompt: str) -> tuple[list[str], list[str]]:
        """Ask model to choose from available in current session to get relevant context."""
        response = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.file_or_not_prompt,
            context=context,
            prompt=prompt
        )

        list_of_files = response.message.content

        content                 = response.message.content
        found_file_paths        = []
        not_found_file_paths    = []

        # Reformats generated string to a clean list
        parsed_names = []
        for token in re.split(r"[,\n]", content):
            name = re.sub(r"[*'\"\'\-]", "", token).strip()
            if name:
                parsed_names.append(name)

        # Segregate file paths into found and not found
        for file in parsed_names:
            file_path = self.store_file_path / file

            if file_path.exists():
                found_file_paths.append(file)
            else:
                not_found_file_paths.append(file)

        return found_file_paths, not_found_file_paths

    def read_files_with_context_prompt(
            self,
            context: list[dict],
            context_prompt: str,
            list_of_files: list[str] | None,
            prompt: str
    ) -> tuple[str, int, int]:
        """Read all files contents from a list of filenames."""
        labelled_content = ["Here are the required context:\n"]

        if list_of_files:

            for filename in list_of_files:
                path    = Path(self.store_file_path / filename)
                ext     = path.suffix.lower()

                if not path.exists():
                    return f"Error: File {path} not found.", 0, 0

                # Route to correct parser, fallback to plain text if unknown
                parser = self.formats.get(ext, self._read_txt)
                file_content = parser(path)

                block = (
                    f"Context from file:"
                    f"Filename = {filename}\n"
                    f"{file_content}\n"
                    "="*40
                )
                labelled_content.append(block)

        else:
            return "Error: No files provided", 0, 0

        next_message    = LLM.user(f"{context_prompt}\n{labelled_content}\nUser input:\n{prompt}")
        messages        = context + [next_message]

        response, prompt_tokens, output_tokens = LLM.model_response(messages, self.model)
        return response, prompt_tokens, output_tokens

    # ==================================================
    # Auto function
    # ==================================================

    def toggle_auto_read_dropbox(
            self,
            max_tokens: int,
            auto_read_dropbox: bool,
            messages: list[dict],
            prompt: str
    ) -> tuple[str, list[str], bool]:
        """
        Auto fetches previous file contents ability, return relevant files if its toggled on.

        Model decides from {memory_entries} + {prompt} --> {file_content} from dropbox
        """
        if auto_read_dropbox == True and max_tokens > config.AUTO_READ_DROPBOX_TOKENS:

            # Check if dropbox is empty
            content_exists = _is_dir_empty(config.DROPBOX_DIR)
            
            require_file = self.require_file_or_not(messages, prompt)

            if require_file == True:
                found_files, not_found_files = self.get_filenames(messages, prompt)

                return self.load_file_contents(found_files), found_files, True

            return "", [], False

        else:
            return "", [], False
