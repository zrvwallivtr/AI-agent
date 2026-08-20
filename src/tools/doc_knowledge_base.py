import os
from os.path import exists
import re
import json
import mimetypes
import warnings
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Any

from src.agent.llm import LLM
from src.agent.chat import Chat
from src.agent.tokens_handler import Tokens
from src.tools.parsers import Parsers
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


def session_path(session: str | None = None) -> Path:
    if session is not None:
        return config.DROPBOX_DIR / session
    return config.DROPBOX_DIR / "chat"

def _is_dir_empty(path: Path) -> bool:
    """Return True if directory is empty, otherwise 'False'."""
    if not path.exists():
        return False
    with os.scandir(path) as entries:
        return next(entries, None) is None

def _read_file(path) -> str:
    """Return content in file path."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return content


class DocKnowledgeBase:
    def __init__(
        self,
        session: str | None = None,
    ):
        self.session            = session
        self.session_name       = self.session or "default_session"
        self.chat               = Chat(session=session)
        self.model              = config.MODEL
        self.tokens             = Tokens(model=self.model)
        self.parsers            = Parsers()

        # Files
        self.dropbox_dir        = session_path(session)
        self.file_or_not_prompt = config.FILE_OR_NOT_PROMPT
        self.file_list_prompt   = config.GET_FILE_LIST_PROMPT
        
        # Dropbox
        self.metadata_path      = session_path(session) / "file_metadata.json"
        self.file_metadata      = self._load_file_metadata()
        self.gen_summary_prompt = config.GEN_SUMMARY_PROMPT

    # ================================================
    # Metadata (file_metadata.json)
    # ================================================

    def _load_file_metadata(self) -> dict[str, dict[str, Any]]:
        """Fetch/create metadata file, returns metadata as dict."""
        if not self.dropbox_dir.exists():
            self.dropbox_dir.mkdir(parents=True, exist_ok=True)

        # Return '{}' if metadata file not exists
        if not self.metadata_path.exists():
            default_data = {}
            self.metadata_path.write_text(json.dumps(default_data, indent=4), encoding="utf-8")
            logger.info(
                "Initiated 'file_metadata.json' in dropbox: session=%s, path='%s'",
                self.session_name,
                self.metadata_path
            )
            return default_data

        try:
            # Return entire dictionary
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            logger.info(
                "Extracted all metadata in dropbox: session=%s, path='%s'",
                self.session_name,
                self.metadata_path
            )
            return metadata

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode JSON payload: path='%s', line=%d, col=%d, error=%s",
                self.metadata_path,
                e.lineno,
                e.colno,
                e.msg
            )
            return {}

    def _add_file_metadata(self, filename: str, summary: str):
        """
        Add metadata of files that are already in
        the dropbox to 'file_metadata.json'.

        {
            "filename": {
                "summary": ,
                "mime_type": ,
                "size_bytes":
            }
        }
        """
        path = self.dropbox_dir / filename
        mime_type, _ = mimetypes.guess_type(path)

        # Ensure 'self.file_metadata' is a dictionary
        if not isinstance(self.file_metadata, dict):
            raise TypeError(
                f"Error: Expected 'file_metadata' to be a dict, but got {type(self.file_metadata).__name__}"
                f"with value: {self.file_metadata!r}"
            )

        # Prevent duplicated filename in the metadata
        entry_name = path.name
        counter = 1
        while entry_name in self.file_metadata:
            entry_name = f"{path.stem}({counter}){path.suffix}"
            counter += 1

        # Add entry for file
        self.file_metadata[entry_name] = {
            "summary": f"{summary}",
            "mime_type": mime_type,
            "size_bytes": path.stat().st_size if path.exists() else 0
        }
        self.dropbox_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(self.file_metadata, indent=4), encoding="utf-8")
        logger.info(
            "New file entry added to 'file_metadata.json': file=%s, session=%s, path='%s'",
            filename,
            self.session_name,
            self.metadata_path
        )

    def _get_filenames_from_metadata(self) -> list[str]:
        """Return all filenames from 'file_metadata.json'."""
        if not self.metadata_path.exists():
            logger.warning(
                "Failed to retrive filename(s). 'file_metadata.json' does not exists: session=%s, path='%s'",
                self.session_name,
                self.metadata_path
            )
            return []

        try:
            data        = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            filenames   = list(data.keys())
            logger.info(
                "Retrived %d filename(s) from 'file_metadata.json': session=%s, path='%s'",
                len(filenames),
                self.session_name,
                self.metadata_path
            )
            return filenames

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode JSON payload: path='%s', line=%d, col=%d, error=%s",
                self.metadata_path,
                e.lineno,
                e.colno,
                e.msg
            )
            return []

    def _files_not_in_file_metadata(self) -> list[str]:
        """
        Return a list of available files that is
        not in 'file_metadata.json' (without summary).
        """
        files_with_summary      = self._get_filenames_from_metadata()
        available_files         = self.list_available_files()
        files_without_summary   = list(set(available_files) - set(files_with_summary))

        logger.info(
            "%d file(s) not registered in 'file_metadata.json': session=%s, path='%s'",
            len(files_without_summary),
            self.session_name,
            self.metadata_path
        )
        return files_without_summary

    # ================================================
    # Dropbox management
    # ================================================

    def list_available_files(self) -> list[str]:
        """Return a list of files currently in the directory."""
        if not self.dropbox_dir.exists():
            logger.warning(
                "No available files was retrieved. Session dropbox not found: session=%s, path='%s'",
                self.session_name,
                self.dropbox_dir
            )
            return []

        # Get all file names in the directory
        available_files = []
        for file in self.dropbox_dir.iterdir():
            if file.is_file() and file.name != "file_metadata.json":
                available_files.append(file.name)
        logger.info(
            "%d available file(s) detected in dropbox: session=%s, path='%s'",
            len(available_files),
            self.session_name,
            self.dropbox_dir
        )
        return available_files

    def store_file_in_dropbox(self, content: str, file_path: Path) -> Path | None:
        """Store file contents into the dropbox."""
        # Prevent duplicated filename in the metadata
        filename = file_path.name
        counter = 1
        while filename in self.file_metadata:
            filename = f"{file_path.stem}({counter}){file_path.suffix}"
            counter += 1

        try:
            self.dropbox_dir.mkdir(parents=True, exist_ok=True)
            path = self.dropbox_dir / filename
            path.write_text(content, encoding="utf-8")
            logger.info(
                "Uploaded file to dropbox: file=%s, session=%s, path='%s'",
                filename,
                self.session_name,
                self.dropbox_dir
            )
            return path

        except Exception as e:
            logger.error(
                "Failed to upload file to dropbox: file=%s, session=%s, path='%s', error=%s",
                filename,
                self.session_name,
                self.dropbox_dir,
                e
            )

    def clear_session_dropbox(self):
        """Remove session dropbox directory when cleaning history."""
        if self.dropbox_dir.exists():
            for child in self.dropbox_dir.iterdir():
                if child.is_file():
                    child.unlink()

            self.dropbox_dir.rmdir()
            logger.info(
                "Cleared all files in dropbox: session=%s, path='%s'",
                self.session_name,
                self.dropbox_dir
            )

        logger.error(
            "Failed to clear dropbox. Dropbox directory not found: session=%s, path='%s'",
            self.session_name,
            self.dropbox_dir
        )

    # ================================================
    # File content
    # ================================================

    def load_file_content(self, file_path: Path) -> tuple[dict[str, str], bool]:
        """Return file content as a string using mapped parsers."""
        ext = file_path.suffix.lower()

        if not file_path.exists():
            logger.warning(
                "Failed to load file content. File does not exists: path='%s'",
                file_path
            )
            return {}, False

        # Route to correct parser, fallback to plain text if unknown
        logger.info("Routing to correct parser to extract file content")
        logger.info(
            "User selected Docling as default parser method. Loading converter"
        ) if config.DOCLING_DEFAULT else None
        file_data = {}
        parser = self.parsers.formats.get(ext, self.parsers._read_txt)
        filename = file_path.name
        content = parser(file_path)
        file_data[filename] = content

        return file_data, True

    def _load_contents_from_file_list(self, file_paths: list[Path]) -> dict[str, str]:
        """From a list of filenames, returns filename and contents as dictionary."""
        files_data = {}

        for path in file_paths:
            file_data, path_exists = self.load_file_content(path)
            files_data.update(file_data) if path_exists else None
            if not file_data:
                logger.warning(
                    "Failed to read file. No valid content found: path='%s'",
                    path
                )
                continue

        return files_data

    # ==================================================
    # Model integration (metadata)
    # ==================================================

    def _generate_short_summary(self, filename: str, content: str) -> tuple[str, int, int]:
        """Model generates a short summary about the file content."""
        formatted_file_content = (
            f"# Filename: {filename}\n"
            f"File content:\n"
            f"{content}"
        )

        logger.info(
            "Model generating short summary for file: model=%s, filename=%s",
            self.model,
            filename
        )
        summary, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.gen_summary_prompt,
            prompt=formatted_file_content
        )
        logger.info("Generated short summary for file: filename=%s", filename)
        return summary, p_tkns, o_tkns

    def add_metadata_and_summary(self, files_data: dict[str, str]):
        """Add summary and metadata to files."""
        for filename, content in files_data.items():
            summary, p_tkns, o_tkns = self._generate_short_summary(filename, content)
            self._add_file_metadata(filename, summary)

    # ==================================================
    # Model choose relevant context
    # ==================================================

    def require_file_or_not(self, context: list[dict], prompt: str) -> tuple[bool, int, int]:
        """Query model to decide if file context is needed, return 'True' or 'False' only."""
        logger.info(
            "Model deciding if query requires extra context in dropbox: model=%s",
            self.model
        )
        output, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.file_or_not_prompt,
            prompt=prompt,
            context=context
        )
        if 'true' in output.lower():
            logger.info("Query require extra context")
            return True, p_tkns, o_tkns
        else:
            logger.info("Query does not require extra context")
            return False, p_tkns, o_tkns

    def model_request_files(self, context: list[dict], available_files_prompt: str) -> tuple[list[Path], list[Path], int, int]:
        """
        Ask model to choose from available in current session to get relevant context,
        formats model output to get clean lists of data.
        """
        # Model selects from a list of files
        list_of_files, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
            model=self.model,
            system_prompt=self.file_list_prompt,
            prompt=available_files_prompt,
            context=context
        )
        logger.info(
            "Model decided to retrieve %d file(s): model=%s",
            len(list_of_files),
            self.model
        )

        found_file_paths        = []
        not_found_file_paths    = []

        # Reformats model generated string to a clean list
        parsed_names = []
        for token in re.split(r"[,\n]", list_of_files):
            name = re.sub(r"[*'\"\'\-]", "", token).strip()
            if name:
                parsed_names.append(name)

        # Segregate file paths into found and not found
        for file in parsed_names:
            file_path = self.dropbox_dir / file
            if file_path.exists():
                found_file_paths.append(file)
            else:
                not_found_file_paths.append(file)

        logger.warning("%d file(s) not found in session dropbox", len(not_found_file_paths))
        return found_file_paths, not_found_file_paths, p_tkns, o_tkns

    def _structured_file_string(self, filename: str, summary: str) -> str:
        """Format contents for model to read."""
        string = (
            f"# Filename: {filename}\n"
            f"File summary: {summary}\n\n"
            f"---\n\n"
        )
        return string

    def _list_available_file_with_summary(self) -> str:
        """
        Steps:
        1. Add summary and metadata that are not in 'file_metadata.json'.
        2. List all filenames in dropbox.
        3. Format filenames and summary to a string for model to read.
        """
        # Add summary and metadata if not exists
        logger.info(
            "Verifying if 'file_metadata.json' is up to date: session=%s, path='%s'",
            self.session_name,
            self.metadata_path
        )
        files_without_summary = self._files_not_in_file_metadata()
        file_paths = [self.dropbox_dir / file for file in files_without_summary]
        files_data = {}
        for file in file_paths:
            filename = file.name
            content = _read_file(file)
            files_data[filename] = content
        self.add_metadata_and_summary(files_data)
        logger.info("Updated %d file(s) in 'file_metadata.json'", len(files_data))

        # Re-sync metadata
        if hasattr(self, "_load_file_metadata"):
            self.file_metadata = self._load_file_metadata() or {}

        # Structure filenames and summary into string
        filenames = self.list_available_files()
        blocks = []
        for file in filenames:
            metadata = self.file_metadata.get(file, {})
            summary = metadata.get("summary", "No summary available")
            blocks.append(self._structured_file_string(file, summary))
        entries = "\n\n".join(blocks)
        logger.info("Retrieved all available file(s) and summary(s)")
        return f"# All available files\n\n{entries}"

    # ==================================================
    # Auto function
    # ==================================================

    def toggle_auto_read_dropbox(
            self,
            messages: list[dict],
            prompt: str,
            enable_attachments: bool,
            enable_auto_read_dropbox: bool,
            memory_entries: str | None = None,
            attach_file_data: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], list[Path], bool]:
        """
        Auto fetches previous file contents ability, return relevant files if its toggled on.

        Model decides from {memory_entries} + {prompt} + {attached_file} --> {file_content} from dropbox
        """
        if (
            enable_auto_read_dropbox == False
            or self.tokens.model_max_tokens <= config.AUTO_READ_DROPBOX_TOKENS
        ):
            return {}, [], False

        if _is_dir_empty(self.dropbox_dir):
            logger.info(
                "No extra file context retrieved. Dropbox is empty: session=%s, path='%s'",
                self.session_name,
                self.dropbox_dir
            )
            return {}, [], False

        # Prepare context for model to decide require file or not
        memory_sect = (
            f"# Retrieved memory entry(s)\n\n"
            f"{memory_entries}\n\n"
            f"---\n\n"
        ) if memory_entries else ""
        prompt_sect = (
            f"# User prompt\n\n"
            f"{prompt}\n\n"
            f"---\n\n"
        )
        attach_sect = "# Attachments\n\n".join(
            f"Filename: {filename}\n"
            f"Content: {content}\n\n"
            for filename, content in attach_file_data.items()
        ).join("---\n\n") if attach_file_data else ""
        cmbind_prompt = memory_sect + prompt_sect + attach_sect

        # Model selects filename(s)
        require_file, _, _ = self.require_file_or_not(messages, cmbind_prompt)
        if require_file:
            available_files_prompt  = self._list_available_file_with_summary()
            found_files, _, _, _    = self.model_request_files(messages, available_files_prompt)
            found_file_paths        = [self.dropbox_dir / file for file in found_files] # Turn found_files to paths
            return self._load_contents_from_file_list(found_file_paths), found_files, True
        return {}, [], False

    # ==================================================
    # Manual read files with given paths
    # ==================================================

    def read_files_with_context_prompt(
            self,
            context: list[dict],
            file_paths: list[Path] | None,
    ) -> dict[str, str] | None:
        """Return all file's name and content from a list of filenames."""
        file_data = {}

        if not file_paths:
            return None

        files_data = {}
        for path in file_paths:
            filename = path.name
            file_data, path_exists = self.load_file_content(path)
            files_data.update(file_data) if path_exists else None
        return files_data
