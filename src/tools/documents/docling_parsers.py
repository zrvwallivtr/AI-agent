from pathlib import Path
from docling.document_converter import DocumentConverter


from src import config
from src.logger import get_logger


log = get_logger(__name__)


class DoclingParsers:
    def __init__(
        self,
    ):
        # Map formats to their parsing methods
        self.formats = {
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

        self.converter = DocumentConverter()

    # =======================================================
    # DOCLING
    # =======================================================

    def _docling(self, path: Path) -> str:
        """Convert any file to markdown via docling."""
        log.info(f"Converting CSV file: {path}")
        print(f"Converting document: {path}")
        result = self.converter.convert(path)
        return result.document.export_to_markdown()


    # =======================================================
    # DATA & CONFIGURATION FORMATS
    # =======================================================

    def _read_csv(self, path: Path) -> str:
        """Reads csv as plain text files."""
        return self._docling(path)


    def _read_xlsx(self, path: Path) -> str:
        return self._docling(path)


    def _read_yaml(self, path: Path) -> str:
        return self._docling(path)


    def _read_yml(self, path: Path) -> str:
        return self._docling(path)


    def _read_toml(self, path: Path) -> str:
        return self._docling(path)


    def _read_xml(self, path: Path) -> str:
        return self._docling(path)


    # =======================================================
    # TEXT DOCUMENTS
    # =======================================================

    def _read_pdf(self, path: Path) -> str:
        return self._docling(path)


    def _read_docx(self, path: Path) -> str:
        return self._docling(path)


    def _read_epub(self, path: Path) -> str:
        return self._docling(path)


    # =======================================================
    # PROGRAMMING
    # =======================================================

    def _read_code(self, path: Path) -> str:
        return self._docling(path)
