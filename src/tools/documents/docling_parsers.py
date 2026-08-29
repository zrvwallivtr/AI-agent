from pathlib import Path
from docling.document_converter import DocumentConverter


from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


class DoclingParsers:
    def __init__(
        self,
    ):
        # Map formats to their parsing methods
        self.formats = {
            # Data & configuration formats
            ".csv": self._read_csv,
            ".xlsx": self._read_xlsx,
            ".xml": self._read_xml,
            
            # Text documents
            ".pdf": self._read_pdf,
            ".docx": self._read_docx,
            ".epub": self._read_epub,
        }

        self.converter = DocumentConverter()

    # =======================================================
    # DOCLING
    # =======================================================

    def _docling(self, path: Path) -> str:
        """Convert any file to markdown via docling."""
        app_log.info(f"Converting CSV file: {path}")
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
