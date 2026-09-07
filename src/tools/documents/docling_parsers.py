from pathlib import Path

from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


class DoclingParsers:
    def __init__(
        self,
    ):
        # Map formats to their parsing methods
        self.formats = {
            # LIBREOFFICE
            ".odt": self._read_odt,
            ".ods": self._read_ods,
            ".odp": self._read_odp,

            # MICROSOFT OFFICE
            ".docx": self._read_docx,
            ".xlsx": self._read_xlsx,
            ".pptx": self._read_pptx,

            # DATA & CONFIGURATION FORMATS
            ".csv": self._read_csv,
            ".xml": self._read_xml,
            
            # TEXT DOCUMENTS
            ".pdf": self._read_pdf,
            ".epub": self._read_epub,
        }

        self.converter = None


    # =======================================================
    # LAZY IMPORT DOCLING
    # =======================================================

    def _get_converter(self):
        if self.converter is None:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
        return self.converter


    # =======================================================
    # DEFAULT DOCLING CONVERTER
    # =======================================================

    def _docling(self, path: Path) -> str:
        """Convert any file to markdown via docling."""
        result = self.converter.convert(path)
        return result.document.export_to_markdown()


    # =======================================================
    # LIBREOFFICE
    # =======================================================

    def _read_odt(self, path: Path) -> str:
        app_log.info(f"Docling converting ODT: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    def _read_ods(self, path: Path) -> str:
        app_log.info(f"Docling converting ODS: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    def _read_odp(self, path: Path) -> str:
        app_log.info(f"Docling converting ODP: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    # =======================================================
    # MICROSOFT OFFICE
    # =======================================================

    def _read_docx(self, path: Path) -> str:
        app_log.info(f"Docling converting DOCX: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    def _read_xlsx(self, path: Path) -> str:
        app_log.info(f"Docling converting XLSX: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    def _read_pptx(self, path: Path) -> str:
        app_log.info(f"Docling converting PPTX: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    # =======================================================
    # DATA & CONFIGURATION FORMATS
    # =======================================================

    def _read_csv(self, path: Path) -> str:
        """Reads csv as plain text files."""
        app_log.info(f"Docling converting CSV: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    def _read_xml(self, path: Path) -> str:
        app_log.info(f"Docling converting XML: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    # =======================================================
    # TEXT DOCUMENTS
    # =======================================================

    def _read_pdf(self, path: Path) -> str:
        app_log.info(f"Docling converting PDF: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)


    def _read_epub(self, path: Path) -> str:
        app_log.info(f"Docling converting EPUB: {path}")
        print(f"Docling converting content: {path}")
        return self._docling(path)
