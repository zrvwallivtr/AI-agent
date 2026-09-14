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

    def _docling_convert_to_md(self, path: Path) -> tuple[str, str] | None:
        """Convert any file to markdown via docling."""
        app_log.debug(
            "Docling converting %s file '%s' to MD format", path.suffix.lstrip(".").upper(), path
        )
        result = self.converter.convert(path)

        if not result:
            app_log.warning(
                "Docling failed to convert %s file '%s' to MD format",
                path.suffix.lstrip(".").upper,
                path
            )
            return

        app_log.info("%s file '%s' converted to MD format", path.suffix.lstrip(".").upper(), path)
        return result.document.export_to_markdown(), "md"


    # =======================================================
    # LIBREOFFICE
    # =======================================================

    def _read_odt(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    def _read_ods(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    def _read_odp(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    # =======================================================
    # MICROSOFT OFFICE
    # =======================================================

    def _read_docx(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    def _read_xlsx(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    def _read_pptx(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    # =======================================================
    # DATA & CONFIGURATION FORMATS
    # =======================================================

    def _read_csv(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    def _read_xml(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    # =======================================================
    # TEXT DOCUMENTS
    # =======================================================

    def _read_pdf(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)


    def _read_epub(self, path: Path) -> tuple[str, str] | None:
        return self._docling_convert_to_md(path)
