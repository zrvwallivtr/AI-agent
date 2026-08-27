from pathlib import Path


from src.config.files_and_directories import UPLOAD_DIR
from src.config.documents import DOCLING_DEFAULT
from src.tools.documents.basic_parsers import BasicParsers
from src.tools.documents.docling_parsers import DoclingParsers


class PathTraversalError(Exception):
    pass


class DocumentReader:
    def __init__(self):
        self.bs_prsrs       = BasicParsers()
        self.docling_prsrs  = DoclingParsers()


    def _validate_path(self, path: Path) -> Path:
        """Resolve 'path' to its real, absolute form."""
        resolved = path.resolve()

        try:
            resolved.relative_to(UPLOAD_DIR)

        except ValueError:
            raise PathTraversalError(
                f"Path '{path}' resolves to '{resolved}', which is outside "
                f"the allowed directory '{UPLOAD_DIR}'"
            )

        return resolved


    def read_document(self, path: Path) -> str:
        """Return file content as a string using mapped parsers."""
        safe_path = self._validate_path(path)

        if not safe_path.exists():
            raise FileNotFoundError(f"'{safe_path}' does not exist")
        if not safe_path.is_file():
            raise ValueError(f"'{safe_path}' is not a regular file")

        # Match with the correct parser
        ext = safe_path.suffix.lower()

        if DOCLING_DEFAULT:
            parser = self.docling_prsrs.formats.get(ext, self.bs_prsrs.read_txt)
        else:
            parser = self.bs_prsrs.formats.get(ext, self.bs_prsrs.read_txt)

        return parser(safe_path)
