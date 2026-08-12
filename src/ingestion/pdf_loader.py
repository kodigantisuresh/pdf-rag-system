"""
    Why Path?:
        Instead of manipulating paths as raw strings: "data/raw/sample.pdf"
        we use: Path("data/raw/sample.pdf")
        "pathlib.Path" provides cleaner cross-platform path handling.
"""

"""
    The load() Method: "def load(self):"
    Conceptually:
        PDFLoader
            │
            └── load()
                    │
                    ├── open PDF
                    ├── extract pages
                    ├── create metadata
                    └── return documents
"""


from pathlib import Path
import pymupdf  # PyMuPDF library for PDF processing

from src.utils.logger import logger


class PDFLoader:

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def load(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {self.pdf_path.suffix}"
            )

        logger.info(f"Loading PDF: {self.pdf_path}")

        try:
            document = pymupdf.open(self.pdf_path)

            pages = []

            for page_number, page in enumerate(document):

                text = page.get_text()

                pages.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": self.pdf_path.name,
                            "page": page_number + 1,
                        },
                    }
                )

            document.close()

            logger.info(
                f"Successfully extracted {len(pages)} pages "
                f"from {self.pdf_path.name}"
            )

            return pages
        
        except Exception as exc:

            logger.exception(
                f"Failed to process PDF: {self.pdf_path}"
            )

            raise RuntimeError(
                f"Failed to process PDF: {self.pdf_path}"
            ) from exc

        """
            # Suppose PyMuPDF throws: FileDataError
            # We want our application to expose a meaningful application-level error: Failed to process PDF
            # but we don't want to lose the original exception.
                Therefore:

                    raise RuntimeError(...) from exc
                    preserves the original error as the cause.
                    This becomes very useful during debugging.
        """