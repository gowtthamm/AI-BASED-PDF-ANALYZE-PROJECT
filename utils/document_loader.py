import os
import shutil
from typing import List

# 1. Load environment variables immediately
from dotenv import load_dotenv
load_dotenv()

import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# A page is considered "text-empty" (likely scanned/image-based) if it has
# fewer than this many non-whitespace characters of extracted text.
MIN_TEXT_LENGTH_THRESHOLD = 20

# Environment variable keys
TESSERACT_CMD_ENV = "TESSERACT_CMD"
POPPLER_PATH_ENV = "POPPLER_PATH"

# 2. Explicitly configure Tesseract path for Windows if provided in .env
tesseract_env_path = os.getenv(TESSERACT_CMD_ENV)
if tesseract_env_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_env_path
else:
    # Default fallback guess if .env is blank
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    """
    Persist a Streamlit `UploadedFile` object to disk so that LangChain's
    PyPDFLoader (which expects a file path) can read it.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def _ocr_available() -> bool:
    """
    Check whether OCR dependencies (pytesseract, pdf2image, Tesseract
    binary, Poppler) appear to be available.
    """
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_path  # noqa: F401
    except ImportError:
        return False

    # Check if tesseract path exists from env or system PATH
    if tesseract_env_path and os.path.exists(tesseract_env_path):
        return True

    return shutil.which("tesseract") is not None


def _ocr_pdf(file_path: str, page_numbers: List[int]) -> dict:
    """
    Run OCR on specific (1-indexed) pages of a PDF and return their text.
    """
    from pdf2image import convert_from_path

    poppler_path = os.getenv(POPPLER_PATH_ENV) or r"C:\poppler\Library\bin"

    results = {}
    for page_num in page_numbers:
        try:
            # convert_from_path uses 1-indexed first_page/last_page.
            images = convert_from_path(
                file_path,
                first_page=page_num,
                last_page=page_num,
                poppler_path=poppler_path if os.path.exists(poppler_path) else None,
                dpi=300,
            )
            if images:
                results[page_num] = pytesseract.image_to_string(images[0])
            else:
                results[page_num] = ""
        except Exception as e:
            print(f"Error during OCR extraction on page {page_num}: {e}")
            results[page_num] = ""

    return results


def load_pdf(file_path: str, use_ocr_fallback: bool = True) -> List[Document]:
    """
    Load a single PDF file and return a list of LangChain Document objects,
    one per page, each carrying metadata: {"source": filename, "page": n}.

    If a page has almost no extractable text (a strong sign it's a scanned
    image page), and OCR dependencies are available, the page is
    automatically re-processed with OCR.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    filename = os.path.basename(file_path)

    # Normalize metadata to be human friendly (1-indexed).
    for page in pages:
        page.metadata["source"] = filename
        if "page" in page.metadata:
            page.metadata["page"] = page.metadata["page"] + 1

    # Identify pages that look like scanned images (little/no real text).
    low_text_pages = [
        page.metadata.get("page", idx + 1)
        for idx, page in enumerate(pages)
        if len(page.page_content.strip()) < MIN_TEXT_LENGTH_THRESHOLD
    ]

    if low_text_pages and use_ocr_fallback:
        if _ocr_available():
            ocr_results = _ocr_pdf(file_path, low_text_pages)
            for page in pages:
                page_num = page.metadata.get("page")
                if page_num in ocr_results:
                    ocr_text = ocr_results[page_num].strip()
                    if ocr_text:
                        page.page_content = ocr_text + "\n[Extracted via OCR Fallback]"
                        page.metadata["ocr_used"] = True
        else:
            # OCR binaries not found — flag via metadata so app.py can warn the user
            for page in pages:
                if page.metadata.get("page") in low_text_pages:
                    page.metadata["ocr_needed"] = True

    return pages


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Load multiple PDF files and combine all extracted pages into a single
    list of Document objects.
    """
    all_documents: List[Document] = []
    for path in file_paths:
        try:
            docs = load_pdf(path)
            all_documents.extend(docs)
        except Exception as exc:
            raise RuntimeError(f"Failed to load PDF '{path}': {exc}") from exc

    return all_documents


def count_pages_needing_ocr(documents: List[Document]) -> int:
    """
    Count how many pages were flagged as needing OCR but couldn't be
    processed because OCR dependencies weren't available.
    """
    return sum(1 for doc in documents if doc.metadata.get("ocr_needed"))