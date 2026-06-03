from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pathlib import Path
from logging import Logger
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from llm.ocr_cleaner import clean_ocr_text

logger = Logger("__name__")

#-----------------main PDF parsing function------------------------
def parse_pdf(path: str) -> str:
    """
    Extract text from a PDF file.
    Falls back to OCR for scanned/image-based pages.
    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or is encrypted.
        RuntimeError: If no text could be extracted from the document.
    """

    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File Not Found: {path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")
    
    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as e:
        raise ValueError(f"Could not read PDF (possibly corrupt)")
    
    if reader.is_encrypted:
        raise ValueError(f"File is encrypted, cannot be read")
    
    pages_text = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            logger.warning("Page %d of %s has no text. Trying with OCR...", page_number + 1, pdf_path.name)

            # If the page has no extractable text, it may be a scanned image. Use OCR as a fallback.
            ocr_text = _ocr_pdf_page(str(pdf_path), page_number)
            if ocr_text:
                pages_text.append(clean_ocr_text(ocr_text))
            else:
                logger.warning("OCR also failed to extract text from page %d of '%s'", page_number + 1, pdf_path.name)
        
        else:
            pages_text.append(text)

    full_text = "\n".join(pages_text).strip()

    # If after processing all pages we still have no text, raise an error.
    if not full_text:
        raise RuntimeError(
            f"No text could be extracted from '{path}'. "
        )
        
    return full_text

#-----------------helper functions------------------------
def _ocr_pdf_page(pdf_path: str, page_num: int) -> str:
    """
    Render a single PDF page as an image and run OCR on it.

    Args:
        pdf_path: Path to the PDF.
        page_num: Zero-indexed page number.

    Returns:
        OCR-extracted text, or empty string if nothing found.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # DPI scaling 
    # 2x = 144 DPI - enough for most documents
    # increase to 3 (216 DPI) if getting bad results
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)

    img = Image.frombytes("RGB", tuple([pix.width, pix.height]), pix.samples)
    img = img.convert("L")  # grayscale improves OCR accuracy

    text = pytesseract.image_to_string(img, lang="eng").strip()
    doc.close()

    return text
