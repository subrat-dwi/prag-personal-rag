from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from llm.utils import extract_main_content

def get_llm():
    return init_chat_model(
        model= "qwen2.5:3b-instruct",
        model_provider="ollama"
    )

#-----------------main OCR cleaning function------------------------
def clean_ocr_text(raw_ocr_text: str) -> str:
    """
    Use a language model to clean and correct OCR-extracted text.

    Args:
        ocr_text: Raw text extracted from OCR.

    Returns:
        Cleaned and corrected text.
    """
    llm = get_llm()
    
    system_prompt = SystemMessage(
        f"""You are a document text cleaner. 
You will receive raw OCR output from a scanned Indian document (like Aadhar card, certificate, or result sheet).
The text may have garbled characters, broken words, wrong spacing, and mixed Hindi/English noise.

Your job:
- Reconstruct the readable English information only
- Fix broken words and spacing
- Extract key fields if identifiable (Name, DOB, Address, ID numbers etc.)
- Ignore decorative characters, watermarks, and unreadable fragments
- Output clean plain text only, no explanation

RAW OCR TEXT:
{raw_ocr_text}
"""
    )
    
    response = llm.invoke([system_prompt])
    
    return extract_main_content(response)