# utility functions for LLM interations
def extract_main_content(value) -> str:
    """Extract the main human-readable text from AI/Human messages or content blocks."""
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(extract_main_content(item) for item in value)

    if isinstance(value, dict):
        if "content" in value:
            return extract_main_content(value.get("content"))
        if "text" in value:
            return extract_main_content(value.get("text"))
        if "delta" in value:
            return extract_main_content(value.get("delta"))
        return ""

    content = getattr(value, "content", None)
    if content is not None:
        return extract_main_content(content)

    text = getattr(value, "text", None)
    if text is not None:
        return extract_main_content(text)

    return ""
