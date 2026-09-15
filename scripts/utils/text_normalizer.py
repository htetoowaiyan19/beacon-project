"""Myanmar Unicode text normalization and cleaning utilities."""

from __future__ import annotations

import re
import unicodedata


# Regex to strip synthetic noise like #123, #4618, #001
SYNTHETIC_HASH_REGEX = re.compile(r"\s*#\d+\b")

# Regex to strip multiple spaces/newlines
MULTI_SPACE_REGEX = re.compile(r"[ \t]+")
MULTI_NEWLINE_REGEX = re.compile(r"\n{3,}")


def is_myanmar_text(text: str) -> bool:
    """Check if the text contains Myanmar Unicode characters (\u1000-\u109F, \uAA60-\uAA7F)."""
    return any(
        ("\u1000" <= char <= "\u109F") or ("\uAA60" <= char <= "\uAA7F")
        for char in text
    )


def normalize_myanmar_text(text: str) -> str:
    """Clean and normalize Myanmar and multilingual instruction text.
    
    1. Removes synthetic ID tags like '#4618'.
    2. Applies Unicode NFC normalization.
    3. Trims whitespace and redundant blank lines.
    4. Normalizes Burmese punctuation and quotes.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove synthetic hashtag noise
    text = SYNTHETIC_HASH_REGEX.sub("", text)

    # 3. Clean up weird quote combinations often left after removing tags like 'Hello #123'
    text = re.sub(r"'\s*'\s*", "", text)
    text = re.sub(r'""', '"', text)

    # 4. Normalize spacing
    lines = [MULTI_SPACE_REGEX.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    text = MULTI_NEWLINE_REGEX.sub("\n\n", text)

    # 5. Fix common Burmese tone/diacritic spacing artifacts (e.g. space before ်, ့, း, ၊, ။)
    text = re.sub(r"\s+([့်း၊။])", r"\1", text)

    return text.strip()


def standardize_system_prompt(content: str) -> str:
    """Standardize variations of generic system prompts to a polite, fluent Burmese persona."""
    content = content.strip()
    generic_english = {
        "",
        "You are a helpful assistant.",
        "You are a helpful assistant",
        "You are a helpful AI assistant.",
    }
    if content in generic_english:
        return "သင်သည် အကူအညီပေးသော မြန်မာ AI လက်ထောက်တစ်ဦး ဖြစ်ပါသည်။"

    # Normalize if already Burmese
    return normalize_myanmar_text(content)
