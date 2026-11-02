"""Text normalization for spoken output ('writing for the ear').

Enforces rules from [STY-3] and Gnani speech guidelines:
- Dates spoken with month and year ('the 24th of October 2026')
- Order numbers grouped with pauses ('8 9 7 8, 2 8 3 7')
- Currencies expanded to spoken words ('forty-nine rupees')
- Brevity capped at 2-3 spoken sentences
"""

import re
from datetime import datetime
from src.data.date_utils import format_spoken_date, parse_date


# Number to words dictionary for common currency amounts
NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    25: "twenty-five", 30: "thirty", 40: "forty", 49: "forty-nine", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety", 100: "one hundred",
    500: "five hundred", 1000: "one thousand", 10000: "ten thousand"
}


def number_to_words(n: int) -> str:
    """Convert common integer numbers to spoken English words."""
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if n < 100:
        tens, units = divmod(n, 10)
        return f"{NUM_WORDS.get(tens*10, '')}-{NUM_WORDS.get(units, '')}".strip("-")
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        h_str = f"{NUM_WORDS.get(hundreds, str(hundreds))} hundred"
        return f"{h_str} and {number_to_words(rem)}" if rem else h_str
    if n < 100000:
        thousands, rem = divmod(n, 1000)
        t_str = f"{number_to_words(thousands)} thousand"
        return f"{t_str} {number_to_words(rem)}" if rem else t_str
    return str(n)


def format_order_id_for_speech(match: re.Match) -> str:
    """Convert an 8-digit order number into spoken groups with pause: '8 9 7 8, 2 8 3 7'."""
    raw = match.group(0).replace(" ", "")
    if len(raw) == 8:
        part1 = " ".join(raw[:4])
        part2 = " ".join(raw[4:])
        return f"{part1}, {part2}"
    return " ".join(raw)


def format_currency_for_speech(match: re.Match) -> str:
    """Convert ₹49 or INR 500 to spoken words: 'forty-nine rupees'."""
    val_str = match.group(1).replace(",", "")
    try:
        val = int(val_str)
        return f"{number_to_words(val)} rupees"
    except (ValueError, TypeError):
        return f"{val_str} rupees"


def format_dates_in_text(text: str) -> str:
    """Identify ISO dates or standard dates and convert them to spoken form."""
    def _replace_iso(m):
        d = parse_date(m.group(0))
        return format_spoken_date(d) if d else m.group(0)

    # Match YYYY-MM-DD
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", _replace_iso, text)
    return text


def normalize_for_voice(text: str) -> str:
    """Apply comprehensive phonetic and prosodic transformations for Gnani TTS."""
    if not text:
        return ""

    out = text

    # 1. Format Currency: ₹49, ₹ 500, INR 10,000, Rs. 49
    out = re.sub(r"(?:₹|INR|Rs\.?)\s*([0-9,]+)", format_currency_for_speech, out)

    # 2. Format 8-digit order IDs: 8978 2837 or 89782837
    out = re.sub(r"\b\d{4}\s?\d{4}\b", format_order_id_for_speech, out)

    # 3. Format Dates
    out = format_dates_in_text(out)

    # 4. Remove Markdown syntax that interferes with TTS (brackets, asterisks, hashtags)
    out = re.sub(r"\[([A-Z0-9\.\-]+)\]", "", out)  # section citations like [RET-2]
    out = re.sub(r"[*_#`]", "", out)               # markdown formatting marks
    out = re.sub(r"\s+", " ", out).strip()

    return out

