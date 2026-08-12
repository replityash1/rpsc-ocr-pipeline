"""
Deterministic pattern helpers used by structure_questions.py.

Nothing in this file "guesses" content — it only classifies and splits text
that PaddleOCR-VL already extracted. That is a deliberate design choice: the
one component allowed to assemble question objects must never hallucinate,
so it is kept to regex + unicode-range checks rather than a generative model.
"""
import re
import unicodedata

DEVANAGARI_START, DEVANAGARI_END = 0x0900, 0x097F

# Devanagari digits 0-9 -> Arabic digits, so "१२३" and "123" both parse as 123
DEVANAGARI_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

# Two tiers, on purpose:
#  - EXPLICIT patterns ("Q.12", "प्रश्न 12") are unambiguous — nothing else in
#    an exam paper is prefixed this way, so these ALWAYS start a new question
#    and are allowed to advance the running question-number sequence.
#  - BARE patterns ("12." with nothing else) are ambiguous — a statement list
#    inside a question body ("1. Statement one  2. Statement two") uses the
#    exact same shape. A bare number is therefore only trusted as a question
#    boundary when it repeats the CURRENT question's number (the common
#    RPSC layout where the Hindi twin of "Q.12" is printed as a bare "12."),
#    never to introduce a brand-new number on its own. See
#    structure_questions.py's stateful splitter for how this is enforced.
EXPLICIT_NUMBER_PATTERNS = [
    re.compile(r'^\s*Q\.?\s*(\d{1,3})[\.\)]', re.IGNORECASE),
    re.compile(r'^\s*(?:प्रश्न|प्र\.?)\s*[-:.]?\s*([०-९0-9]{1,3})\s*[\.\)]?'),
]

BARE_NUMBER_PATTERNS = [
    re.compile(r'^\s*(\d{1,3})[\.\)]\s'),
    re.compile(r'^\s*([०-९]{1,3})[\.\)]\s'),
]

QUESTION_NUMBER_PATTERNS = EXPLICIT_NUMBER_PATTERNS + BARE_NUMBER_PATTERNS

# Letter-based markers are unambiguous within a question body (nothing else
# looks like "(A)"/"(B)"). Numeric "(1)(2)(3)(4)" markers are NOT
# unambiguous — a statement list inside the question ("1. Statement one")
# uses the identical shape. structure_questions.py only enables the numeric
# matcher for a block when no letter-based options were found anywhere in
# it, so a paper using numeric option codes still works, but a lettered
# question with an embedded numbered statement list doesn't have its
# statements misread as options.
LETTER_OPTION_PATTERNS = [
    re.compile(r'^\s*\(?([A-Da-d])\)?[\.\)]\s+'),
]

NUMERIC_OPTION_PATTERNS = [
    re.compile(r'^\s*\(?([1-4])\)?[\.\)]\s+'),
]

ROMAN_OPTION_PATTERNS = [
    re.compile(r'^\s*\(?([iI]{1,3}[vV]?)\)?[\.\)]\s+'),
]

OPTION_LABEL_PATTERNS = LETTER_OPTION_PATTERNS + NUMERIC_OPTION_PATTERNS + ROMAN_OPTION_PATTERNS


def _first_match(patterns, text: str):
    for pat in patterns:
        m = pat.match(text)
        if m:
            raw = m.group(0)
            digits = re.search(r'[०-९0-9]{1,3}', raw)
            if digits:
                normalized = digits.group(0).translate(DEVANAGARI_DIGIT_MAP)
                try:
                    return int(normalized), raw
                except ValueError:
                    return None, raw
            return None, raw
    return None, ""


def match_explicit_number(text: str):
    """Match only the unambiguous 'Q.N' / 'प्रश्न N' markers."""
    return _first_match(EXPLICIT_NUMBER_PATTERNS, text.strip())


def match_bare_number(text: str):
    """Match a bare 'N.' style marker (ambiguous — could be a statement list
    item rather than a question number; caller must apply sequence logic)."""
    return _first_match(BARE_NUMBER_PATTERNS, text.strip())


def normalize_number_token(text: str):
    """Extract a leading question number as an int, handling both Devanagari
    and Arabic numerals, explicit or bare. Returns (number:int|None,
    raw_matched_text:str). Convenience wrapper — prefer match_explicit_number
    / match_bare_number directly when boundary logic needs to tell them apart."""
    return _first_match(QUESTION_NUMBER_PATTERNS, text.strip())


def _match_against(patterns, line: str):
    for pat in patterns:
        m = pat.match(line)
        if m:
            return m.group(1).upper(), line[m.end():].strip()
    return None, line


def has_letter_option(line: str) -> bool:
    label, _ = _match_against(LETTER_OPTION_PATTERNS, line)
    return label is not None


def match_option_label(line: str, allow_numeric: bool = True):
    """Return (label, remainder_text) if the line starts with an option
    marker, else (None, line). `allow_numeric` should be False for blocks
    that already contain letter-based options, so an embedded numbered
    statement list isn't misread as options — see module docstring."""
    label, remainder = _match_against(LETTER_OPTION_PATTERNS, line)
    if label:
        return label, remainder
    label, remainder = _match_against(ROMAN_OPTION_PATTERNS, line)
    if label:
        return label, remainder
    if allow_numeric:
        label, remainder = _match_against(NUMERIC_OPTION_PATTERNS, line)
        if label:
            return label, remainder
    return None, line


def devanagari_ratio(text: str) -> float:
    """Fraction of alphabetic characters in `text` that fall in the
    Devanagari unicode block. Used to classify a line as Hindi vs English."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)
    return dev / len(letters)


def classify_line_language(text: str, threshold: float = 0.25) -> str:
    ratio = devanagari_ratio(text)
    if ratio >= threshold:
        return "hi"
    if ratio == 0.0 and any(c.isalpha() for c in text):
        return "en"
    if ratio > 0:
        return "hi"
    return "unknown"


def strip_number_prefix(text: str) -> str:
    """Remove a leading question-number token from a block, keeping the rest
    verbatim (no rewording)."""
    for pat in QUESTION_NUMBER_PATTERNS:
        m = pat.match(text)
        if m:
            return text[m.end():].strip()
    return text.strip()
