"""
Shared data structures for the extraction pipeline.
Keeping this in one place means every stage (OCR, structuring, validation,
merge) writes/reads the exact same JSON shape.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Option:
    label: str                     # "A", "1", "i", etc. — verbatim as printed
    hi: str = ""
    en: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Question:
    question_no: Optional[int]     # None if the number itself could not be read
    raw_number_text: str           # exactly what OCR saw for the number token
    page: int
    question_hi: str = ""
    question_en: str = ""
    options: list = field(default_factory=list)     # list[Option]
    has_formula: bool = False
    has_table: bool = False
    figures: list = field(default_factory=list)      # relative paths
    ocr_confidence: Optional[float] = None
    cross_engine_agreement: Optional[float] = None
    needs_review: bool = False
    review_reasons: list = field(default_factory=list)
    source_blocks: list = field(default_factory=list)  # raw OCR block ids, for audit

    def to_dict(self):
        d = asdict(self)
        d["options"] = [o.to_dict() if isinstance(o, Option) else o for o in self.options]
        return d


@dataclass
class PageResult:
    page: int
    markdown: str
    blocks: list                 # list of layout blocks from PaddleOCR-VL
    mean_confidence: Optional[float]
    tesseract_text: str = ""
    figures: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
