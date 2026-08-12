#!/usr/bin/env python3
"""
Stage 3 (job: structure — single job, runs after all OCR batches finish)

Takes the per-page PageResult JSON files and turns them into Question
objects: splits each page's markdown at question-number boundaries, then
within each question block separates Hindi lines from English lines and
option markers from body text.

This module is intentionally NOT a language model. It only reorganizes text
that PaddleOCR-VL already produced, using regex + Unicode range checks, so
it cannot invent content — the #1 requirement from the spec.

Usage:
    python structure_questions.py <ocr_out_dir> <config_path> <manifest_path> <out_path>
"""
import sys
import json
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.schema import Question, Option  # noqa: E402
from common.question_patterns import (  # noqa: E402
    normalize_number_token, match_explicit_number, match_bare_number,
    match_option_label, has_letter_option, classify_line_language,
    strip_number_prefix,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pages(ocr_out_dir: Path) -> list:
    pages = []
    for p in sorted(ocr_out_dir.glob("page_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            pages.append(json.load(f))
    return pages


class SequenceState:
    """Threads question-numbering state across the whole document (pages are
    processed in order) so a bare 'N.' line can be told apart from an
    embedded statement-list item — see question_patterns.py's module
    docstring for the reasoning."""

    def __init__(self):
        self.last_number = None
        self.twin_seen = False

    def is_boundary(self, line: str):
        """Returns (is_boundary: bool, block_number: int|None, raw: str)."""
        num, raw = match_explicit_number(line)
        if raw:
            self.last_number = num
            self.twin_seen = False
            return True, num, raw

        num, raw = match_bare_number(line)
        if raw:
            if self.last_number is None:
                # No explicit marker has been seen yet anywhere in the
                # document — accept the first bare number as a starting
                # point rather than losing the page entirely.
                self.last_number = num
                self.twin_seen = True
                return True, num, raw
            if num == self.last_number and not self.twin_seen:
                # The Hindi (or English) twin of the current question,
                # printed as a bare number without "Q."/"प्रश्न".
                self.twin_seen = True
                return True, num, raw
            # Any other bare number (statement-list items, repeated option
            # codes, etc.) is NOT trusted to open a new question on its own.
            return False, None, ""

        return False, None, ""


def split_document_into_blocks(pages: list) -> list:
    """Flatten all pages (in order) into a single line stream and split it
    into question blocks using sequence-aware boundary detection. Returns a
    list of dicts: {"page": int, "lines": [str, ...]}."""
    state = SequenceState()
    blocks = []
    current = None  # {"page": int, "lines": [...]}

    for page in pages:
        page_num = page["page"]
        lines = [l for l in page["markdown"].split("\n") if l.strip() != ""]
        for line in lines:
            is_boundary, _num, _raw = state.is_boundary(line)
            if is_boundary:
                if current is not None:
                    blocks.append(current)
                current = {"page": page_num, "lines": [line]}
            else:
                if current is None:
                    # Content before the first recognizable question number
                    # (paper header, instructions) — collect it under a
                    # synthetic "page N preface" block so nothing is silently
                    # dropped, but it will simply fail number parsing and get
                    # flagged rather than mis-numbered.
                    current = {"page": page_num, "lines": [line]}
                else:
                    current["lines"].append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def parse_block(lines: list, page_num: int, min_dev_ratio: float,
                 expected_option_counts: list) -> Question:
    first_line = lines[0]
    number, raw_number_text = normalize_number_token(first_line)
    body_lines = [strip_number_prefix(first_line)] + lines[1:]
    body_lines = [l for l in body_lines if l.strip()]

    # If this block has ANY letter-based option ("(A)"/"(B)"...) anywhere,
    # numeric "(1)(2)(3)(4)" markers are disabled for the whole block — RPSC
    # papers use one style or the other per question, so a numbered
    # statement list ("1. Statement one") won't be misread as options B/C.
    allow_numeric_options = not any(has_letter_option(l) for l in body_lines)

    question_hi_parts, question_en_parts = [], []
    options = []  # list[Option], built label -> option index
    label_index = {}
    in_options = False

    for line in body_lines:
        label, remainder = match_option_label(line, allow_numeric=allow_numeric_options)
        if label:
            in_options = True
            lang = classify_line_language(remainder, min_dev_ratio)
            if label not in label_index:
                opt = Option(label=label)
                options.append(opt)
                label_index[label] = len(options) - 1
            opt = options[label_index[label]]
            if lang == "hi":
                opt.hi = (opt.hi + " " + remainder).strip()
            else:
                opt.en = (opt.en + " " + remainder).strip()
            continue

        if not in_options:
            lang = classify_line_language(line, min_dev_ratio)
            if lang == "hi":
                question_hi_parts.append(line)
            else:
                question_en_parts.append(line)
        else:
            # Continuation of the most recent option's text (wrapped line)
            if options:
                lang = classify_line_language(line, min_dev_ratio)
                opt = options[-1]
                if lang == "hi":
                    opt.hi = (opt.hi + " " + line).strip()
                else:
                    opt.en = (opt.en + " " + line).strip()

    q = Question(
        question_no=number,
        raw_number_text=raw_number_text.strip(),
        page=page_num,
        question_hi=" ".join(question_hi_parts).strip(),
        question_en=" ".join(question_en_parts).strip(),
        options=options,
    )

    has_formula_marker = any(("$" in l or "\\(" in l or "\\[" in l) for l in lines)
    q.has_formula = has_formula_marker

    recompute_reasons(q, expected_option_counts)
    return q


def config_flag_missing_pair(q: Question) -> bool:
    return not (q.question_hi and q.question_en)


def recompute_reasons(q: Question, expected_option_counts: list) -> None:
    """Single source of truth for why a question is flagged. Called both
    right after parsing a block and again after dedup_merge combines two
    blocks, so reasons never go stale relative to the question's current
    (possibly just-merged) content."""
    reasons = []
    if config_flag_missing_pair(q):
        reasons.append("missing_bilingual_pair")
    if len(q.options) not in expected_option_counts:
        reasons.append(f"unexpected_option_count:{len(q.options)}")
    if q.question_no is None:
        reasons.append("unreadable_question_number")
    q.review_reasons = reasons
    q.needs_review = len(reasons) > 0


def dedup_merge(questions: list, expected_option_counts: list) -> list:
    """RPSC papers occasionally split a single bilingual question across two
    detected blocks (e.g. Hindi paragraph OCR'd as its own block due to a
    stray page break). If the SAME question number appears twice with one
    block missing Hindi and the other missing English, merge them instead of
    emitting a duplicate — this is the paper's #1 stated failure mode to
    avoid."""
    by_number = {}
    ordered = []
    for q in questions:
        if q.question_no is None:
            ordered.append(q)
            continue
        if q.question_no in by_number:
            existing = by_number[q.question_no]
            if not existing.question_hi and q.question_hi:
                existing.question_hi = q.question_hi
            if not existing.question_en and q.question_en:
                existing.question_en = q.question_en
            existing_labels = {o.label for o in existing.options}
            for opt in q.options:
                if opt.label in existing_labels:
                    # Same label already present (e.g. both halves carried
                    # an "A" option) — fill in whichever language is empty
                    # rather than dropping the second block's text.
                    idx = next(i for i, o in enumerate(existing.options) if o.label == opt.label)
                    if not existing.options[idx].hi and opt.hi:
                        existing.options[idx].hi = opt.hi
                    if not existing.options[idx].en and opt.en:
                        existing.options[idx].en = opt.en
                else:
                    existing.options.append(opt)
            existing.has_formula = existing.has_formula or q.has_formula
            existing.figures = list(dict.fromkeys(existing.figures + q.figures))
            existing.source_blocks.append(f"page{q.page}:merged_duplicate")
            recompute_reasons(existing, expected_option_counts)
        else:
            by_number[q.question_no] = q
            ordered.append(q)
    return ordered


def main():
    if len(sys.argv) != 5:
        print("Usage: structure_questions.py <ocr_out_dir> <config_path> "
              "<manifest_path> <out_path>", file=sys.stderr)
        sys.exit(1)

    ocr_out_dir = Path(sys.argv[1])
    config = load_config(sys.argv[2])
    with open(sys.argv[3], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out_path = Path(sys.argv[4])

    struct_cfg = config["structuring"]
    val_cfg = config["validation"]

    pages = load_pages(ocr_out_dir)
    pages.sort(key=lambda p: p["page"])
    figures_by_page = {p["page"]: p.get("figures", []) for p in pages}
    confidence_by_page = {p["page"]: p.get("mean_confidence") for p in pages}

    blocks = split_document_into_blocks(pages)
    all_questions = []
    for block in blocks:
        q = parse_block(
            block["lines"],
            page_num=block["page"],
            min_dev_ratio=struct_cfg["min_devanagari_ratio_for_hindi"],
            expected_option_counts=val_cfg["expected_option_counts"],
        )
        q.ocr_confidence = confidence_by_page.get(block["page"])
        q.figures = figures_by_page.get(block["page"], [])
        q.source_blocks = [f"page{block['page']}"]
        all_questions.append(q)

    all_questions = dedup_merge(all_questions, val_cfg["expected_option_counts"])
    all_questions.sort(key=lambda q: (q.question_no is None, q.question_no or 0, q.page))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "doc": manifest["doc_stem"],
            "num_pages": manifest["num_pages"],
            "questions": [q.to_dict() for q in all_questions],
        }, f, ensure_ascii=False, indent=2)

    print(f"Structured {len(all_questions)} questions from "
          f"{manifest['num_pages']} pages -> {out_path}")


if __name__ == "__main__":
    main()
