#!/usr/bin/env python3
"""
Stage 4 (job: structure, runs right after structure_questions.py)

Adds cross-engine agreement scores (PaddleOCR-VL vs Tesseract, per page) to
each question, applies confidence/consistency thresholds, and produces the
final split: questions.json (clean) + review_queue.json (flagged).

Nothing here rewrites extracted text — validation only ever ADDS metadata
(needs_review, review_reasons, cross_engine_agreement) or reorders items
between the two output files.

Usage:
    python validate.py <structured_json> <ocr_out_dir> <config_path> <final_out_dir>
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import yaml
from rapidfuzz import fuzz


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tesseract_by_page(ocr_out_dir: Path) -> dict:
    result = {}
    for p in ocr_out_dir.glob("page_*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        result[data["page"]] = data.get("tesseract_text", "")
    return result


def load_markdown_by_page(ocr_out_dir: Path) -> dict:
    result = {}
    for p in ocr_out_dir.glob("page_*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        result[data["page"]] = data.get("markdown", "")
    return result


def cross_engine_similarity(paddle_text: str, tesseract_text: str) -> float:
    if not paddle_text.strip() or not tesseract_text.strip():
        return 0.0
    return fuzz.token_sort_ratio(paddle_text, tesseract_text) / 100.0


def flag_numbering_gaps(questions: list) -> set:
    numbers = sorted({q["question_no"] for q in questions if q["question_no"] is not None})
    missing = set()
    for a, b in zip(numbers, numbers[1:]):
        if b - a > 1:
            missing.update(range(a + 1, b))
    return missing


def flag_duplicates(questions: list) -> set:
    seen = defaultdict(int)
    for q in questions:
        if q["question_no"] is not None:
            seen[q["question_no"]] += 1
    return {n for n, c in seen.items() if c > 1}


def main():
    if len(sys.argv) != 5:
        print("Usage: validate.py <structured_json> <ocr_out_dir> "
              "<config_path> <final_out_dir>", file=sys.stderr)
        sys.exit(1)

    structured_path = Path(sys.argv[1])
    ocr_out_dir = Path(sys.argv[2])
    config = load_config(sys.argv[3])
    final_out_dir = Path(sys.argv[4])
    val_cfg = config["validation"]

    with open(structured_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    questions = doc["questions"]

    tesseract_by_page = load_tesseract_by_page(ocr_out_dir)
    markdown_by_page = load_markdown_by_page(ocr_out_dir)

    dup_numbers = flag_duplicates(questions) if val_cfg["flag_duplicate_question_numbers"] else set()
    missing_numbers = flag_numbering_gaps(questions) if val_cfg["flag_numbering_gaps"] else set()

    for q in questions:
        page = q["page"]
        combined_text = (q.get("question_hi", "") + " " + q.get("question_en", "")).strip()
        page_tess = tesseract_by_page.get(page, "")
        agreement = cross_engine_similarity(combined_text, page_tess) if combined_text else None
        q["cross_engine_agreement"] = agreement

        reasons = set(q.get("review_reasons", []))

        if q.get("ocr_confidence") is not None and q["ocr_confidence"] < val_cfg["min_ocr_confidence"]:
            reasons.add("low_ocr_confidence")

        if agreement is not None and agreement < val_cfg["min_cross_engine_similarity"]:
            reasons.add("low_cross_engine_agreement")

        if q["question_no"] in dup_numbers:
            reasons.add("duplicate_question_number")

        if not combined_text:
            reasons.add("empty_question_text")

        q["review_reasons"] = sorted(reasons)
        q["needs_review"] = len(reasons) > 0

    clean = [q for q in questions if not q["needs_review"]]
    flagged = [q for q in questions if q["needs_review"]]

    final_out_dir.mkdir(parents=True, exist_ok=True)

    with open(final_out_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump({
            "doc": doc["doc"],
            "num_pages": doc["num_pages"],
            "total_questions": len(questions),
            "clean_count": len(clean),
            "flagged_count": len(flagged),
            "questions": clean,
        }, f, ensure_ascii=False, indent=2)

    if val_cfg.get("write_review_queue", True):
        with open(final_out_dir / "review_queue.json", "w", encoding="utf-8") as f:
            json.dump({
                "doc": doc["doc"],
                "flagged_count": len(flagged),
                "missing_question_numbers": sorted(missing_numbers),
                "duplicate_question_numbers": sorted(dup_numbers),
                "questions": flagged,
            }, f, ensure_ascii=False, indent=2)

    if config["output"].get("write_debug_markdown", True):
        debug_dir = final_out_dir / "debug_markdown"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for page_num, md in markdown_by_page.items():
            with open(debug_dir / f"page_{page_num:04d}.md", "w", encoding="utf-8") as f:
                f.write(md)

    print(f"Validation done: {len(clean)} clean, {len(flagged)} flagged "
          f"(missing #s: {sorted(missing_numbers)}, dup #s: {sorted(dup_numbers)})")


if __name__ == "__main__":
    main()
