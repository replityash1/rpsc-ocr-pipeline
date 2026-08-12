#!/usr/bin/env python3
"""
Stage 2 (job: ocr, matrix — one process per batch of pages)

For every page image in the assigned range:
  1. Run PaddleOCR-VL (layout + OCR + table + formula, single pass, CPU).
  2. Run Tesseract (hin+eng) independently over the same page as a second,
     architecturally-different opinion — used later purely for agreement
     scoring, never to overwrite PaddleOCR-VL's output.
  3. Crop out figure/diagram/map/chart regions PaddleOCR-VL's layout model
     found, and save them as standalone PNGs instead of relying on a full
     page screenshot.

Usage:
    python ocr_batch.py <pages_dir> <start_page> <end_page> <out_dir> <config_path>

Writes one JSON file per page: <out_dir>/page_XXXX.json  (PageResult schema)
plus <out_dir>/figures/page_XXXX_figN.png
"""
import sys
import json
from pathlib import Path

import cv2
import pytesseract
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.schema import PageResult  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_val(obj, *keys_or_attrs, default=None):
    """Safely fetch a key or attribute from a dict or object."""
    if isinstance(obj, dict):
        for key in keys_or_attrs:
            if key in obj and obj[key] is not None:
                return obj[key]
    else:
        for attr in keys_or_attrs:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is not None:
                    return val
    return default


def run_paddleocr_vl(image_path: Path, pipeline):
    """Returns (markdown:str, blocks:list, mean_confidence:float|None)."""
    results = pipeline.predict(str(image_path))
    markdown_parts = []
    blocks = []
    confidences = []
    for res in results:
        md = _get_val(res, "markdown", default="")
        if md:
            markdown_parts.append(md)

        layout_blocks = _get_val(res, "layout_parsing_result", "parsing_res_list", default=[])
        if isinstance(layout_blocks, list):
            blocks.extend(layout_blocks)

        rec_scores = _get_val(res, "rec_scores", "text_scores")
        if rec_scores:
            confidences.extend([s for s in rec_scores if isinstance(s, (int, float))])

    mean_conf = sum(confidences) / len(confidences) if confidences else None
    return "\n".join(markdown_parts), blocks, mean_conf


def run_tesseract(image_path: Path, langs: str) -> str:
    img = cv2.imread(str(image_path))
    try:
        return pytesseract.image_to_string(img, lang=langs)
    except pytesseract.TesseractError as e:
        print(f"[warn] tesseract failed on {image_path.name}: {e}", file=sys.stderr)
        return ""


def crop_figures(image_path: Path, blocks: list, out_dir: Path, page_num: int) -> list:
    """Pull out figure/chart/table/map regions as standalone images using
    layout bounding boxes reported by PaddleOCR-VL."""
    figure_labels = {"image", "figure", "chart", "picture", "table", "seal"}
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    saved = []
    fig_idx = 0
    for block in blocks:
        raw_label = _get_val(block, "label", "block_label", default="")
        label = str(raw_label).lower()
        bbox = _get_val(block, "bbox", "block_bbox")

        if label not in figure_labels or not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        fig_idx += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"page_{page_num:04d}_fig{fig_idx}.png"
        cv2.imwrite(str(fig_path), img[y1:y2, x1:x2])
        saved.append(str(fig_path))
    return saved


def main():
    if len(sys.argv) != 6:
        print("Usage: ocr_batch.py <pages_dir> <start_page> <end_page> "
              "<out_dir> <config_path>", file=sys.stderr)
        sys.exit(1)

    pages_dir = Path(sys.argv[1])
    start_page = int(sys.argv[2])
    end_page = int(sys.argv[3])
    out_dir = Path(sys.argv[4])
    config = load_config(sys.argv[5])

    ocr_cfg = config["ocr"]
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / config["output"]["figures_subdir"]

    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL(device=ocr_cfg["device"])

    for page_num in range(start_page, end_page + 1):
        image_path = pages_dir / f"page_{page_num:04d}.png"
        if not image_path.exists():
            print(f"[warn] missing page image {image_path}", file=sys.stderr)
            continue

        markdown, blocks, mean_conf = run_paddleocr_vl(image_path, pipeline)

        tesseract_text = ""
        if ocr_cfg["cross_validate_with_tesseract"]:
            tesseract_text = run_tesseract(image_path, ocr_cfg["tesseract_langs"])

        figures = crop_figures(image_path, blocks, figures_dir, page_num)

        # Convert block objects to dicts if required by PageResult JSON serialization
        serializable_blocks = [
            b.to_dict() if hasattr(b, "to_dict")
            else vars(b) if hasattr(b, "__dict__")
            else b
            for b in blocks
        ]

        page_result = PageResult(
            page=page_num,
            markdown=markdown,
            blocks=serializable_blocks,
            mean_confidence=mean_conf,
            tesseract_text=tesseract_text,
            figures=figures,
        )

        out_path = out_dir / f"page_{page_num:04d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(page_result.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"page {page_num}: {len(markdown)} md chars, "
              f"conf={mean_conf}, figures={len(figures)}")

    pipeline.close() if hasattr(pipeline, "close") else None


if __name__ == "__main__":
    main()
