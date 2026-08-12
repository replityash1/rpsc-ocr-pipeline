#!/usr/bin/env python3
"""
Stage 1 (job: discover)
Render a scanned PDF to per-page PNGs, clean them up (deskew, denoise,
contrast), and emit a GitHub Actions matrix (JSON) that splits the pages
into batches for the parallel OCR jobs.

Usage:
    python preprocess.py <pdf_path> <work_dir> <config_path>

Writes:
    <work_dir>/pages/page_0001.png ...
    <work_dir>/matrix.json                (for GITHUB_OUTPUT)
    <work_dir>/manifest.json              (page count, doc name, etc.)
"""
import sys
import json
import math
from pathlib import Path

import fitz  # PyMuPDF
import cv2
import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return img  # not enough signal to estimate a reliable angle
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Ignore near-zero corrections and implausibly large ones (false positive
    # on dense text blocks) — safer to leave the page alone than to distort it.
    if abs(angle) < 0.1 or abs(angle) > 15:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def render_pdf(pdf_path: Path, out_dir: Path, dpi: int, do_deskew: bool,
                do_denoise: bool, do_clahe: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if do_deskew:
            img = deskew(img)
        if do_denoise:
            img = denoise(img)
        if do_clahe:
            img = apply_clahe(img)
        out_path = out_dir / f"page_{i+1:04d}.png"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return len(doc)


def build_matrix(num_pages: int, pages_per_batch: int) -> list:
    n_batches = math.ceil(num_pages / pages_per_batch)
    batches = []
    for b in range(n_batches):
        start = b * pages_per_batch + 1
        end = min(start + pages_per_batch - 1, num_pages)
        batches.append({"batch_id": f"{b+1:03d}", "start_page": start, "end_page": end})
    return batches


def main():
    if len(sys.argv) != 4:
        print("Usage: preprocess.py <pdf_path> <work_dir> <config_path>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    work_dir = Path(sys.argv[2])
    config = load_config(sys.argv[3])

    pre_cfg = config["preprocess"]
    batch_cfg = config["batching"]

    pages_dir = work_dir / "pages"
    num_pages = render_pdf(
        pdf_path, pages_dir,
        dpi=pre_cfg["dpi"],
        do_deskew=pre_cfg["deskew"],
        do_denoise=pre_cfg["denoise"],
        do_clahe=pre_cfg["clahe_contrast"],
    )

    matrix = build_matrix(num_pages, batch_cfg["pages_per_batch"])

    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / "matrix.json", "w", encoding="utf-8") as f:
        json.dump({"include": matrix}, f)

    manifest = {
        "pdf_name": pdf_path.name,
        "doc_stem": pdf_path.stem,
        "num_pages": num_pages,
        "num_batches": len(matrix),
    }
    with open(work_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Rendered {num_pages} pages into {len(matrix)} batches "
          f"({batch_cfg['pages_per_batch']} pages/batch).")


if __name__ == "__main__":
    main()
