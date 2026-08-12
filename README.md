# RPSC/RAS PYQ Extraction Pipeline — ₹0, GitHub Actions

Drop a scanned bilingual (Hindi+English) RPSC/RAS previous-year-paper PDF into
`input/`, push it, and GitHub Actions produces structured `questions.json`
(+ a `review_queue.json` for anything uncertain) in `output/<paper-name>/`.
Zero paid APIs, zero credits, runs entirely on GitHub's free public-repo
compute.

## 1. Architecture at a glance

```
push PDF to input/
        │
        ▼
 ┌───────────────┐   render @300dpi, deskew, denoise, CLAHE
 │  discover job │   → compute dynamic batch matrix (N pages/batch)
 └───────┬───────┘
         │ matrix: [{batch 1: pages 1-4}, {batch 2: pages 5-8}, ...]
         ▼
 ┌─────────────────────────────────────────────────────┐
 │  ocr job  (matrix, up to 18 in parallel)             │
 │  per page:                                           │
 │    • PaddleOCR-VL  → markdown + layout + tables       │
 │      + formulas (LaTeX) + reading order + confidence  │
 │    • Tesseract (hin+eng) → independent 2nd opinion    │
 │    • crop figure/chart/map regions → standalone PNGs  │
 └───────────────────────┬───────────────────────────────┘
                          │ per-page JSON artifacts
                          ▼
 ┌─────────────────────────────────────────────────────┐
 │  structure_and_validate job (single job)             │
 │    1. structure_questions.py — regex-only splitting  │
 │       into questions, Hindi/English pairing, options  │
 │    2. validate.py — cross-engine agreement scoring,   │
 │       confidence thresholds, dup/gap detection         │
 │    3. commit output/<doc>/questions.json +            │
 │       review_queue.json + figures/                    │
 └─────────────────────────────────────────────────────┘
```

## 2. Model / tool choices, and why

| Component | Choice | Why |
|---|---|---|
| **Primary reader** | **PaddleOCR-VL (0.9B, v1.6)** via the `paddleocr` pip package | Purpose-built document-parsing VLM: single pass does layout detection, OCR, table structure, and formula→LaTeX together, with reading order. Scores ~96% on OmniDocBench v1.6 — currently at or near the top of open, freely-licensed document parsers. At 0.9B parameters it is small enough to run on a **CPU-only GitHub Actions runner** in a realistic time budget; a 7B+ general VLM (Qwen2.5-VL-7B etc.) would be far more accurate-*sounding* but is 5-10x too slow on CPU for a ₹0 pipeline with no GPU runner available on the free tier. Natively supports 109 languages including Devanagari, so Hindi and English on the same page are read in one pass with no language-switching logic needed. Apache-2.0 licensed, no API key, no usage cap. |
| **Cross-validation reader** | **Tesseract OCR** (`hin+eng` traineddata) | Architecturally independent from PaddleOCR-VL (classical engine, not a VLM). Running it in parallel is nearly free (CPU-light, seconds/page) and gives a second, uncorrelated opinion — when two very different engines agree, that's real evidence of correctness; when they disagree sharply, that's the cheapest possible hallucination/misread detector. It is *never* used to overwrite PaddleOCR-VL's output, only to score agreement. |
| **Question/option structuring** | **Deterministic regex + Unicode-range classification** (`scripts/structure_questions.py`), not a language model | This is the component the spec is strictest about: it must never invent text. A generative model doing "clean this up into JSON" is exactly the kind of step that silently drops or paraphrases things. Regex over already-OCR'd markdown can only reorganize/relabel — it physically cannot hallucinate new words. Numbering (Arabic + Devanagari digits), option markers (A/B/C/D, 1/2/3/4, i/ii/iii/iv, code-based combinations like "1 and 2"), and Hindi-vs-English line classification (by Devanagari Unicode block ratio) are all handled this way. |
| **Optional structuring assist** | Local **Qwen2.5-3B-Instruct** GGUF via `llama-cpp-python`, **disabled by default** | Left in `pipeline_config.yaml` as `structuring.llm_assist.enabled: false` for cases where regex genuinely can't resolve an edge case (e.g. a badly OCR'd number). If you turn it on, its suggestions are written to `review_queue.json` only — they never get silently merged into `questions.json`. Kept off by default because it adds ~2GB of model download and CPU time for a benefit that, given PaddleOCR-VL's reading order + regex, is usually marginal. |
| **Diagrams/tables/maps** | Cropped out as standalone PNGs using PaddleOCR-VL's layout bounding boxes | Per your requirement: extracted as separate image assets referenced by path in the JSON, rather than relying on a blurry full-page screenshot or trying to describe them in text. |
| **Compute** | GitHub-hosted `ubuntu-latest` runners, **public repo → unlimited free minutes**, CPU only | Confirmed as of 2026: public repositories get unlimited free standard-runner minutes; GPU ("larger") runners are a paid/Team-plan feature, so the whole design deliberately targets CPU-feasible models rather than assuming GPU access. |

### Why not just send every page to one big VLM (e.g. GPT-4V-class or Qwen2.5-VL-7B)?
- No GPU on free public-repo runners → a 7B-class VLM on CPU would take minutes *per page*, and 30-40 pages sequentially would blow past sane workflow times (and risk the 6-hour job cap if run serially).
- PaddleOCR-VL's layout-aware, purpose-trained-for-documents approach benchmarks *ahead* of many larger general VLMs on document OCR specifically (OmniDocBench), so you're not trading accuracy for speed here — for this exact task (structured document parsing, not open-ended visual reasoning) the small specialist model wins on both axes.

## 3. Parallelization strategy

- `discover` renders all pages once (cheap: PDF rendering + OpenCV ops, no ML), then computes a **dynamic GitHub Actions matrix** — one job per `pages_per_batch` pages (default 4).
- The `ocr` job matrix runs **up to 18 batches concurrently** (kept under the 20-concurrent-job free-tier cap so `discover`/`structure` still get a slot). For a 40-page paper at 4 pages/batch that's 10 batches — all run at once, so OCR wall-clock is roughly "time for one batch," not "time for the whole paper."
- `structure_and_validate` runs once, after all batches finish, on lightweight text-only processing (no ML), so it's fast regardless of paper length.

## 4. Caching (this matters a lot for speed)

Two `actions/cache` entries:
1. **pip cache**, keyed on `requirements.txt` hash — avoids re-resolving/downloading Python packages every run.
2. **PaddleOCR-VL model weights** (~1.5GB), keyed on `MODEL_CACHE_VERSION` + `requirements.txt` hash, caching `~/.paddlex`, `~/.paddleocr`, `~/.cache/huggingface`. **This is the single biggest speed lever** — without it, every matrix job would re-download ~1.5GB before doing any work. With it, only the very first run after a cache-key change pays that cost; every run after that loads weights from disk in a few seconds. Bump `MODEL_CACHE_VERSION` in the workflow env if you ever need to force a clean re-download (e.g. after pinning a newer PaddleOCR-VL release).

## 5. Validation / anti-hallucination design

Every question carries:
- `ocr_confidence` — PaddleOCR-VL's own recognition score for that page.
- `cross_engine_agreement` — token-level similarity (RapidFuzz) between PaddleOCR-VL's text and Tesseract's independent read of the same page.
- `needs_review` + `review_reasons` — set when: confidence or agreement falls below threshold, a question's Hindi or English half is missing, the option count is outside `[2,3,4,5]`, the question number couldn't be parsed, or the question number is a duplicate.

Anything flagged goes to `review_queue.json` instead of `questions.json`. **Nothing is ever guessed to fill a gap** — a low-confidence or unpaired question is surfaced for a human to check, never silently "completed." Document-level gaps (e.g. question #47 never detected at all) and duplicate numbers are reported at the top of `review_queue.json` too.

Be realistic: **no free (or paid) OCR/VLM pipeline gets literally 100% on scanned exam papers with equations, code-based options, and mixed scripts.** This design's honesty is in refusing to paper over that — it pushes uncertain cases to a review queue rather than confidently emitting a wrong answer. Expect the clean `questions.json` to need occasional spot-checks and the review queue to need a manual pass, especially early on while you tune `pipeline_config.yaml` thresholds against your specific papers.

## 6. Hindi + English pairing logic

Within each detected question block (bounded by consecutive question-number tokens), every line is classified Hindi or English by the fraction of its alphabetic characters in the Devanagari Unicode block (U+0900–U+097F), threshold configurable. Lines before the first option marker become `question_hi`/`question_en`; lines after an option marker are attached to that option's `hi`/`en` field. If a question number is detected twice in a row with only one language filled in each time (a known layout quirk where a stray line break splits the bilingual pair across two "blocks"), `dedup_merge()` combines them into one question **instead of emitting a duplicate** — this directly targets the "must not duplicate bilingual questions" requirement.

## 7. Equations, tables, statement-based/code options

- **Equations**: PaddleOCR-VL's formula recognition emits LaTeX inline in the markdown (`$...$`); this is preserved verbatim in `question_en`/`question_hi`/option text. `has_formula` is set on any block containing LaTeX delimiters, so you can filter these for extra scrutiny.
- **Tables**: PaddleOCR-VL emits table structure as HTML/markdown tables inside the page markdown; kept verbatim.
- **Statement-based questions / code options** (e.g. "1 and 2 only", "(a) A, B and D are correct"): the option parser captures whatever text follows an option marker **verbatim**, with no attempt to interpret or evaluate the code — this avoids the model "solving" or reinterpreting the option, which would risk silently changing its meaning.

## 8. Expected processing time (20–40 pages, ~150 questions)

Rough, CPU-only, after the model-weight cache is warm:

| Stage | Time |
|---|---|
| `discover` (render + deskew ~40 pages) | ~45–90 s |
| `ocr` matrix (10 batches × 4 pages, running in parallel) | ~2–4 min wall-clock (bounded by the slowest batch, not the total page count) |
| `structure_and_validate` (regex only, ~150 questions) | ~15–30 s |
| commit | ~10 s |
| **Total** | **≈ 4–7 minutes**, most of it parallelized |

First run after a fresh clone (cold model cache) will additionally spend ~2-4 minutes downloading PaddleOCR-VL weights *once*, in whichever batch job wins the cache race; every run after that is fast.

## 9. Repository structure

```
.
├── .github/workflows/extract.yml   # the 3-stage pipeline described above
├── input/                          # drop PYQ PDFs here
├── output/                         # <doc_stem>/questions.json, review_queue.json, figures/
├── config/pipeline_config.yaml     # every tunable knob (DPI, batch size, thresholds...)
├── scripts/
│   ├── preprocess.py               # PDF -> cleaned page PNGs + dynamic matrix
│   ├── ocr_batch.py                # PaddleOCR-VL + Tesseract per batch, figure cropping
│   ├── structure_questions.py      # regex-only question/option/bilingual structuring
│   ├── validate.py                 # cross-engine scoring, thresholds, clean/review split
│   └── common/
│       ├── schema.py                # Question/Option/PageResult dataclasses
│       └── question_patterns.py    # numbering/option regex, Devanagari detection
└── requirements.txt
```

## 10. Setup instructions

1. Create a **public** GitHub repository (public is required for the free-unlimited-minutes tier) and push this folder's contents to it.
2. No secrets or API keys are needed — nothing in this pipeline calls a paid service.
3. Drop a PDF into `input/`, e.g. `input/rpsc-ras-2023-gs1.pdf`, commit, and push.
4. The workflow triggers automatically on any push that adds/changes a file under `input/**.pdf`. You can also run it manually from the **Actions** tab → *Extract RPSC/RAS Questions* → *Run workflow*, optionally naming a specific file in the `pdf_filename` input if you have more than one PDF in `input/`.
5. Watch progress in the **Actions** tab — you'll see `discover` → parallel `ocr` batch jobs → `structure_and_validate`.
6. When it finishes, `output/<pdf-name>/questions.json` and `output/<pdf-name>/review_queue.json` are committed straight back to the repo (and also uploaded as a downloadable workflow artifact, kept 90 days, as a backup).
7. Open `review_queue.json` first on a new paper type — check a handful of flagged items against the source PDF and adjust the thresholds in `config/pipeline_config.yaml` (`validation:` section) if they're too strict/loose for your scan quality.

### Tuning for a specific paper
- Blurry/low-quality scan → raise `preprocess.dpi` to 350–400.
- Papers running long/timing out → lower `batching.pages_per_batch` (more, smaller parallel jobs) — but stay under `256` total matrix jobs and mind the 20-concurrent-job cap.
- Too many false "needs_review" flags → lower `validation.min_ocr_confidence` / `min_cross_engine_similarity` slightly, but re-spot-check accuracy after any change.
- Non-standard option markers your paper uses → add a pattern to `structuring.option_label_patterns` / mirror it in `scripts/common/question_patterns.py`.

## 11. Known limitations (stated up front, not discovered later)

- This is v1 as requested: PDF → JSON. No web UI, no database, no dedup across multiple *different* PDFs of the same exam year.
- Handwriting, extremely skewed/torn scans, or very low-DPI source PDFs will produce more `needs_review` items — that's by design (flagging beats guessing), but it means the review queue does real work; budget for a manual pass on your first few papers.
- `llm_assist` (optional local LLM structuring) is untuned/off by default; treat it as an experimental toggle, not a relied-upon part of the accuracy story.
- Table/formula fidelity depends on PaddleOCR-VL's own table/formula sub-modules; very unusual table layouts (nested merged cells across a page break, etc.) are the most likely source of imperfect (not hallucinated — just imperfect) reconstruction.
