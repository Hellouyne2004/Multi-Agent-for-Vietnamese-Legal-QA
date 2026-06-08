"""
Validate prepared legal corpus text/chunks before indexing.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path


if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline.chunker import chunk_text_file
from src.data_pipeline.pdf_preprocessor import (
    count_articles,
    normalize_ocr_legal_markers,
    validate_article_sequence,
    validate_chapter_sequence,
)


EXPECTED_ARTICLE_RANGES = {
    "45/2019/QH14": (1, 220),
    "74/2025/QH15": (1, 55),
}
EXPECTED_CHAPTER_RANGES = {
    "45/2019/QH14": (1, 17),
    "74/2025/QH15": (1, 8),
}

FALSE_ARTICLE_HEADING_RE = re.compile(r"(?i)^Điều\s+\d+\s+của\s+(?:Luật|Bộ luật)\s+này\b")
MIN_CHUNK_CHARS = 120


def _read_header_value(text: str, key: str) -> str:
    prefix = key.upper() + ":"
    for line in text.splitlines()[:80]:
        if line.upper().startswith(prefix):
            return line.split(":", 1)[1].strip()
        if not line.strip():
            break
    return ""


def validate_text_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    normalized_text = normalize_ocr_legal_markers(text)
    so_hieu = _read_header_value(text, "SO_HIEU_VAN_BAN")
    index_ready = _read_header_value(text, "INDEX_READY").lower() == "true"
    article_count = count_articles(normalized_text)
    expected_min, expected_max = EXPECTED_ARTICLE_RANGES.get(so_hieu, (None, None))
    sequence_report = validate_article_sequence(normalized_text, expected_min, expected_max)
    expected_chapter_min, expected_chapter_max = EXPECTED_CHAPTER_RANGES.get(so_hieu, (None, None))
    chapter_report = validate_chapter_sequence(normalized_text, expected_chapter_min, expected_chapter_max)
    chunks = chunk_text_file(str(path))

    errors = []
    if len(text.strip()) < 1000:
        errors.append("text_too_short")
    if expected_min is not None and expected_max is not None:
        expected_count = expected_max - expected_min + 1
        if article_count != expected_count:
            errors.append(f"article_count_mismatch:{article_count}!={expected_count}")
    elif article_count < 1:
        errors.append("article_count_below_min:0<1")
    if sequence_report["missing_articles"]:
        errors.append(f"missing_articles:{sequence_report['missing_articles']}")
    if sequence_report["duplicate_articles"]:
        errors.append(f"duplicate_articles:{sequence_report['duplicate_articles']}")
    if chapter_report["missing_chapters"]:
        errors.append(f"missing_chapters:{chapter_report['missing_chapters']}")
    if chapter_report["duplicate_chapters"]:
        errors.append(f"duplicate_chapters:{chapter_report['duplicate_chapters']}")
    if not chunks:
        errors.append("no_chunks_created")
    if any(not chunk.dieu and "Điều " in chunk.content for chunk in chunks):
        errors.append("article_chunk_missing_dieu_metadata")
    false_chunk_headings = [
        chunk.dieu
        for chunk in chunks
        if chunk.dieu and FALSE_ARTICLE_HEADING_RE.search(chunk.dieu)
    ]
    if false_chunk_headings:
        errors.append(f"false_article_headings:{false_chunk_headings[:10]}")
    short_chunks = [
        chunk.chunk_index
        for chunk in chunks
        if len(chunk.content.strip()) < MIN_CHUNK_CHARS
    ]
    if short_chunks:
        errors.append(f"short_chunks:{short_chunks[:10]}")
    if not index_ready:
        errors.append("index_ready_false")

    return {
        "file": str(path),
        "so_hieu_van_ban": so_hieu,
        "index_ready": index_ready,
        "article_count": article_count,
        "expected_article_range": [expected_min, expected_max],
        "sequence": sequence_report,
        "expected_chapter_range": [expected_chapter_min, expected_chapter_max],
        "chapter_sequence": chapter_report,
        "chunk_count": len(chunks),
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate prepared legal RAG corpus.")
    parser.add_argument(
        "--input",
        default=str(ROOT_DIR / "data" / "processed" / "text"),
        help="Directory containing prepared .txt files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    reports = [validate_text_file(path) for path in sorted(input_dir.glob("*.txt"))]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0 if reports and all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
