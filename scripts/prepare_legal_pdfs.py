"""
Prepare the first legal PDF corpus for RAG.

Default corpus:
- Bộ luật Lao động 45/2019/QH14
- Luật Việc làm 74/2025/QH15

Output:
- data/raw/pdf_original/*.pdf
- data/processed/pdf_ocr/*.pdf
- data/processed/text/*.txt
- data/processed/metadata/*.metadata.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
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

from src.data_pipeline.pdf_preprocessor import prepare_default_legal_pdfs
from src.utils.logger import logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR and normalize the initial labor/employment legal PDF corpus."
    )
    parser.add_argument(
        "--root",
        default=str(ROOT_DIR),
        help="Project root directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared OCR/text artifacts.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Run OCR even if the PDF already has a usable text layer.",
    )
    parser.add_argument(
        "--mark-reviewed",
        action="store_true",
        help="Mark output text as reviewed/index-ready after you manually inspect it.",
    )
    args = parser.parse_args()

    try:
        reports = prepare_default_legal_pdfs(
            root_dir=args.root,
            overwrite=args.overwrite,
            force_ocr=args.force_ocr,
            mark_reviewed=args.mark_reviewed,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        logger.error(
            "Install OCR dependencies first, then rerun: "
            "ocrmypdf, Tesseract with Vietnamese language data, Ghostscript, and Poppler."
        )
        return 2
    except Exception as exc:
        logger.error(f"Failed to prepare legal PDFs: {exc}", exc_info=True)
        return 1

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
