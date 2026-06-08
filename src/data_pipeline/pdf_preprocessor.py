"""
PDF preprocessing utilities for legal RAG corpora.

This module keeps scanned-PDF handling outside retrieval/indexing:
raw PDF -> searchable OCR PDF + reviewed text sidecar + metadata JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logger import logger


ARTICLE_WORD_PATTERN = r"(?:Điều|Điền|Điện|Diéu|Diều|Điêu|Ðiều|Dieu)"
ARTICLE_HEADING_RE = re.compile(
    rf"(?im)^[ \t]*[“\"']?(?P<word>{ARTICLE_WORD_PATTERN})[ \t]+"
    r"(?P<number>\d{1,3}|J\d{1,2})[ \t]*[\.,:][ \t]*(?P<title>[^\n]+)"
)
CHAPTER_NUMBER_PATTERN = r"(?:XVIH|XIIH|XIH|XHI|XHII|XH|VIH|VH|TH|TV|IH|H|LL|I1|1I|[IVXLCDMỊỈÌ]+|\d+)"
CHAPTER_HEADING_RE = re.compile(
    rf"(?im)^\s*(Chương|Chuong)\s+(?P<number>{CHAPTER_NUMBER_PATTERN})(?=\s|$)(?P<title>[^\n]*)"
)
ARTICLE_RE = ARTICLE_HEADING_RE
CHAPTER_RE = CHAPTER_HEADING_RE


@dataclass(frozen=True)
class ArticleHeading:
    number: int
    heading: str
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class LegalPdfConfig:
    slug: str
    input_filename: str
    metadata: dict[str, Any]
    min_article_count: int


DEFAULT_LEGAL_PDFS: tuple[LegalPdfConfig, ...] = (
    LegalPdfConfig(
        slug="bo_luat_lao_dong_2019",
        input_filename="VanBanGoc_BO LUAT 45 QH14.pdf",
        min_article_count=180,
        metadata={
            "so_hieu_van_ban": "45/2019/QH14",
            "ten_van_ban": "Bộ luật Lao động",
            "loai_van_ban": "Bộ luật",
            "co_quan_ban_hanh": "Quốc hội",
            "ngay_ban_hanh": "2019-11-20",
            "ngay_hieu_luc": "2021-01-01",
            "nam_ban_hanh": 2019,
            "trang_thai": "Hết hiệu lực một phần",
            "tinh_trang_hieu_luc": "het_hieu_luc_mot_phan",
            "linh_vuc": "lao_dong",
            "source_url": "https://vbpl.vn/bolaodong/Pages/vbpq-lichsu.aspx?ItemID=139264",
            "file_url": "",
            "file_source": "pdf_scan",
        },
    ),
    LegalPdfConfig(
        slug="luat_viec_lam_2025",
        input_filename="VanBanGoc_74.L.2025-Luat Viec lam.pdf",
        min_article_count=50,
        metadata={
            "so_hieu_van_ban": "74/2025/QH15",
            "ten_van_ban": "Luật Việc làm",
            "loai_van_ban": "Luật",
            "co_quan_ban_hanh": "Quốc hội",
            "ngay_ban_hanh": "2025-06-16",
            "ngay_hieu_luc": "2026-01-01",
            "nam_ban_hanh": 2025,
            "trang_thai": "Hiện hành",
            "tinh_trang_hieu_luc": "con_hieu_luc",
            "linh_vuc": "viec_lam",
            "source_url": "https://vbpl.vn/bolaodong/Pages/ivbpq-thuoctinh.aspx?ItemID=179273&Keyword=",
            "file_url": "https://vbpl.vn/FileData/TW/Lists/vbpq/Attachments/179273/VanBanGoc_74.L.2025-Luat%20Viec%20lam.pdf",
            "file_source": "pdf_scan",
        },
    ),
)


def ensure_legal_corpus_dirs(root_dir: str | Path = ".") -> dict[str, Path]:
    root = Path(root_dir)
    dirs = {
        "raw_pdf": root / "data" / "raw" / "pdf_original",
        "ocr_pdf": root / "data" / "processed" / "pdf_ocr",
        "text": root / "data" / "processed" / "text",
        "metadata": root / "data" / "processed" / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_input_pdf(config: LegalPdfConfig, root_dir: str | Path = ".") -> Path:
    root = Path(root_dir)
    candidates = [
        root / "data" / "raw" / "pdf_original" / config.input_filename,
        root / "data" / "raw" / config.input_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list((root / "data" / "raw").rglob(config.input_filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find PDF: {config.input_filename}")


def copy_original_pdf(
    input_pdf: Path,
    raw_pdf_dir: Path,
    overwrite: bool = False,
) -> Path:
    target = raw_pdf_dir / input_pdf.name
    if target.resolve() == input_pdf.resolve():
        return target
    if target.exists() and not overwrite:
        return target
    shutil.copy2(input_pdf, target)
    return target


def extract_pdf_text_layer(pdf_path: str | Path) -> str:
    """Extract embedded text layer only. OCR is handled separately."""
    path = Path(pdf_path)

    try:
        import fitz  # PyMuPDF

        parts: list[str] = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    parts.append(page_text)
        return "\n".join(parts)
    except Exception as fitz_error:
        logger.debug(f"[PDF_PREPROCESS] PyMuPDF extraction skipped: {fitz_error}")

    try:
        import PyPDF2

        parts = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text)
        return "\n".join(parts)
    except Exception as pypdf_error:
        logger.debug(f"[PDF_PREPROCESS] PyPDF2 extraction skipped: {pypdf_error}")
        return ""


def _normalize_article_number(raw_number: str) -> int | None:
    token = (raw_number or "").strip().upper()
    if token.startswith("J") and token[1:].isdigit():
        token = "7" + token[1:]
    if not token.isdigit():
        return None
    return int(token)


def _trim_article_title(title: str) -> str:
    title = re.split(r"\s+(?=\d+\.\s)", title.strip(), maxsplit=1)[0]
    title = re.sub(r"\s+", " ", title)
    return title.strip(" _-")


def _is_false_article_heading(title: str) -> bool:
    title = title.strip()
    if not title:
        return True
    first_char = title[0]
    if first_char.isdigit() or first_char.islower():
        return True
    return bool(re.match(r"(?i)^(?:của|và|theo|tại)\b", title))


def find_article_headings(text: str, strict_sequence: bool = True) -> list[ArticleHeading]:
    """Return real article headings, excluding inline references and backward nested amendments."""
    normalized_text = normalize_ocr_legal_markers(text or "")
    headings: list[ArticleHeading] = []
    last_number = 0

    for match in ARTICLE_HEADING_RE.finditer(normalized_text):
        number = _normalize_article_number(match.group("number"))
        if number is None:
            continue

        title = _trim_article_title(match.group("title"))
        if not title or _is_false_article_heading(title):
            continue
        if strict_sequence and headings and number <= last_number:
            continue

        heading = f"Điều {number}. {title}"
        headings.append(
            ArticleHeading(
                number=number,
                heading=heading[:180],
                title=title,
                start=match.start("word"),
                end=match.end(0),
            )
        )
        last_number = number

    return headings


def validate_article_sequence(
    text: str,
    expected_min: int | None = None,
    expected_max: int | None = None,
) -> dict[str, Any]:
    all_candidates = find_article_headings(text, strict_sequence=False)
    headings = find_article_headings(text, strict_sequence=True)
    numbers = [heading.number for heading in headings]
    candidate_numbers = [heading.number for heading in all_candidates]

    if expected_min is None and numbers:
        expected_min = min(numbers)
    if expected_max is None and numbers:
        expected_max = max(numbers)

    missing: list[int] = []
    if expected_min is not None and expected_max is not None:
        present = set(numbers)
        missing = [number for number in range(expected_min, expected_max + 1) if number not in present]

    duplicates = sorted(
        number
        for number in set(numbers)
        if numbers.count(number) > 1
    )
    ignored_backward = [
        heading.heading
        for heading in all_candidates
        if heading.number in candidate_numbers and heading not in headings
    ]

    return {
        "article_count": len(headings),
        "candidate_count": len(all_candidates),
        "first_article": numbers[0] if numbers else None,
        "last_article": numbers[-1] if numbers else None,
        "missing_articles": missing,
        "duplicate_articles": duplicates,
        "ignored_backward_headings": ignored_backward,
    }


def count_articles(text: str) -> int:
    return len(find_article_headings(text or ""))


def count_chapters(text: str) -> int:
    return len(CHAPTER_RE.findall(text or ""))


def find_chapter_numbers(text: str) -> list[int]:
    normalized_text = normalize_ocr_legal_markers(text or "")
    numbers: list[int] = []
    for match in CHAPTER_HEADING_RE.finditer(normalized_text):
        roman = _normalize_roman_token(match.group("number"))
        number = ROMAN_TO_INT.get(roman, 0)
        if number:
            numbers.append(number)
    return numbers


def validate_chapter_sequence(
    text: str,
    expected_min: int | None = None,
    expected_max: int | None = None,
) -> dict[str, Any]:
    numbers = find_chapter_numbers(text)
    if expected_min is None and numbers:
        expected_min = min(numbers)
    if expected_max is None and numbers:
        expected_max = max(numbers)

    missing: list[int] = []
    if expected_min is not None and expected_max is not None:
        present = set(numbers)
        missing = [number for number in range(expected_min, expected_max + 1) if number not in present]

    duplicates = sorted(
        number
        for number in set(numbers)
        if numbers.count(number) > 1
    )
    return {
        "chapter_count": len(numbers),
        "first_chapter": numbers[0] if numbers else None,
        "last_chapter": numbers[-1] if numbers else None,
        "missing_chapters": missing,
        "duplicate_chapters": duplicates,
        "chapters": numbers,
    }


def text_layer_is_usable(
    text: str,
    min_chars: int = 2000,
    min_article_count: int = 5,
) -> bool:
    normalized_text = normalize_ocr_legal_markers(text)
    return len((text or "").strip()) >= min_chars and count_articles(normalized_text) >= min_article_count


def run_ocrmypdf(
    input_pdf: Path,
    output_pdf: Path,
    sidecar_txt: Path,
    force_ocr: bool = False,
    languages: str = "vie+eng",
    invalidate_digital_signatures: bool = True,
) -> None:
    executable = shutil.which("ocrmypdf")
    if not executable:
        raise RuntimeError(
            "ocrmypdf was not found. Install ocrmypdf, Tesseract, Vietnamese language data, "
            "Ghostscript, and Poppler before OCRing scanned PDFs."
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    sidecar_txt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        executable,
        "-l",
        languages,
        "--deskew",
        "--rotate-pages",
        "--output-type",
        "pdfa",
        "--sidecar",
        str(sidecar_txt),
    ]
    if invalidate_digital_signatures:
        cmd.append("--invalidate-digital-signatures")
    if force_ocr:
        cmd.append("--force-ocr")
    cmd.extend([str(input_pdf), str(output_pdf)])

    logger.info("[PDF_PREPROCESS] Running OCR: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
    "XVIII": 18,
}
INT_TO_ROMAN = {value: key for key, value in ROMAN_TO_INT.items()}
OCR_ROMAN_FIXES = {
    "H": "II",
    "LL": "II",
    "I1": "II",
    "1I": "II",
    "IH": "III",
    "TH": "III",
    "TV": "IV",
    "VH": "VII",
    "VIH": "VIII",
    "XH": "XII",
    "XIH": "XIII",
    "XHI": "XIII",
    "XIIH": "XIII",
    "XHII": "XIII",
    "XVIH": "XVII",
}

LEGAL_CHAPTER_BOUNDARIES: dict[str, list[tuple[int, int, str]]] = {
    "bo_luat_lao_dong_2019": [
        (1, 1, "NHỮNG QUY ĐỊNH CHUNG"),
        (2, 9, "VIỆC LÀM, TUYỂN DỤNG VÀ QUẢN LÝ LAO ĐỘNG"),
        (3, 13, "HỢP ĐỒNG LAO ĐỘNG"),
        (4, 59, "GIÁO DỤC NGHỀ NGHIỆP VÀ PHÁT TRIỂN KỸ NĂNG NGHỀ"),
        (5, 63, "ĐỐI THOẠI TẠI NƠI LÀM VIỆC, THƯƠNG LƯỢNG TẬP THỂ, THỎA ƯỚC LAO ĐỘNG TẬP THỂ"),
        (6, 90, "TIỀN LƯƠNG"),
        (7, 105, "THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI"),
        (8, 117, "KỶ LUẬT LAO ĐỘNG, TRÁCH NHIỆM VẬT CHẤT"),
        (9, 132, "AN TOÀN, VỆ SINH LAO ĐỘNG"),
        (10, 135, "NHỮNG QUY ĐỊNH RIÊNG ĐỐI VỚI LAO ĐỘNG NỮ VÀ BẢO ĐẢM BÌNH ĐẲNG GIỚI"),
        (11, 143, "NHỮNG QUY ĐỊNH RIÊNG ĐỐI VỚI LAO ĐỘNG CHƯA THÀNH NIÊN VÀ MỘT SỐ LAO ĐỘNG KHÁC"),
        (12, 168, "BẢO HIỂM XÃ HỘI, BẢO HIỂM Y TẾ, BẢO HIỂM THẤT NGHIỆP"),
        (13, 170, "TỔ CHỨC ĐẠI DIỆN NGƯỜI LAO ĐỘNG TẠI CƠ SỞ"),
        (14, 179, "GIẢI QUYẾT TRANH CHẤP LAO ĐỘNG"),
        (15, 212, "QUẢN LÝ NHÀ NƯỚC VỀ LAO ĐỘNG"),
        (16, 214, "THANH TRA LAO ĐỘNG, XỬ LÝ VI PHẠM PHÁP LUẬT VỀ LAO ĐỘNG"),
        (17, 218, "ĐIỀU KHOẢN THI HÀNH"),
    ],
    "luat_viec_lam_2025": [
        (1, 1, "NHỮNG QUY ĐỊNH CHUNG"),
        (2, 8, "CHÍNH SÁCH HỖ TRỢ TẠO VIỆC LÀM"),
        (3, 16, "ĐĂNG KÝ LAO ĐỘNG"),
        (4, 19, "HỆ THỐNG THÔNG TIN THỊ TRƯỜNG LAO ĐỘNG"),
        (5, 22, "PHÁT TRIỂN KỸ NĂNG NGHỀ"),
        (6, 27, "DỊCH VỤ VIỆC LÀM"),
        (7, 29, "BẢO HIỂM THẤT NGHIỆP"),
        (8, 53, "ĐIỀU KHOẢN THI HÀNH"),
    ],
}
LEGAL_CHAPTER_KEY_ALIASES = {
    "45/2019/QH14": "bo_luat_lao_dong_2019",
    "74/2025/QH15": "luat_viec_lam_2025",
}


def _normalize_roman_token(token: str) -> str:
    token = re.sub(r"\s+", "", token or "").upper()
    token = token.replace("Ị", "I").replace("Ỉ", "I").replace("Ì", "I")
    return OCR_ROMAN_FIXES.get(token, token)


def _normalize_chapter_sequence(text: str) -> str:
    last_chapter = 0
    normalized_lines: list[str] = []
    chapter_line_re = re.compile(
        rf"^(\s*Chương\s+)({CHAPTER_NUMBER_PATTERN})(?=\s|$)(.*)$",
        re.IGNORECASE,
    )

    for line in text.split("\n"):
        match = chapter_line_re.match(line)
        if not match:
            normalized_lines.append(line)
            continue

        roman = _normalize_roman_token(match.group(2))
        chapter_number = ROMAN_TO_INT.get(roman, 0)
        if chapter_number and last_chapter and chapter_number <= last_chapter and last_chapter + 1 in INT_TO_ROMAN:
            chapter_number = last_chapter + 1
            roman = INT_TO_ROMAN[chapter_number]
        if chapter_number:
            last_chapter = chapter_number

        normalized_lines.append(f"{match.group(1)}{roman}{match.group(3)}")

    return "\n".join(normalized_lines)


def repair_chapter_headings_by_article(text: str, document_key: str) -> str:
    """Insert canonical chapter headings when OCR loses a chapter marker.

    The two demo legal documents have stable article ranges per chapter. This
    fallback keeps chapter metadata correct for chunking even when Tesseract
    drops or distorts an isolated chapter heading.
    """
    canonical_key = LEGAL_CHAPTER_KEY_ALIASES.get((document_key or "").strip(), document_key)
    chapter_specs = LEGAL_CHAPTER_BOUNDARIES.get(canonical_key)
    if not chapter_specs:
        return text

    normalized_text = normalize_ocr_legal_markers(text or "")
    present_chapters = set(find_chapter_numbers(normalized_text))
    missing_by_article = {
        start_article: (chapter_number, title)
        for chapter_number, start_article, title in chapter_specs
        if chapter_number not in present_chapters
    }
    if not missing_by_article:
        return normalized_text

    repaired_lines: list[str] = []
    inserted_chapters: set[int] = set()
    for line in normalized_text.split("\n"):
        match = ARTICLE_HEADING_RE.match(line)
        article_number = _normalize_article_number(match.group("number")) if match else None
        if article_number in missing_by_article:
            chapter_number, title = missing_by_article[article_number]
            if chapter_number not in inserted_chapters:
                repaired_lines.append(f"Chương {INT_TO_ROMAN[chapter_number]}")
                if title:
                    repaired_lines.append(title)
                inserted_chapters.add(chapter_number)
        repaired_lines.append(line)

    return "\n".join(repaired_lines)


def normalize_ocr_legal_markers(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    replacements = {
        "\ufeff": "",
        "Diều": "Điều",
        "Điền": "Điều",
        "Điện": "Điều",
        "Diéu": "Điều",
        "Ðiều": "Điều",
        "Điêu": "Điều",
        "Đìều": "Điều",
        "D iều": "Điều",
        "Dieu": "Điều",
        "Khỏan": "Khoản",
        "Khoân": "Khoản",
        "Chưong": "Chương",
        "Chươmg": "Chương",
        "Chuong": "Chương",
        "Muc": "Mục",
        "QH1S": "QH15",
        "QH l5": "QH15",
        "QH I5": "QH15",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"(?im)^(\s*Điều\s+)J(\d{1,2})(\s*[\.,:])", r"\g<1>7\2\3", text)
    text = re.sub(r"(?im)^(\s*Điều\s+)114(\.\s+Nghỉ hằng năm\b)", r"\g<1>113\2", text)

    marker_words = rf"(?:{ARTICLE_WORD_PATTERN})"
    text = re.sub(rf"\s+(?=[“\"']?{marker_words}\s+(?:\d{{1,3}}|J\d{{1,2}})\s*[\.,:])", "\n", text)
    text = re.sub(rf"\s+(?=(?:Chương|Chuong)\s+{CHAPTER_NUMBER_PATTERN}(?=\s|$))", "\n", text)
    text = re.sub(r"\s+(?=(?:Mục|Muc)\s+\d+)", "\n", text)

    return _normalize_chapter_sequence(text)


def clean_legal_text(text: str) -> str:
    text = normalize_ocr_legal_markers(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    raw_lines = [line.strip() for line in text.split("\n")]

    lines: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", line):
            continue
        lines.append(line)

    marker_re = re.compile(
        rf"^(?:Chương\s+{CHAPTER_NUMBER_PATTERN}(?=\s|$)|Mục\s+\d+|[“\"']?Điều\s+(?:\d{{1,3}}|J\d{{1,2}})\s*[\.,:]?|\d+\.|[a-zđ]\))",
        re.IGNORECASE,
    )

    merged: list[str] = []
    for line in lines:
        starts_marker = bool(marker_re.match(line))
        if not merged or starts_marker:
            merged.append(line)
            continue

        previous = merged[-1]
        previous_is_heading = bool(re.match(r"^(?:Chương|Mục|[“\"']?Điều)\s+", previous, re.IGNORECASE))
        if previous_is_heading:
            merged.append(line)
        else:
            merged[-1] = f"{previous} {line}"

    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_prepared_header(text: str) -> str:
    """Remove the metadata header from a prepared text file, if present."""
    lines = text.splitlines()
    saw_header = False
    for i, line in enumerate(lines[:80]):
        stripped = line.strip()
        if not stripped and saw_header:
            return "\n".join(lines[i + 1 :])
        if ":" in stripped and stripped.split(":", 1)[0].isupper():
            saw_header = True
            continue
        break
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_header(metadata: dict[str, Any]) -> str:
    header_order = [
        "so_hieu_van_ban",
        "ten_van_ban",
        "loai_van_ban",
        "nam_ban_hanh",
        "trang_thai",
        "tinh_trang_hieu_luc",
        "co_quan_ban_hanh",
        "ngay_ban_hanh",
        "ngay_hieu_luc",
        "linh_vuc",
        "source_url",
        "file_url",
        "file_source",
        "ocr_used",
        "ocr_engine",
        "ocr_languages",
        "ocr_quality",
        "index_ready",
        "original_filename",
        "original_sha256",
    ]
    lines: list[str] = []
    for key in header_order:
        value = metadata.get(key, "")
        if isinstance(value, bool):
            value = "true" if value else "false"
        lines.append(f"{key.upper()}: {value}")
    return "\n".join(lines)


def prepare_pdf_document(
    config: LegalPdfConfig,
    root_dir: str | Path = ".",
    overwrite: bool = False,
    force_ocr: bool = False,
    mark_reviewed: bool = False,
) -> dict[str, Any]:
    dirs = ensure_legal_corpus_dirs(root_dir)
    input_pdf = resolve_input_pdf(config, root_dir)
    original_pdf = copy_original_pdf(input_pdf, dirs["raw_pdf"], overwrite=overwrite)

    output_pdf = dirs["ocr_pdf"] / f"{config.slug}.pdf"
    output_txt = dirs["text"] / f"{config.slug}.txt"
    metadata_json = dirs["metadata"] / f"{config.slug}.metadata.json"

    text_layer = extract_pdf_text_layer(original_pdf)
    use_text_layer = text_layer_is_usable(
        text_layer,
        min_article_count=config.min_article_count,
    ) and not force_ocr

    ocr_used = False
    raw_text = text_layer
    if not use_text_layer:
        if output_txt.exists() and output_pdf.exists() and not overwrite:
            raw_text = strip_prepared_header(output_txt.read_text(encoding="utf-8", errors="ignore"))
            ocr_used = True
        else:
            run_ocrmypdf(
                input_pdf=original_pdf,
                output_pdf=output_pdf,
                sidecar_txt=output_txt,
                force_ocr=force_ocr,
            )
            raw_text = output_txt.read_text(encoding="utf-8", errors="ignore")
            ocr_used = True
    elif output_pdf.exists() is False:
        shutil.copy2(original_pdf, output_pdf)

    cleaned_text = repair_chapter_headings_by_article(
        clean_legal_text(raw_text),
        config.slug,
    )
    article_count = count_articles(cleaned_text)
    quality_status = "reviewed" if mark_reviewed else "needs_review"

    metadata = dict(config.metadata)
    metadata.update(
        {
            "ocr_used": ocr_used,
            "ocr_engine": "ocrmypdf+tesseract" if ocr_used else "text_layer",
            "ocr_languages": "vie+eng" if ocr_used else "",
            "ocr_quality": quality_status,
            "index_ready": quality_status == "reviewed",
            "original_filename": original_pdf.name,
            "original_sha256": sha256_file(original_pdf),
            "prepared_text_path": str(output_txt),
            "searchable_pdf_path": str(output_pdf),
            "article_count": article_count,
            "chapter_count": count_chapters(cleaned_text),
            "min_article_count": config.min_article_count,
            "quality_passed": article_count >= config.min_article_count,
        }
    )

    output_txt.write_text(
        metadata_header(metadata) + "\n\n" + cleaned_text + "\n",
        encoding="utf-8",
    )
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "[PDF_PREPROCESS] Prepared {}: {} articles, OCR used={}",
        config.slug,
        article_count,
        ocr_used,
    )
    return metadata


def prepare_default_legal_pdfs(
    root_dir: str | Path = ".",
    overwrite: bool = False,
    force_ocr: bool = False,
    mark_reviewed: bool = False,
) -> list[dict[str, Any]]:
    reports = []
    for config in DEFAULT_LEGAL_PDFS:
        reports.append(
            prepare_pdf_document(
                config=config,
                root_dir=root_dir,
                overwrite=overwrite,
                force_ocr=force_ocr,
                mark_reviewed=mark_reviewed,
            )
        )
    return reports


def find_prepared_text_for_pdf(pdf_path: str | Path, root_dir: str | Path = ".") -> Path | None:
    pdf_name = Path(pdf_path).name
    metadata_dir = Path(root_dir) / "data" / "processed" / "metadata"
    if not metadata_dir.exists():
        return None
    for metadata_file in metadata_dir.glob("*.metadata.json"):
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if metadata.get("original_filename") == pdf_name:
            candidate = Path(metadata.get("prepared_text_path", ""))
            if candidate.exists():
                return candidate
    return None
