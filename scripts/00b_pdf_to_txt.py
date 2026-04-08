# -*- coding: utf-8 -*-
"""
PDF -> TXT ROBUST (ưu tiên không vỡ font - cải thiện xử lý encoding + OCR cho scanned PDF)
Thứ tự:
  1) PyMuPDF (fitz) - tốt nhất cho tiếng Việt
  2) Poppler pdftotext (thử nhiều encoding)
  3) pdfplumber - tốt cho layout phức tạp
  4) pypdfium2 (extract text)
  5) Tự động phát hiện scanned PDF -> chuyển sang OCR
  6) PaddleOCR (NHANH NHẤT - Model 0.9B, hỗ trợ 109 ngôn ngữ)
  7) OCRmyPDF (force-ocr) -> pdftotext
  8) pdf2image + Tesseract (OCR trực tiếp - không cần poppler)

Usage:
  python scripts/00b_pdf_to_txt.py --pdf "D:/path/file.pdf" --out "data/interim/analysis"
  python scripts/00b_pdf_to_txt.py --pdf-dir "D:/pdfs" --out "data/interim/analysis"
  python scripts/00b_pdf_to_txt.py --pdf "file.pdf" --verbose  # Hiển thị chi tiết
  python scripts/00b_pdf_to_txt.py --pdf "file.pdf" --force-ocr  # Bỏ qua text extraction, chỉ OCR
Env/Tools (khuyên cài):
  - PyMuPDF: pip install pymupdf (tốt nhất cho tiếng Việt)
  - Poppler: pdftotext trong PATH (choco install poppler) - optional
  - pdfplumber: pip install pdfplumber
  - pypdfium2: pip install pypdfium2
  - OCR: 
    * PaddleOCR (NHANH NHẤT): pip install paddlepaddle paddleocr
    * Tesseract: pip install pytesseract pillow (Tesseract OCR engine)
      (cài gói ngôn ngữ vie cho Tesseract: tesseract-ocr-vie)
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import chardet
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------- Utils ----------
# Character mapping để fix các ký tự thường bị lỗi
_CHAR_MAPPING = {
    '\x92': "'",  # Right single quotation mark
    '\x93': '"',  # Left double quotation mark
    '\x94': '"',  # Right double quotation mark
    '\x96': '-',  # En dash
    '\x97': '--',  # Em dash
    '\xa0': ' ',  # Non-breaking space
    '\xad': '',   # Soft hyphen
}

def fix_common_encoding_issues(text: str) -> str:
    """Fix các ký tự thường bị lỗi khi convert encoding."""
    for old, new in _CHAR_MAPPING.items():
        text = text.replace(old, new)
    return text

def detect_encoding(text_bytes: bytes) -> Tuple[str, float]:
    """Detect encoding của text bytes."""
    try:
        result = chardet.detect(text_bytes)
        if result and result['encoding']:
            return result['encoding'], result.get('confidence', 0.0)
    except Exception:
        pass
    return 'utf-8', 0.0

def decode_with_fallbacks(text_bytes: bytes, encodings: List[str] = None) -> Optional[str]:
    """Thử decode với nhiều encoding khác nhau."""
    if encodings is None:
        encodings = ['utf-8', 'utf-8-sig', 'cp1258', 'windows-1252', 'latin-1', 'cp1252']
    
    # Detect encoding trước
    detected_enc, confidence = detect_encoding(text_bytes)
    if confidence > 0.7:
        encodings.insert(0, detected_enc)
    
    for encoding in encodings:
        try:
            decoded = text_bytes.decode(encoding)
            # Kiểm tra chất lượng
            if not text_quality_bad(decoded):
                return decoded
        except (UnicodeDecodeError, LookupError):
            continue
    
    # Fallback: decode với errors='replace'
    try:
        return text_bytes.decode('utf-8', errors='replace')
    except Exception:
        return None

def norm_text(t: str) -> str:
    """Normalize text với xử lý encoding tốt hơn."""
    if not t:
        return ""
    
    # Fix common encoding issues
    t = fix_common_encoding_issues(t)
    
    # Normalize Unicode
    t = unicodedata.normalize("NFC", t)
    
    # Replace non-breaking spaces
    t = t.replace("\u00A0", " ")
    t = t.replace("\u200B", "")  # Zero-width space
    
    # Fix line endings
    t = re.sub(r"\r\n?", "\n", t)
    
    # Clean up whitespace
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    
    # Remove control characters (except newlines and tabs)
    t = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", t)
    
    return t.strip()

def remove_watermarks(text: str) -> str:
    """Loại bỏ watermark/header lặp lại phổ biến."""
    if not text:
        return ""
    
    # Các pattern watermark phổ biến
    watermark_patterns = [
        r"Downloaded by.*?@.*?\.com\)",
        r"lOMoARcPSD\|\d+",
        r"Scan to open on (Studeersnel|Studocu)",
        r"Studocu is not sponsored.*",
        r"Ebook.*?\n.*?\n.*?sư phạm.*?\n.*?Scan to open.*?\n.*?Studocu.*",
        r"Dit document is beschikbaar op.*?studeersnel",  # Watermark tiếng Hà Lan
        r"Dit documert is beschikbaar op.*?studeersnel",  # Typo trong watermark
    ]
    
    cleaned = text
    for pattern in watermark_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Loại bỏ dòng chỉ có email/watermark/số trang đơn độc
    lines = cleaned.split("\n")
    filtered_lines = []
    prev_line = ""
    for line in lines:
        line_stripped = line.strip()
        
        # Bỏ qua dòng quá ngắn
        if len(line_stripped) < 3:
            # Giữ lại nếu là số trang hợp lý (1-3 chữ số)
            if re.match(r"^\d{1,3}$", line_stripped):
                # Chỉ giữ nếu dòng trước đó không phải là số trang
                if not re.match(r"^\d{1,3}$", prev_line.strip()):
                    continue  # Có thể là số trang, bỏ qua
            continue
        
        # Bỏ qua email
        if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", line_stripped):
            continue
        
        # Bỏ qua ID ngắn (chỉ có chữ và số)
        if re.match(r"^[A-Za-z0-9|]+$", line_stripped) and len(line_stripped) < 20:
            continue
        
        # Bỏ qua dòng chỉ có tên tác giả lặp lại (PHAN NGOC, PHAN NGỌC)
        if re.match(r"^(PHAN|PHẠM|NGUYỄN|NGỌC)\s+(NGỌC|PHẠM|NGUYỄN|DU)?$", line_stripped, re.IGNORECASE):
            # Chỉ giữ nếu không phải là dòng lặp lại
            if prev_line.strip().upper() == line_stripped.upper():
                continue
        
        # Bỏ qua dòng chỉ có số trang (vd: "206", "207")
        if re.match(r"^\d{2,4}\s*$", line_stripped):
            # Nếu dòng trước đó cũng là số trang -> bỏ qua
            if re.match(r"^\d{2,4}\s*$", prev_line.strip()):
                continue
        
        filtered_lines.append(line)
        prev_line = line
    
    return "\n".join(filtered_lines)

def fix_ocr_errors(text: str) -> str:
    """Sửa các lỗi OCR phổ biến trong tiếng Việt."""
    if not text:
        return ""
    
    # Mapping các lỗi OCR phổ biến (theo thứ tự ưu tiên)
    ocr_fixes = [
        # Ký tự bị nhầm (7, ?, g4p)
        (r'\b7ruyện\b', 'Truyện'),
        (r'\b7zuyện\b', 'Truyện'),
        (r'\b\?zuyện\b', 'Truyện'),
        (r'\b\?ruyện\b', 'Truyện'),
        (r'\bg4p\b', 'gặp'),
        (r'\bG4p\b', 'Gặp'),
        
        # Dấu câu bị nhầm (d -> đ)
        (r'\bdé\b', 'đề'), (r'\bDé\b', 'Đề'),
        (r'\bdân\b', 'đàn'), (r'\bDân\b', 'Đàn'),
        (r'\bdần\b', 'đàn'), (r'\bDần\b', 'Đàn'),
        (r'\bdang\b', 'đang'), (r'\bDang\b', 'Đang'),
        (r'\bdại\b', 'đại'), (r'\bDại\b', 'Đại'),
        (r'\bdoi\b', 'đòi'), (r'\bDoi\b', 'Đòi'),
        (r'\bdo\b', 'đó'), (r'\bDo\b', 'Đó'),
        (r'\bdối\b', 'đối'), (r'\bDối\b', 'Đối'),
        (r'\bdau\b', 'đâu'), (r'\bDau\b', 'Đâu'),
        (r'\bday\b', 'đây'), (r'\bDay\b', 'Đây'),
        (r'\bdẫn\b', 'đẫn'), (r'\bDẫn\b', 'Đẫn'),
        (r'\bdắt\b', 'đắt'), (r'\bDắt\b', 'Đắt'),
        (r'\bdặt\b', 'đặt'), (r'\bDặt\b', 'Đặt'),
        (r'\bdạt\b', 'đạt'), (r'\bDạt\b', 'Đạt'),
        (r'\bdạy\b', 'đạy'), (r'\bDạy\b', 'Đạy'),
        (r'\bdẩy\b', 'đẩy'), (r'\bDẩy\b', 'Đẩy'),
        (r'\bdấu\b', 'đấu'), (r'\bDấu\b', 'Đấu'),
        (r'\bdầu\b', 'đầu'), (r'\bDầu\b', 'Đầu'),
        (r'\bdầy\b', 'đầy'), (r'\bDầy\b', 'Đầy'),
        (r'\bdậy\b', 'đậy'), (r'\bDậy\b', 'Đậy'),
        (r'\bdấy\b', 'đấy'), (r'\bDấy\b', 'Đấy'),
        
        # Từ bị nhầm (u -> ư, a -> ă, i -> ĩ)
        (r'\btu tưởng\b', 'tư tưởng'), (r'\bTu tưởng\b', 'Tư tưởng'),
        (r'\btình vêu\b', 'tình yêu'), (r'\bTình vêu\b', 'Tình yêu'),
        (r'\btinh yêu\b', 'tình yêu'), (r'\bTinh yêu\b', 'Tình yêu'),
        (r'\bnghia\b', 'nghĩa'), (r'\bNghia\b', 'Nghĩa'),
        (r'\bvan\b', 'văn'), (r'\bVan\b', 'Văn'),
        (r'\bvan hoc\b', 'văn học'), (r'\bVan hoc\b', 'Văn học'),
        (r'\bcan\b', 'căn'), (r'\bCan\b', 'Căn'),
        (r'\bcan cứ\b', 'căn cứ'), (r'\bCan cứ\b', 'Căn cứ'),
        (r'\bcháng\b', 'chẳng'), (r'\bCháng\b', 'Chẳng'),
        (r'\bchang\b', 'chẳng'), (r'\bChang\b', 'Chẳng'),
    ]
    
    # Áp dụng các fix theo thứ tự
    cleaned = text
    for pattern, replacement in ocr_fixes:
        cleaned = re.sub(pattern, replacement, cleaned)
    
    # Sửa lỗi khoảng trắng và dòng trống (sau khi fix các từ)
    cleaned = re.sub(r' +', ' ', cleaned)  # Nhiều khoảng trắng thành 1
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Nhiều dòng trống thành 2
    
    return cleaned

def is_mostly_watermark(text: str) -> bool:
    """Kiểm tra xem text có phải chủ yếu là watermark/header lặp lại không."""
    if not text or len(text.strip()) < 50:
        return True
    
    # Đếm số dòng unique vs tổng số dòng
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    
    unique_lines = set(lines)
    # Nếu tỉ lệ unique lines < 30% -> có thể là watermark lặp lại
    if len(unique_lines) / len(lines) < 0.3:
        return True
    
    # Kiểm tra các từ watermark phổ biến
    watermark_keywords = [
        "downloaded by", "lomoarcpsd", "studocu", "studeersnel",
        "scan to open", "not sponsored", "kataroto2017"
    ]
    text_lower = text.lower()
    watermark_count = sum(1 for kw in watermark_keywords if kw in text_lower)
    # Nếu có quá nhiều watermark keywords -> có thể là scanned PDF
    if watermark_count >= 3:
        return True
    
    return False

def score_text_quality(text: str) -> Dict[str, float]:
    """
    Đánh giá chất lượng text từ OCR/Extraction.
    Trả về dictionary với các metrics:
    - score: điểm tổng hợp (0-1, cao hơn = tốt hơn)
    - length_score: điểm dựa trên độ dài
    - vietnamese_score: điểm dựa trên tỉ lệ ký tự tiếng Việt
    - watermark_score: điểm dựa trên tỉ lệ watermark (thấp hơn = tốt hơn)
    - error_score: điểm dựa trên tỉ lệ ký tự lỗi (thấp hơn = tốt hơn)
    """
    if not text or len(text.strip()) < 10:
        return {
            'score': 0.0,
            'length_score': 0.0,
            'vietnamese_score': 0.0,
            'watermark_score': 1.0,  # 100% watermark = bad
            'error_score': 1.0,  # 100% error = bad
        }
    
    cleaned = remove_watermarks(text)
    
    # 1. Length score (normalized, tối đa 1.0)
    length = len(cleaned.strip())
    length_score = min(1.0, length / 1000.0)  # 1000 chars = 1.0
    
    # 2. Vietnamese character score
    vietnamese_chars = sum(1 for c in cleaned if '\u00C0' <= c <= '\u1EF9' or c in 'đĐ')
    total_chars = len([c for c in cleaned if c.isalpha()])
    vietnamese_score = (vietnamese_chars / max(1, total_chars)) if total_chars > 0 else 0.0
    
    # 3. Watermark score (tỉ lệ watermark, thấp hơn = tốt hơn)
    original_length = len(text.strip())
    cleaned_length = len(cleaned.strip())
    watermark_ratio = 1.0 - (cleaned_length / max(1, original_length))
    watermark_score = watermark_ratio
    
    # 4. Error score (tỉ lệ ký tự lỗi, thấp hơn = tốt hơn)
    bad_chars = cleaned.count("") + cleaned.count("\ufffd")
    error_ratio = bad_chars / max(1, len(cleaned))
    error_score = min(1.0, error_ratio * 10)  # Scale up
    
    # 5. Non-printable score
    non_printable = sum(1 for c in cleaned if not c.isprintable() and c not in '\n\r\t ')
    non_printable_ratio = non_printable / max(1, len(cleaned))
    non_printable_score = min(1.0, non_printable_ratio * 20)
    
    # 6. Unique lines ratio (tránh lặp lại)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    unique_lines = set(lines)
    uniqueness_score = len(unique_lines) / max(1, len(lines)) if lines else 0.0
    
    # Tổng hợp điểm: weighted average
    # Ưu tiên: length > vietnamese > uniqueness > (watermark + error + non_printable)
    final_score = (
        length_score * 0.3 +
        vietnamese_score * 0.25 +
        uniqueness_score * 0.2 +
        (1.0 - watermark_score) * 0.15 +
        (1.0 - error_score) * 0.05 +
        (1.0 - non_printable_score) * 0.05
    )
    
    return {
        'score': final_score,
        'length_score': length_score,
        'vietnamese_score': vietnamese_score,
        'watermark_score': watermark_score,
        'error_score': error_score,
        'non_printable_score': non_printable_score,
        'uniqueness_score': uniqueness_score,
        'length': length,
    }

def text_quality_bad(t: str) -> bool:
    """Heuristic: rỗng, quá ngắn, hoặc nhiều dấu / ký tự rác -> coi như kém."""
    if not t or len(t.strip()) < 50:  # Tăng threshold lên 50
        return True
    
    # Loại bỏ watermark trước khi kiểm tra
    cleaned = remove_watermarks(t)
    if len(cleaned.strip()) < 50:
        return True
    
    # Kiểm tra xem có phải chủ yếu là watermark không
    if is_mostly_watermark(cleaned):
        return True
    
    # Tỉ lệ replacement char hoặc khuyết dấu
    bad_chars = cleaned.count("") + cleaned.count("\ufffd")  # Replacement character
    if bad_chars / max(1, len(cleaned)) > 0.01:  # >1% ký tự lỗi
        return True
    
    # Kiểm tra có quá nhiều ký tự không in được (trừ spaces, newlines)
    non_printable = sum(1 for c in cleaned if not c.isprintable() and c not in '\n\r\t ')
    if non_printable / max(1, len(cleaned)) > 0.05:  # >5% ký tự không in được
        return True
    
    return False

def write_out(text: str, pdf_path: Path, out_dir: Path, method: str = "unknown") -> Path:
    """Write output với logging và loại bỏ watermark + sửa lỗi OCR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / (pdf_path.stem + ".txt")
    # Loại bỏ watermark trước
    cleaned = remove_watermarks(text)
    # Sửa lỗi OCR phổ biến
    cleaned = fix_ocr_errors(cleaned)
    # Normalize text
    normalized = norm_text(cleaned)
    outp.write_text(normalized + "\n", encoding="utf-8")
    if method != "unknown":
        logger.info(f"Wrote {len(normalized)} chars using {method}")
    return outp


# ---------- Layer 0: PyMuPDF (fitz) - TỐT NHẤT CHO TIẾNG VIỆT ----------
def pymupdf_extract(pdf: Path) -> Optional[str]:
    """Extract text bằng PyMuPDF - thường tốt nhất cho tiếng Việt."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    
    try:
        doc = fitz.open(str(pdf))
        parts: List[str] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text với flags để preserve layout
            text = page.get_text("text", flags=11)  # flags: preserve layout
            if text and text.strip():
                parts.append(text.strip())
        doc.close()
        result = "\n\n".join(parts)
        return result if result.strip() else None
    except Exception as e:
        logger.debug(f"PyMuPDF failed: {e}")
        return None

# ---------- Layer 1: Poppler với nhiều encoding ----------
def have_pdftotext() -> bool:
    return shutil.which("pdftotext") is not None

def pdftotext_extract(pdf: Path, encodings: List[str] = None) -> Optional[str]:
    """Extract với pdftotext, thử nhiều encoding."""
    if not have_pdftotext():
        return None
    
    if encodings is None:
        encodings = ["UTF-8", "Latin1", "ASCII"]
    
    with tempfile.TemporaryDirectory() as td:
        for encoding in encodings:
            out_txt = Path(td) / f"out_{encoding}.txt"
            cmd = ["pdftotext", "-enc", encoding, "-layout", "-nopgbrk", str(pdf), str(out_txt)]
            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    shell=False,
                    timeout=60
                )
                if out_txt.exists():
                    # Đọc dưới dạng bytes trước
                    text_bytes = out_txt.read_bytes()
                    decoded = decode_with_fallbacks(text_bytes)
                    if decoded and not text_quality_bad(decoded):
                        return decoded
            except subprocess.TimeoutExpired:
                logger.debug(f"pdftotext timeout với encoding {encoding}")
                continue
            except Exception as e:
                logger.debug(f"pdftotext failed với encoding {encoding}: {e}")
                continue
    
    return None

# ---------- Layer 1.5: pdfplumber ----------
def pdfplumber_extract(pdf: Path) -> Optional[str]:
    """Extract text bằng pdfplumber - tốt cho layout phức tạp."""
    try:
        import pdfplumber
    except ImportError:
        return None
    
    try:
        parts: List[str] = []
        with pdfplumber.open(str(pdf)) as pdf_doc:
            for page in pdf_doc.pages:
                text = page.extract_text()
                if text and text.strip():
                    parts.append(text.strip())
        result = "\n\n".join(parts)
        return result if result.strip() else None
    except Exception as e:
        logger.debug(f"pdfplumber failed: {e}")
        return None

# ---------- Layer 3: pypdfium2 ----------
def pypdfium2_extract(pdf: Path) -> Optional[str]:
    """Extract text bằng pypdfium2."""
    try:
        import pypdfium2 as pdfium  # pip install pypdfium2
    except ImportError:
        return None
    
    try:
        doc = pdfium.PdfDocument(str(pdf))
        parts: List[str] = []
        for i in range(len(doc)):
            page = doc.get_page(i)
            txtpage = page.get_textpage()
            text = txtpage.get_text_bounded()
            if text and text.strip():
                parts.append(text.strip())
            txtpage.close()
            page.close()
        doc.close()
        result = "\n\n".join(parts)
        return result if result.strip() else None
    except Exception as e:
        logger.debug(f"pypdfium2 failed: {e}")
        return None

# ---------- Layer 4: OCRmyPDF ----------
def have_ocrmypdf() -> bool:
    return shutil.which("ocrmypdf") is not None

def ocrmypdf_then_pdftotext(pdf: Path) -> Optional[str]:
    """OCR PDF rồi extract text."""
    if not have_ocrmypdf() or not have_pdftotext():
        return None
    
    with tempfile.TemporaryDirectory() as td:
        ocr_pdf = Path(td) / "ocr.pdf"
        # --- tạo OCR layer (force) ---
        cmd1 = [
            "ocrmypdf", 
            "--force-ocr", 
            "--language", "vie+eng", 
            "--deskew", 
            "--rotate-pages",
            "--quiet",  # Giảm output
            str(pdf), 
            str(ocr_pdf)
        ]
        try:
            subprocess.run(
                cmd1, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=False,
                timeout=300  # 5 phút timeout
            )
        except subprocess.TimeoutExpired:
            logger.debug("ocrmypdf timeout")
            return None
        except Exception as e:
            logger.debug(f"ocrmypdf failed: {e}")
            return None
        
        # --- trích text từ PDF đã OCR ---
        out_txt = Path(td) / "out.txt"
        cmd2 = ["pdftotext", "-enc", "UTF-8", "-layout", "-nopgbrk", str(ocr_pdf), str(out_txt)]
        try:
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            if out_txt.exists():
                text_bytes = out_txt.read_bytes()
                decoded = decode_with_fallbacks(text_bytes)
                return decoded
        except Exception as e:
            logger.debug(f"pdftotext sau OCR failed: {e}")
            return None
    
    return None

# ---------- Layer 4.5: PaddleOCR (NHANH NHẤT - Model 0.9B, hỗ trợ 109 ngôn ngữ) ----------
def paddleocr_extract(pdf: Path, dpi=200, lang="vi") -> Optional[str]:
    """OCR bằng PaddleOCR - nhanh hơn Tesseract nhiều lần, hỗ trợ tiếng Việt."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("  PaddleOCR chưa được cài. Chạy: pip install paddlepaddle paddleocr")
        return None
    
    try:
        # Khởi tạo PaddleOCR (chỉ 1 lần, reuse sau)
        if not hasattr(paddleocr_extract, '_ocr'):
            logger.info("  Đang khởi tạo PaddleOCR (lần đầu hơi chậm)...")
            paddleocr_extract._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)
        
        # Convert PDF sang images
        import fitz  # PyMuPDF
        logger.info(f"  Đang convert PDF sang images bằng PyMuPDF (DPI={dpi})...")
        doc = fitz.open(str(pdf))
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # DPI thấp hơn để nhanh hơn
            pix = page.get_pixmap(matrix=mat)
            from PIL import Image
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        doc.close()
        logger.info(f"  Đã convert {len(images)} trang, bắt đầu OCR bằng PaddleOCR...")
        
        # OCR từng trang
        parts: List[str] = []
        for i, im in enumerate(images):
            try:
                # PaddleOCR trả về list of [bbox, (text, confidence)]
                result = paddleocr_extract._ocr.ocr(im, cls=True)
                
                # Extract text từ kết quả
                page_text = []
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]  # Text content
                            if text and text.strip():
                                page_text.append(text.strip())
                
                if page_text:
                    parts.append("\n".join(page_text))
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Đã OCR {i + 1}/{len(images)} trang...")
            
            except Exception as e:
                logger.debug(f"PaddleOCR failed cho trang {i+1}: {e}")
                continue
        
        result = "\n\n".join(parts)
        cleaned = remove_watermarks(result)
        return cleaned if cleaned.strip() else None
    
    except Exception as e:
        logger.debug(f"PaddleOCR failed: {e}")
        return None

# ---------- Layer 5: pdf2image + Tesseract (CẢI THIỆN CHO TIẾNG VIỆT) ----------
def ocr_direct_each_page(pdf: Path, dpi=200, lang="vie+eng") -> Optional[str]:
    """OCR trực tiếp từng trang bằng Tesseract với settings tốt hơn cho tiếng Việt."""
    try:
        import pytesseract
    except ImportError:
        logger.warning("  pytesseract chưa được cài. Chạy: pip install pytesseract pillow")
        return None
    
    # Kiểm tra và tự động detect Tesseract path
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
    ]
    
    # Tìm Tesseract trong PATH hoặc các đường dẫn phổ biến
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            logger.debug(f"  Tìm thấy Tesseract tại: {path}")
            break
    
    # Kiểm tra Tesseract có hoạt động không
    try:
        langs = pytesseract.get_languages()
        if not tesseract_found:
            logger.debug("  Tesseract tìm thấy trong PATH")
        if lang.split("+")[0] not in langs:
            logger.warning(f"  Ngôn ngữ '{lang.split('+')[0]}' chưa được cài. Chỉ có: {', '.join(langs[:5])}...")
    except Exception as e:
        logger.warning(f"  Tesseract OCR chưa được cài hoặc không có trong PATH.")
        logger.warning(f"  Hướng dẫn cài đặt:")
        logger.warning(f"    1. Download từ: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.warning(f"    2. Cài đặt và chọn Vietnamese language pack")
        return None
    
    # Thử dùng PyMuPDF để convert PDF sang images (không cần poppler)
    # Giảm DPI xuống 200 để nhanh hơn (từ 300)
    images = None
    try:
        import fitz  # PyMuPDF
        logger.info(f"  Đang convert PDF sang images bằng PyMuPDF (DPI={dpi})...")
        doc = fitz.open(str(pdf))
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page sang image với DPI thấp hơn để nhanh hơn
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor
            pix = page.get_pixmap(matrix=mat)
            # Convert sang PIL Image
            from PIL import Image
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        doc.close()
        logger.info(f"  Đã convert {len(images)} trang, bắt đầu OCR...")
    except Exception as e:
        logger.debug(f"PyMuPDF convert failed: {e}, thử pdf2image...")
        # Fallback: dùng pdf2image (cần poppler)
        try:
            from pdf2image import convert_from_path
            logger.info(f"  Đang convert PDF sang images bằng pdf2image (DPI={dpi})...")
            images = convert_from_path(str(pdf), dpi=dpi, fmt='png')
            logger.info(f"  Đã convert {len(images)} trang, bắt đầu OCR...")
        except Exception as e2:
            logger.debug(f"pdf2image failed: {e2}")
            return None
    
    if not images:
        return None
    
    try:
        parts: List[str] = []
        # Sử dụng PSM 6 (nhanh hơn, chỉ thử 1 config thay vì 2)
        config = '--psm 6 -c preserve_interword_spaces=1'
        
        for i, im in enumerate(images):
            try:
                # Chỉ thử 1 config để nhanh hơn
                txt = pytesseract.image_to_string(im, lang=lang, config=config)
                if txt and txt.strip():
                    parts.append(txt.strip())
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Đã OCR {i + 1}/{len(images)} trang...")
            
            except Exception as e:
                logger.debug(f"Tesseract failed cho trang {i+1}: {e}")
                continue
        
        result = "\n\n".join(parts)
        cleaned = remove_watermarks(result)
        return cleaned if cleaned.strip() else None
    except Exception as e:
        logger.debug(f"OCR direct failed: {e}")
        return None

# ---------- Ensemble OCR: Kết hợp nhiều OCR engines ----------
def ensemble_ocr(pdf: Path, methods: List[Tuple[str, Any]], verbose: bool = False) -> Optional[str]:
    """
    Kết hợp kết quả từ nhiều OCR engines bằng voting/confidence scoring.
    
    Args:
        pdf: Path to PDF file
        methods: List of (method_name, extract_func) tuples
        verbose: Verbose logging
    
    Returns:
        Best combined text result
    """
    results: List[Tuple[str, str, Dict[str, float]]] = []  # (method, text, scores)
    
    # Chạy tất cả OCR methods
    for method_name, extract_func in methods:
        try:
            if verbose:
                logger.info(f"  Đang chạy {method_name}...")
            text = extract_func(pdf)
            if text and text.strip():
                scores = score_text_quality(text)
                results.append((method_name, text, scores))
                if verbose:
                    logger.info(f"  {method_name}: score={scores['score']:.3f}, length={scores['length']}")
        except Exception as e:
            if verbose:
                logger.debug(f"  {method_name} failed: {e}")
            continue
    
    if not results:
        return None
    
    # Sắp xếp theo score
    results.sort(key=lambda x: x[2]['score'], reverse=True)
    
    # Strategy 1: Chọn kết quả tốt nhất (highest score)
    best_method, best_text, best_scores = results[0]
    
    # Strategy 2: Nếu có nhiều kết quả tốt tương đương, merge chúng
    # Lấy tất cả kết quả có score > 80% của best score
    threshold = best_scores['score'] * 0.8
    good_results = [r for r in results if r[2]['score'] >= threshold]
    
    if len(good_results) > 1 and best_scores['score'] < 0.7:
        # Merge multiple results bằng cách:
        # 1. Lấy text dài nhất làm base
        # 2. Bổ sung các phần unique từ các text khác
        if verbose:
            logger.info(f"  Merging {len(good_results)} results...")
        
        base_text = best_text
        base_lines = set(base_text.split("\n"))
        
        for method_name, text, scores in good_results[1:]:
            new_lines = set(text.split("\n"))
            # Thêm các dòng unique từ text khác
            unique_new = new_lines - base_lines
            if unique_new:
                # Thêm các dòng unique vào cuối
                base_text += "\n" + "\n".join(sorted(unique_new, key=lambda x: len(x), reverse=True)[:50])
                base_lines.update(unique_new)
        
        # Clean merged text
        merged_cleaned = remove_watermarks(base_text)
        merged_scores = score_text_quality(merged_cleaned)
        
        # Chọn giữa best single và merged
        if merged_scores['score'] > best_scores['score'] * 1.1:  # Merged tốt hơn 10%
            if verbose:
                logger.info(f"  Merged result tốt hơn: {merged_scores['score']:.3f} vs {best_scores['score']:.3f}")
            return merged_cleaned
        else:
            if verbose:
                logger.info(f"  Sử dụng best single result: {best_method}")
            return remove_watermarks(best_text)
    else:
        # Chỉ dùng best result
        if verbose:
            logger.info(f"  Sử dụng best result: {best_method} (score={best_scores['score']:.3f})")
        return remove_watermarks(best_text)

# ---------- Orchestrator ----------
def convert_pdf(pdf: Path, out_dir: Path, verbose: bool = False, force_ocr: bool = False, ensemble: bool = True) -> Path:
    """Convert PDF sang TXT với nhiều method fallback. Tự động chuyển sang OCR nếu phát hiện scanned PDF."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.info(f"Converting {pdf.name}...")
    
    # Nếu force OCR, skip text extraction methods
    if force_ocr:
        logger.info("  Force OCR mode - bỏ qua text extraction")
        # Ưu tiên PaddleOCR trước (nhanh nhất)
        methods = [
            ("PaddleOCR (Nhanh)", paddleocr_extract),
            ("OCRmyPDF", ocrmypdf_then_pdftotext),
            ("Tesseract OCR", ocr_direct_each_page),
        ]
    else:
        methods = [
            ("PyMuPDF (fitz)", pymupdf_extract),
            ("pdfplumber", pdfplumber_extract),
            ("Poppler pdftotext", lambda p: pdftotext_extract(p)),
            ("pypdfium2", pypdfium2_extract),
        ]
    
    best_text = None
    best_method = None
    needs_ocr = False
    
    # Thử text extraction methods trước
    for method_name, extract_func in methods:
        try:
            if verbose:
                logger.info(f"  Thử {method_name}...")
            txt = extract_func(pdf)
            if txt:
                # Loại bỏ watermark trước khi kiểm tra
                cleaned = remove_watermarks(txt)
                
                # Kiểm tra chất lượng
                if not text_quality_bad(cleaned) and len(cleaned.strip()) > 100:
                    best_text = cleaned
                    best_method = method_name
                    if verbose:
                        logger.info(f"  ✓ Thành công với {method_name} ({len(cleaned)} chars)")
                    break
                elif is_mostly_watermark(txt):
                    # Phát hiện scanned PDF - cần OCR
                    needs_ocr = True
                    if verbose:
                        logger.warning(f"  ✗ {method_name} chỉ extract được watermark, cần OCR")
                    break
                elif txt and best_text is None:
                    # Lưu kết quả tạm dù chất lượng kém
                    best_text = cleaned if cleaned else txt
                    best_method = method_name
        except Exception as e:
            if verbose:
                logger.debug(f"  ✗ {method_name} failed: {e}")
            continue
    
    # Nếu phát hiện scanned PDF hoặc text extraction thất bại, thử OCR
    if needs_ocr or (best_text is None or len(best_text.strip()) < 100):
        logger.info("  Chuyển sang OCR mode...")
        # Ưu tiên PaddleOCR trước (nhanh nhất), sau đó mới Tesseract
        ocr_methods = [
            ("PaddleOCR (Nhanh)", paddleocr_extract),
            ("OCRmyPDF", ocrmypdf_then_pdftotext),
            ("Tesseract OCR", ocr_direct_each_page),
        ]
        
        # Nếu ensemble mode: chạy tất cả và kết hợp
        if ensemble:
            ensemble_result = ensemble_ocr(pdf, ocr_methods, verbose=verbose)
            if ensemble_result and len(ensemble_result.strip()) > 100:
                best_text = ensemble_result
                best_method = "Ensemble OCR"
                if verbose:
                    logger.info(f"  ✓ Ensemble OCR thành công ({len(ensemble_result)} chars)")
            else:
                # Fallback: thử từng method một
                for method_name, extract_func in ocr_methods:
                    try:
                        if verbose:
                            logger.info(f"  Thử OCR: {method_name}...")
                        txt = extract_func(pdf)
                        if txt:
                            cleaned = remove_watermarks(txt)
                            if cleaned and len(cleaned.strip()) > 100:
                                best_text = cleaned
                                best_method = method_name
                                if verbose:
                                    logger.info(f"  ✓ OCR thành công với {method_name} ({len(cleaned)} chars)")
                                break
                    except Exception as e:
                        if verbose:
                            logger.debug(f"  ✗ OCR {method_name} failed: {e}")
                        continue
        else:
            # Non-ensemble: thử từng method một
            for method_name, extract_func in ocr_methods:
                try:
                    if verbose:
                        logger.info(f"  Thử OCR: {method_name}...")
                    txt = extract_func(pdf)
                    if txt:
                        cleaned = remove_watermarks(txt)
                        if cleaned and len(cleaned.strip()) > 100:
                            best_text = cleaned
                            best_method = method_name
                            if verbose:
                                logger.info(f"  ✓ OCR thành công với {method_name} ({len(cleaned)} chars)")
                            break
                except Exception as e:
                    if verbose:
                        logger.debug(f"  ✗ OCR {method_name} failed: {e}")
                    continue
    
    # Write output
    if best_text and len(best_text.strip()) > 50:
        return write_out(best_text, pdf, out_dir, method=best_method or "unknown")
    else:
        # Hết cách: vẫn lưu empty file hoặc warning
        logger.warning(f"Không thể extract text từ {pdf.name}, tạo file rỗng")
        return write_out("", pdf, out_dir, method="none")

def main():
    ap = argparse.ArgumentParser(
        description="Convert PDF sang TXT với xử lý font tốt hơn + OCR cho scanned PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/00b_pdf_to_txt.py --pdf "file.pdf" --out "output/"
  python scripts/00b_pdf_to_txt.py --pdf-dir "pdfs/" --verbose
  python scripts/00b_pdf_to_txt.py --pdf "file.pdf" --force-ocr  # Chỉ dùng OCR
  python scripts/00b_pdf_to_txt.py --pdf "file.pdf" --no-ensemble  # Tắt ensemble mode
        """
    )
    ap.add_argument("--pdf", help="Đường dẫn 1 PDF file")
    ap.add_argument("--pdf-dir", help="Thư mục chứa nhiều PDF")
    ap.add_argument("--out", default="data/interim/analysis", help="Thư mục output")
    ap.add_argument("--verbose", "-v", action="store_true", help="Hiển thị chi tiết quá trình")
    ap.add_argument("--force-ocr", action="store_true", help="Bỏ qua text extraction, chỉ dùng OCR")
    ap.add_argument("--no-ensemble", action="store_true", help="Tắt ensemble mode (chỉ dùng 1 OCR engine)")
    args = ap.parse_args()

    targets: List[Path] = []
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise SystemExit(f"File không tồn tại: {pdf_path}")
        targets = [pdf_path]
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            raise SystemExit(f"Thư mục không tồn tại: {pdf_dir}")
        targets = sorted(pdf_dir.glob("*.pdf"))
    else:
        raise SystemExit("Hãy truyền --pdf hoặc --pdf-dir")

    if not targets:
        raise SystemExit("Không tìm thấy PDF nào.")

    out_dir = Path(args.out)
    success_count = 0
    error_count = 0

    logger.info(f"Bắt đầu convert {len(targets)} file PDF...")
    
    for p in targets:
        try:
            outp = convert_pdf(p, out_dir, verbose=args.verbose, force_ocr=args.force_ocr, ensemble=not args.no_ensemble)
            print(f"[OK] {p.name} -> {outp}")
            success_count += 1
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            error_count += 1
    
    logger.info(f"Hoàn thành: {success_count} thành công, {error_count} lỗi")

if __name__ == "__main__":
    main()

