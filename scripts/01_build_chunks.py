# scripts/01_build_chunks.py
# -*- coding: utf-8 -*-
"""
Make RAG chunks from TXT files and gắn meta.tags tự động.

Input (TXT only):
  data/interim/**.txt
  ├─ poem/       # thơ (Truyện Kiều) – mỗi câu 1 dòng càng tốt
  ├─ analysis/   # bình giảng, nghị luận
  ├─ summary/    # tóm tắt, dàn ý
  └─ bio/        # tiểu sử, bối cảnh

Output:
  data/rag_chunks/*.txt   (mỗi chunk 1 file, dòng đầu là ###META### {json})

Meta bổ sung:
  meta.tags: danh sách tag phẳng, ví dụ:
    ["char:thuy_kieu", "char:thuy_van", "device:uoc_le", "theme:tai_menh", "section:trao_duyen"]
"""

from pathlib import Path
import re, json, unicodedata, hashlib

# ==== Config đường dẫn ====
SRC = Path("data/interim")
DST = Path("data/rag_chunks"); DST.mkdir(parents=True, exist_ok=True)

# Kích thước chunk
PROSE_MAX_WORDS = 220          # văn xuôi/bình giảng: 150–250 từ
POEM_LINES_PER_BLOCK = (2, 4)  # khuyến nghị: 2–4 câu/khổ
POEM_OVERLAP_LINES = 1         # overlap 1–2 câu để giữ mạch

# map thư mục -> type
TYPE_BY_DIR = {
    "poem": "poem",
    "analysis": "analysis",
    "summary": "summary",
    "bio": "bio",
}

# ==== Từ khoá/regex để gắn TAGS ====
# Lưu ý: dùng re.I (không phân biệt hoa/thường); có dấu/không dấu đều khớp tốt trên phần lớn nguồn.
CHAR_PAT = {
    "char:thuy_kieu":  [r"\bthúy?\s*kiều\b", r"\bvu(o|ơ)ng\s*thúy?\s*kiều\b"],
    "char:thuy_van":   [r"\bthúy?\s*vân\b"],
    "char:kim_trong":  [r"\bkim\s*trọng\b"],
    "char:tu_hai":     [r"\bt(ừ|u)\s*h(ả|a)i\b"],
    "char:hoan_thu":   [r"\bhoạn?\s*th(ư|u)\b"],
    "char:ma_giam_sinh":[r"\bm(ã|a)\s*gi(á|a)m\s*sinh\b"],
    "char:so_khanh":   [r"\bs(ở|o)\s*khanh\b"],
    "char:tu_ba":      [r"\bt(ú|u)\s*b(à|a)\b"],
    "char:giac_duyen": [r"\bgi(á|a)c\s*duy(ê|e)n\b"],
    "char:dam_tien":   [r"\b(đ|d)ạm\s*ti(ê|e)n\b"],
    "char:vuong_quan": [r"\bv(ư|u)o(ng)?\s*quan\b"],
}

DEVICE_PAT = {
    "device:uoc_le":        [r"\bước\s*lệ\b", r"\b(ước\s*lệ|tượng\s*trưng)\b"],
    "device:dien_co":       [r"\bđi(ể|e)n\s*c(ố|o)\b", r"\bđi(ể|e)n\s*t(ích|ich)\b"],
    "device:an_du":         [r"\bẩn\s*d(ụ|u)\b"],
    "device:nhan_hoa":      [r"\bnhân\s*h(ó|o)a\b"],
    "device:ta_canh_ngu_tinh":[r"\bt(ả|a)\s*c(ả|a)nh\s*ng(ụ|u)\s*t(ì|i)nh\b"],
    "device:phung_du":      [r"\bphúng\s*d(ụ|u)\b"],
    "device:cuc_ta":        [r"\bc(ư|u)ờng\s*đi(ễ|e)m\b", r"\bph(ó|o)ng\s*đ(ạ|a)i\s*(ch(ử|u)|ph(á|a)p)\b"],
}

THEME_PAT = {
    "theme:tai_menh":   [r"\bt(à|a)i\s*m(ệ|e)nh\b", r"\bt(à|a)i\s*v(ậ|a)n\b"],
    "theme:chu_tam":    [r"\bch(ữ|u)\s*t(â|a)m\b"],
    "theme:nhan_dao":   [r"\bnh(â|a)n\s*đ(ạ|a)o\b"],
    "theme:tinh_yeu":   [r"\bt(ì|i)nh\s*y(ê|e)u\b", r"\bd(uy|uy)ên\b"],
    "theme:so_phan":    [r"\bs(ố|o)\s*ph(ậ|a)n\b", r"\bb(ạ|a)c\s*m(ệ|e)nh\b"],
    "theme:gia_bien":   [r"\bgia\s*bi(ế|e)n\b"],
}

SECTION_PAT = {
    "section:trao_duyen":        [r"\btrao\s*duy(ê|e)n\b"],
    "section:chi_khi_anh_hung":  [r"\bch(í|i)\s*kh(í|i)\s*anh\s*h(ù|u)ng\b"],
    "section:bao_an_bao_oan":    [r"\bb(á|a)o\s*(â|a)n\b.*b(á|a)o\s*(o|ô)an\b", r"\bth(ú|u)y\s*ki(ề|e)u\s*b(á|a)o\s*(â|a)n\b"],
    "section:gap_kim_trong":     [r"\bg(ặ|a)p\s*kim\s*tr(ọ|o)ng\b"],
    "section:khoc_duong_truong": [r"\b(đ|d)o(à|a)n\s*tr(ư|u)ờng\s*t(â|a)n\s*th(à|a)nh\b"],  # tên gốc Trung Hoa
}

# ==== Helpers ====
def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\u00a0", " ")
    # gọn khoảng trắng
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def read_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def file_type(fp: Path) -> str:
    parts = list(fp.relative_to(SRC).parts)
    return TYPE_BY_DIR.get(parts[0], "analysis")

def make_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

# ==== Tag extractor ====
def _find_any(patterns, text):
    return any(re.search(p, text, flags=re.I) for p in patterns)

def extract_tags(text: str, ftype: str) -> list[str]:
    """
    QUA QUY TẮC:
      - char:* nếu xuất hiện tên nhân vật
      - device:* nếu có từ khóa thủ pháp
      - theme:* chủ đề lớn
      - section:* tên đoạn/nhan đề quen thuộc
      - type:* luôn thêm type (poem/analysis/summary/bio)
    """
    tags: set[str] = set()
    # dấu hiệu theo danh mục
    for tag, pats in CHAR_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)
    for tag, pats in DEVICE_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)
    for tag, pats in THEME_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)
    for tag, pats in SECTION_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)

    # “bonus” heuristic: nếu nhắc cả “hoa” và “liễu” trong cùng khối văn xuôi → uớc lệ
    if re.search(r"\bhoa\b", text, flags=re.I) and re.search(r"\bli(ễ|e)u\b", text, flags=re.I):
        tags.add("device:uoc_le")

    tags.add(f"type:{ftype}")
    return sorted(tags)

# ==== Chunkers ====
def split_poem(text: str, min_lines=2, max_lines=4, overlap=1) -> list[dict]:
    """
    Trả list dict:
      {"lines": "...\n...", "line_start": int, "line_end": int}
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    blocks = []
    i = 0
    n = len(lines)
    L = max_lines
    while i < n:
        j = min(i + L, n)
        blk_text = "\n".join(lines[i:j])
        blocks.append({"lines": blk_text, "line_start": i + 1, "line_end": j})
        i += max(1, (L - overlap))
    return blocks

def split_prose(text: str, max_words=PROSE_MAX_WORDS) -> list[dict]:
    """
    Trả list dict:
      {"text": "...", "char_start": int|None, "char_end": int|None}
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks, cur, cur_start = [], "", 0
    cursor = 0

    def find_span(haystack: str, needle: str, start_at: int) -> tuple[int, int]:
        pos = haystack.find(needle, start_at)
        return (pos, pos + len(needle)) if pos >= 0 else (-1, -1)

    for p in paras:
        if len((cur + " " + p).split()) <= max_words:
            if not cur:
                cur_start, _ = find_span(text, p, cursor)
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                s = text.find(cur, cur_start if cur_start >= 0 else 0)
                e = s + len(cur) if s >= 0 else -1
                blocks.append({"text": cur, "char_start": s if s >= 0 else None, "char_end": e if e >= 0 else None})
                cursor = e if e and e > 0 else cursor
                cur = p
                cur_start, _ = find_span(text, p, cursor)
            else:
                sentences = re.split(r"(?<=[\.\?\!…:;])\s+", p)
                buf, buf_start = "", None
                for snt in sentences:
                    if len((buf + " " + snt).split()) <= max_words:
                        if not buf:
                            buf_start, _ = find_span(text, snt, cursor)
                        buf = (buf + " " + snt).strip()
                    else:
                        if buf:
                            s0 = buf_start
                            e0 = s0 + len(buf) if s0 and s0 >= 0 else None
                            blocks.append({"text": buf, "char_start": s0 if s0 and s0 >= 0 else None, "char_end": e0})
                            cursor = e0 if e0 else cursor
                            buf = snt; buf_start, _ = find_span(text, snt, cursor)
                        else:
                            s1, e1 = find_span(text, snt, cursor)
                            blocks.append({"text": snt, "char_start": s1 if s1 >= 0 else None, "char_end": e1 if e1 >= 0 else None})
                            cursor = e1 if e1 else cursor
                if buf:
                    s2 = buf_start
                    e2 = s2 + len(buf) if s2 and s2 >= 0 else None
                    blocks.append({"text": buf, "char_start": s2 if s2 and s2 >= 0 else None, "char_end": e2})
                    cursor = e2 if e2 else cursor
                cur = ""
    if cur:
        s = cur_start
        e = s + len(cur) if s and s >= 0 else None
        blocks.append({"text": cur, "char_start": s if s and s >= 0 else None, "char_end": e})
    return blocks

def write_chunks(blocks: list, meta_base: dict, base_name: str):
    for idx, blk in enumerate(blocks):
        meta = dict(meta_base)
        meta["chunk_index"] = idx
        if meta_base["type"] == "poem":
            meta["line_start"] = blk["line_start"]
            meta["line_end"]   = blk["line_end"]
            content = blk["lines"].strip()
            meta["id"] = f"{meta_base['source_id']}_L{meta['line_start']:04d}-{meta['line_end']:04d}_{make_id(content)[:6]}"
        else:
            meta["char_start"] = blk.get("char_start")
            meta["char_end"]   = blk.get("char_end")
            content = blk["text"].strip()
            meta["id"] = f"{meta_base['source_id']}_{idx:04d}_{make_id(content)[:6]}"

        meta["tags"] = extract_tags(content, meta_base["type"])
        hdr = "###META### " + json.dumps(meta, ensure_ascii=False)
        outp = DST / f"{base_name}_{idx:04d}.txt"
        outp.write_text(hdr + "\n" + content + "\n", encoding="utf-8")

# ==== Main ====
def main():
    files = list(SRC.rglob("*.txt"))
    if not files:
        raise SystemExit("❗ Không tìm thấy .txt trong data/interim/. Hãy đặt file TXT vào đó rồi chạy lại.")

    total_files, total_chunks = 0, 0
    for fp in files:
        raw = read_txt(fp)
        text = normalize_text(raw)
        ftype = file_type(fp)

        rel = fp.relative_to(SRC)
        base_name = fp.stem
        meta_base = {
            "source": str(rel).replace("\\", "/"),
            "source_id": base_name,
            "type": ftype,
            "title": base_name,
        }

        if ftype == "poem":
            blocks = split_poem(text)
        else:
            blocks = split_prose(text)

        write_chunks(blocks, meta_base, base_name)
        total_files += 1
        total_chunks += len(blocks)

    print(f"✔ Done. {total_files} file -> {total_chunks} chunks.")
    print(f"   Output: {DST.resolve()}")

if __name__ == "__main__":
    main()
