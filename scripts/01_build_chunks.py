# scripts/01_build_chunks.py
# -*- coding: utf-8 -*-
"""
Build RAG chunks + vị trí:
- Thơ: line_start/line_end (đếm từ 1)
- Văn xuôi: char_start/char_end (tính theo bản đã normalize)

Input:
  data/interim/poem/*.txt     # thơ (mỗi câu 1 dòng)
  data/interim/analysis/*.txt
  data/interim/summary/*.txt
  data/interim/bio/*.txt

Output:
  data/rag_chunks/*.txt  (dòng đầu: ###META### {json})
"""
from pathlib import Path
import re, json, unicodedata, hashlib
from typing import List, Tuple, Dict, Optional

SRC = Path("data/interim")
DST = Path("data/rag_chunks"); DST.mkdir(parents=True, exist_ok=True)

PROSE_MAX_WORDS = 220
POEM_LINES_PER_BLOCK = (2, 4)
POEM_OVERLAP_LINES = 1

TYPE_BY_DIR = {
    "poem": "poem",
    "analysis": "analysis",
    "summary": "summary",
    "bio": "bio",
}

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
    "section:khoc_duong_truong": [r"\b(đ|d)o(à|a)n\s*tr(ư|u)ờng\s*t(â|a)n\s*th(à|a)nh\b"],
}

import regex as re2  # thêm ở đầu file (đã có 're' nhưng ta dùng 'regex' cho \R và overlapped)

def _relaxed_span(hay: str, needle: str, start_at: int = 0) -> tuple[int, int]:
    """
    Tìm needle trong hay, bỏ qua khác biệt khoảng trắng.
    - Quy tắc: collapse mọi whitespace trong needle thành \\s+ rồi re2.search từ vị trí start_at.
    - Trả (s, e) hoặc (-1, -1) nếu không thấy.
    """
    # escape toàn bộ, rồi thay cụm whitespace liên tiếp trong needle thành \\s+
    esc = re2.escape(needle.strip())
    esc = re2.sub(r"(\\\s)+", r"\\s+", esc)  # phòng TH needle đã có \s do escape
    esc = re2.sub(r"\\\s+", r"\\s+", esc)    # chắc ăn
    # đồng bộ chuẩn hoá khoảng trắng trong haystack vùng xét
    m = re2.search(esc, hay[start_at:], flags=re2.IGNORECASE | re2.DOTALL | re2.MULTILINE)
    if not m:
        return -1, -1
    s = start_at + m.start()
    e = start_at + m.end()
    return s, e


def split_prose(text: str, max_words=PROSE_MAX_WORDS) -> list[dict]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks, cur = [], ""
    cursor = 0            # luôn tìm tiếp từ đây
    cur_start = None

    def append_block(buf: str, s: int | None, e: int | None):
        blocks.append({
            "text": buf.strip(),
            "char_start": int(s) if (isinstance(s, int) and s >= 0) else None,
            "char_end":   int(e) if (isinstance(e, int) and e is not None and e > 0) else None
        })

    for p in paras:
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate.split()) <= max_words:
            if not cur:
                s, e = _relaxed_span(text, p, cursor)
                cur_start = s if s >= 0 else None
            cur = candidate
        else:
            if cur:
                if cur_start is None:
                    s, e = _relaxed_span(text, cur, cursor)
                else:
                    s = cur_start
                    e = s + len(cur) if s is not None and s >= 0 else None
                    if (s is None) or (s < 0):
                        s, e = _relaxed_span(text, cur, cursor)
                append_block(cur, s, e)
                cursor = e if e else cursor
                cur = p
                s, e = _relaxed_span(text, p, cursor)
                cur_start = s if s >= 0 else None
            else:
                # cắt theo câu
                sentences = re.split(r"(?<=[\.\?\!…:;])\s+", p)
                buf, buf_start = "", None
                for snt in sentences:
                    cand2 = (buf + " " + snt).strip() if buf else snt
                    if len(cand2.split()) <= max_words:
                        if not buf:
                            s0, _ = _relaxed_span(text, snt, cursor)
                            buf_start = s0 if s0 >= 0 else None
                        buf = cand2
                    else:
                        if buf:
                            if buf_start is None:
                                s1, e1 = _relaxed_span(text, buf, cursor)
                            else:
                                s1 = buf_start
                                e1 = s1 + len(buf) if s1 is not None and s1 >= 0 else None
                                if (s1 is None) or (s1 < 0):
                                    s1, e1 = _relaxed_span(text, buf, cursor)
                            append_block(buf, s1, e1)
                            cursor = e1 if e1 else cursor
                            buf = snt
                            s2, _ = _relaxed_span(text, snt, cursor)
                            buf_start = s2 if s2 >= 0 else None
                        else:
                            s3, e3 = _relaxed_span(text, snt, cursor)
                            append_block(snt, s3, e3)
                            cursor = e3 if e3 else cursor
                if buf:
                    if buf_start is None:
                        s4, e4 = _relaxed_span(text, buf, cursor)
                    else:
                        s4 = buf_start
                        e4 = s4 + len(buf) if s4 is not None and s4 >= 0 else None
                        if (s4 is None) or (s4 < 0):
                            s4, e4 = _relaxed_span(text, buf, cursor)
                    append_block(buf, s4, e4)
                    cursor = e4 if e4 else cursor
                cur = ""
    if cur:
        if cur_start is None:
            s, e = _relaxed_span(text, cur, cursor)
        else:
            s = cur_start
            e = s + len(cur) if s is not None and s >= 0 else None
            if (s is None) or (s < 0):
                s, e = _relaxed_span(text, cur, cursor)
        append_block(cur, s, e)
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

        tags = extract_tags(content, meta_base["type"])
        if not isinstance(tags, list):   # failsafe
            tags = list(tags) if isinstance(tags, set) else [str(tags)]
        meta["tags"] = tags

        hdr = "###META### " + json.dumps(meta, ensure_ascii=False)
        outp = DST / f"{base_name}_{idx:04d}.txt"
        outp.write_text(hdr + "\n" + content + "\n", encoding="utf-8")


def normalize_text_prose(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def normalize_text_poem(t: str) -> str:
    # không gom dòng, chỉ NFC + bỏ khoảng trắng cuối dòng
    t = unicodedata.normalize("NFC", t).replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+$", "", ln) for ln in t.splitlines()]
    return "\n".join(lines).strip()

def read_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def file_type(fp: Path) -> str:
    parts = list(fp.relative_to(SRC).parts)
    return TYPE_BY_DIR.get(parts[0], "analysis")

def make_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _find_any(patterns, text):
    return any(re.search(p, text, flags=re.I) for p in patterns)

def extract_tags(text: str, ftype: str) -> List[str]:
    tags: set[str] = set()
    for tag, pats in CHAR_PAT.items():
        if _find_any(pats, text): tags.add(tag)
    for tag, pats in DEVICE_PAT.items():
        if _find_any(pats, text): tags.add(tag)
    for tag, pats in THEME_PAT.items():
        if _find_any(pats, text): tags.add(tag)
    for tag, pats in SECTION_PAT.items():
        if _find_any(pats, text): tags.add(tag)
    if re.search(r"\bhoa\b", text, flags=re.I) and re.search(r"\bli(ễ|e)u\b", text, flags=re.I):
        tags.add("device:uoc_le")
    tags.add(f"type:{ftype}")
    return sorted(tags)

# ====== Chunkers ======
def split_poem(text: str, min_lines=2, max_lines=4, overlap=1) -> List[Dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks: List[Dict] = []
    i, n = 0, len(lines)
    L = max_lines
    while i < n:
        j = min(i + L, n)
        blk_text = "\n".join(lines[i:j])
        blocks.append({"lines": blk_text, "line_start": i + 1, "line_end": j})
        i += max(1, (L - overlap))
    return blocks

def split_prose(text: str, max_words=PROSE_MAX_WORDS) -> List[Dict]:
    def find_span(hay: str, needle: str, start_at: int) -> Tuple[int, int]:
        pos = hay.find(needle, start_at)
        return (pos, pos + len(needle)) if pos >= 0 else (-1, -1)

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks: List[Dict] = []
    cur, cur_start = "", -1
    cursor = 0

    for p in paras:
        if len((cur + " " + p).split()) <= max_words:
            if not cur:
                cur_start, _ = find_span(text, p, cursor)
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                s = text.find(cur, cur_start if cur_start >= 0 else 0)
                e = s + len(cur) if s >= 0 else -1
                blocks.append({"text": cur, "char_start": s if s >= 0 else -1, "char_end": e if e >= 0 else -1})
                cursor = e if e >= 0 else cursor
                cur = p
                cur_start, _ = find_span(text, p, cursor)
            else:
                sentences = re.split(r"(?<=[\.\?\!…:;])\s+", p)
                buf, buf_start = "", -1
                for snt in sentences:
                    if len((buf + " " + snt).split()) <= max_words:
                        if not buf:
                            buf_start, _ = find_span(text, snt, cursor)
                        buf = (buf + " " + snt).strip()
                    else:
                        if buf:
                            s0 = buf_start
                            e0 = s0 + len(buf) if s0 >= 0 else -1
                            blocks.append({"text": buf, "char_start": s0 if s0 >= 0 else -1, "char_end": e0 if e0 >= 0 else -1})
                            cursor = e0 if e0 >= 0 else cursor
                            buf = snt; buf_start, _ = find_span(text, snt, cursor)
                        else:
                            s1, e1 = find_span(text, snt, cursor)
                            blocks.append({"text": snt, "char_start": s1 if s1 >= 0 else -1, "char_end": e1 if e1 >= 0 else -1})
                            cursor = e1 if e1 >= 0 else cursor
                if buf:
                    s2 = buf_start
                    e2 = s2 + len(buf) if s2 >= 0 else -1
                    blocks.append({"text": buf, "char_start": s2 if s2 >= 0 else -1, "char_end": e2 if e2 >= 0 else -1})
                    cursor = e2 if e2 >= 0 else cursor
                cur = ""
    if cur:
        s = cur_start
        e = s + len(cur) if s >= 0 else -1
        blocks.append({"text": cur, "char_start": s if s >= 0 else -1, "char_end": e if e >= 0 else -1})
    return blocks

def write_chunks(blocks: List[Dict], meta_base: Dict, base_name: str):
    for idx, blk in enumerate(blocks):
        meta = dict(meta_base)
        meta["chunk_index"] = idx
        if meta_base["type"] == "poem":
            content = blk["lines"].strip()
            meta["line_start"] = int(blk["line_start"])
            meta["line_end"] = int(blk["line_end"])
            meta["id"] = f"{meta_base['source_id']}_L{meta['line_start']:04d}-{meta['line_end']:04d}_{make_id(content)[:6]}"
        else:
            content = blk["text"].strip()
            meta["char_start"] = int(blk.get("char_start", -1))
            meta["char_end"] = int(blk.get("char_end", -1))
            meta["id"] = f"{meta_base['source_id']}_{idx:04d}_{make_id(content)[:6]}"

        meta["tags"] = extract_tags(content, meta_base["type"])
        hdr = "###META### " + json.dumps(meta, ensure_ascii=False)
        outp = DST / f"{base_name}_{idx:04d}.txt"
        outp.write_text(hdr + "\n" + content + "\n", encoding="utf-8")

def main():
    files = list(SRC.rglob("*.txt"))
    if not files:
        raise SystemExit("❗ Không tìm thấy .txt trong data/interim/.")

    total_files, total_chunks = 0, 0
    for fp in files:
        ftype = file_type(fp)
        raw = read_txt(fp)
        text = normalize_text_poem(raw) if ftype == "poem" else normalize_text_prose(raw)

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
