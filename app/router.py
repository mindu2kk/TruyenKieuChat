# app/router.py
# -*- coding: utf-8 -*-
import re

# mở rộng từ số tiếng Việt hay dùng
_VI_NUM = {
    "một":1, "hai":2, "ba":3, "bốn":4, "năm":5, "sáu":6, "bảy":7, "tám":8, "chín":9,
    "mười":10, "mươi":10, "mười một":11, "mười hai":12, "mười ba":13, "mười bốn":14, "mười lăm":15,
    "mười sáu":16, "mười bảy":17, "mười tám":18, "mười chín":19,
    "hai mươi":20, "ba mươi":30, "bốn mươi":40, "năm mươi":50
}

def _normalize_vi_number_words(q: str) -> str:
    qs = q
    # thay các cụm dài trước (vd: "hai mươi") rồi tới đơn
    for w in sorted(_VI_NUM.keys(), key=len, reverse=True):
        qs = re.sub(rf"\b{w}\b", str(_VI_NUM[w]), qs)
    return qs

def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()
    if not qs:
        return "chitchat"
    if parse_poem_request(qs) is not None:
        return "poem"
    if qs in {"hi","hello","xin chào","chào","chào bạn"} or (qs.endswith(("không?","ko?")) and len(qs) <= 20):
        return "chitchat"
    if re.fullmatch(r"\s*\d+\s*[\+\-\*x]\s*\d+\s*=\s*\??\s*", qs):
        return "generic"
    return "domain"

def parse_poem_request(q: str):
    """
    Bắt mọi biến thể thường gặp:
    - '10 câu (thơ) đầu', 'trích 20 câu đầu', 'hai mươi câu đầu'
    - 'câu 241-260', 'từ câu 241 đến 260'
    - 'câu 11', 'câu số 11', 'câu 11 là gì'
    - 'trích câu 11–20'
    """
    qs = _normalize_vi_number_words((q or "").lower())

    # A) "10 câu (thơ) đầu", "trích 10 câu (thơ) đầu"
    m = re.search(r"\b(?:trích|lấy|cho)?\s*(\d+)\s*câu\s*(?:thơ\s*)?đầu\b", qs)
    if m:
        return ("opening", int(m.group(1)))

    # B) "câu 241-260" hoặc "câu 11–20"
    m = re.search(r"\bcâu\s*(\d+)\s*[–-]\s*(\d+)\b", qs)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("range", min(a,b), max(a,b))

    # C) "từ câu 241 đến 260"
    m = re.search(r"\btừ\s*câu\s*(\d+)\s*đến\s*(?:câu\s*)?(\d+)\b", qs)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("range", min(a,b), max(a,b))

    # D) "câu 11", "câu số 11", "câu 11 là gì"
    m = re.search(r"\bcâu\s*(?:số\s*)?(\d+)\b", qs)
    if m:
        n = int(m.group(1))
        return ("range", n, n)

    # E) "(20) câu đầu" sau normalize "hai mươi"->20
    m = re.search(r"\b(\d+)\s*câu\s*(?:thơ\s*)?đầu\b", qs)
    if m:
        return ("opening", int(m.group(1)))

    return None
