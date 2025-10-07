# app/router.py
# -*- coding: utf-8 -*-
import re

_POEM_PATTS = [
    r"^\s*\d+\s*câu\s*đầu\b",
    r"\b(cho|xin|trích|lấy)\s+\d+\s*câu\s*đầu\b",
    r"\bcâu\s*\d+\s*[-–]\s*\d+\b",
    r"\btừ\s*câu\s*\d+\s*đến\s*câu\s*\d+\b",
    r"\btrích\s*(\d+)\s*câu\b",
    r"\b(trích|trích dẫn)\s+(?:toàn bộ|một đoạn|đoạn)\b",
]

def parse_poem_request(q: str):
    qs = (q or "").lower()
    m = re.search(r"(\d+)\s*câu\s*đầu", qs)
    if m: return ("opening", int(m.group(1)))
    m = re.search(r"từ\s*câu\s*(\d+)\s*đến\s*câu\s*(\d+)", qs)
    if m: return ("range", int(m.group(1)), int(m.group(2)))
    m = re.search(r"câu\s*(\d+)\s*[-–]\s*(\d+)", qs)
    if m: return ("range", int(m.group(1)), int(m.group(2)))
    m = re.search(r"trích\s*(\d+)\s*câu", qs)
    if m: return ("opening", int(m.group(1)))
    return None

def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()
    if not qs: return "chitchat"

    # THƠ trước tiên
    if parse_poem_request(q) is not None:
        return "poem"

    # chitchat rất ngắn
    if qs in {"hi","hello","xin chào","chào","chào bạn"} or (qs.endswith(("không?","ko?")) and len(qs) <= 20):
        return "chitchat"

    # toán cực đơn giản
    if re.fullmatch(r"\s*\d+\s*[\+\-\*x]\s*\d+\s*=\s*\??\s*", qs):
        return "generic"

    # mặc định: domain
    return "domain"
