# -*- coding: utf-8 -*-
import re

_POEM_PATTERNS = [
    r"^\s*\d+\s*câu\s*đầu\b",
    r"\b(trích|lấy|cho)\s+\d+\s*câu\s*đầu\b",
    r"\bcâu\s*\d+\s*[-–—]\s*\d+\b",
    r"\btừ\s*câu\s*\d+\s*(đến|tới)\s*câu\s*\d+\b",
    r"\b(trích|lấy)\s*(\d+)\s*câu\b",
    r"\b(trích|lấy)\s*câu\s*(\d+)\s*(đến|tới|-|–|—)\s*(\d+)\b",
]

def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()
    if not qs:
        return "chitchat"

    if parse_poem_request(q) is not None:
        return "poem"

    if qs in {"hi", "hello", "xin chào", "chào", "chào bạn"} or (
        len(qs) <= 20 and qs.endswith(("không?", "ko?"))
    ):
        return "chitchat"

    if re.fullmatch(r"\s*\d+\s*[\+\-\*x×]\s*\d+\s*=\s*\??\s*", qs):
        return "generic"

    return "domain"

def parse_poem_request(q: str):
    qs = (q or "").lower()

    # "10 câu đầu", "trích 10 câu đầu"
    m = re.search(r"(\d+)\s*câu\s*đầu", qs)
    if m:
        return ("opening", int(m.group(1)))

    # "trích 30 câu" -> mặc định là từ đầu
    m = re.search(r"(trích|lấy)\s*(\d+)\s*câu", qs)
    if m:
        return ("opening", int(m.group(2)))

    # "câu 241-260"
    m = re.search(r"câu\s*(\d+)\s*[-–—]\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(1)), int(m.group(2)))

    # "từ câu 241 đến/tới câu 260"
    m = re.search(r"từ\s*câu\s*(\d+)\s*(đến|tới)\s*câu\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(1)), int(m.group(3)))

    # "trích câu 12-40"
    m = re.search(r"(trích|lấy)\s*câu\s*(\d+)\s*(đến|tới|-|–|—)\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(2)), int(m.group(4)))

    return None
