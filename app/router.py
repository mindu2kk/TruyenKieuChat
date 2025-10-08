# -*- coding: utf-8 -*-
import re

def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()
    if not qs:
        return "chitchat"

    # thơ: nếu parse được yêu cầu thơ => poem
    if parse_poem_request(qs) is not None:
        return "poem"

    # chitchat ngắn
    if qs in {"hi","hello","xin chào","chào","chào bạn"} or (qs.endswith(("không?","ko?")) and len(qs) <= 20):
        return "chitchat"

    # phép tính rất cơ bản
    if re.fullmatch(r"\s*\d+\s*[\+\-\*x]\s*\d+\s*=\s*\??\s*", qs):
        return "generic"

    return "domain"

def parse_poem_request(q: str):
    """Trả về:
       ("opening", n)             -> '10 câu đầu', 'trích 20 câu đầu'
       ("range", a, b)            -> 'câu 241-260', 'từ câu 1 đến câu 10'
       ("single", n)              -> 'câu 11', 'câu số 25 là gì'
    """
    qs = (q or "").lower()

    # 10 câu đầu / trích 10 câu đầu
    m = re.search(r"(?:trích\s*)?(\d+)\s*câu\s*đầu\b", qs)
    if m:
        return ("opening", int(m.group(1)))
    
    m = re.search(r"so\s*sánh\s*câu\s*(\d+)\s*(?:với|vs|và)\s*câu\s*(\d+)", qs)
    if m:
        return ("compare", int(m.group(1)), int(m.group(2)))

    # câu 241–260 / từ câu 241 đến câu 260
    m = re.search(r"câu\s*(\d+)\s*[-–]\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(1)), int(m.group(2)))
    m = re.search(r"từ\s*câu\s*(\d+)\s*đến\s*câu\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(1)), int(m.group(2)))

    # câu 11 / câu số 11 / câu 11 là gì / đọc câu 11
    m = re.search(r"(?:câu(?:\s*số)?)\s*(\d+)(?:\s*(?:là\s*gì|gì|đọc|cho|trích))?\b", qs)
    if m:
        return ("single", int(m.group(1)))

    return None
