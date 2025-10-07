# app/router.py
# -*- coding: utf-8 -*-
import re

def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()

    # poem first: hỏi theo số dòng, trích đoạn, "câu X là gì", so sánh câu
    if parse_poem_request(qs) is not None:
        return "poem"

    # chitchat ngắn
    if qs in {"hi","hello","xin chào","chào","chào bạn"}:
        return "chitchat"

    # phép tính đơn
    if re.fullmatch(r"\s*\d+\s*[\+\-\*x]\s*\d+\s*=?\s*\??\s*", qs):
        return "generic"

    # mặc định: domain (RAG)
    return "domain"

def parse_poem_request(qs: str):
    """Nhận dạng:
    - '10 câu đầu', 'trích 10 câu đầu'
    - 'câu 241-260', 'từ câu 241 đến câu 260'
    - 'câu 11 là gì', 'cho mình câu số 11'
    - 'so sánh câu 31 và 32'
    """
    # 10 câu đầu
    m = re.search(r"(\d+)\s*câu\s*đầu", qs)
    if m:
        return ("opening", int(m.group(1)))

    # khoảng câu
    m = re.search(r"câu\s*(\d+)\s*[-–]\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(1)), int(m.group(2)))
    m = re.search(r"từ\s*câu\s*(\d+)\s*đến\s*câu\s*(\d+)", qs)
    if m:
        return ("range", int(m.group(1)), int(m.group(2)))

    # một câu cụ thể
    m = re.search(r"(?:câu\s*số|câu)\s*(\d+)\s*(?:là\s*gì|?)", qs)
    if m:
        return ("single", int(m.group(1)))

    # so sánh hai câu theo số
    m = re.search(r"so sánh\s*câu\s*(\d+)\s*(?:và|&)\s*câu\s*(\d+)", qs)
    if m:
        return ("compare", int(m.group(1)), int(m.group(2)))

    # trích N câu (mặc định từ đầu)
    m = re.search(r"trích\s*(\d+)\s*câu\b", qs)
    if m:
        return ("opening", int(m.group(1)))

    return None
