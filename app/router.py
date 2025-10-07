# app/router.py
# -*- coding: utf-8 -*-
import re

def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()

    # Ưu tiên nhận dạng poem: chỉ khi KHỚP MẪU RÕ RÀNG
    if parse_poem_request(qs) is not None:
        return "poem"

    # chitchat ngắn
    if qs in {"hi", "hello", "xin chào", "chào", "chào bạn"}:
        return "chitchat"

    # phép tính đơn giản
    if re.fullmatch(r"\s*\d+\s*[\+\-\*x]\s*\d+\s*=?\s*\??\s*", qs):
        return "generic"

    # mặc định: domain (Truyện Kiều → RAG)
    return "domain"


def parse_poem_request(qs: str):
    """Nhận dạng các mẫu truy vấn thơ:
    - '10 câu đầu', 'trích 10 câu đầu'
    - 'câu 241-260', 'từ câu 241 đến câu 260'
    - 'câu 11', 'câu số 11', 'câu 11 là gì'
    - 'so sánh câu 31 và câu 32'
    """
    # 1) 10 câu đầu
    m = re.search(r"(?:^|\b)(\d+)\s*câu\s*đầu\b", qs)
    if m:
        return ("opening", int(m.group(1)))

    m = re.search(r"\btrích\s*(\d+)\s*câu\b", qs)
    if m:
        return ("opening", int(m.group(1)))

    # 2) khoảng câu
    m = re.search(r"\bcâu\s*(\d+)\s*[-–]\s*(\d+)\b", qs)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("range", a, b)

    m = re.search(r"\btừ\s*câu\s*(\d+)\s*đến\s*câu\s*(\d+)\b", qs)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("range", a, b)

    # 3) một câu cụ thể (không kèm khoảng  A–B)
    #    ví dụ: 'câu 11', 'câu số 11', 'câu 11 là gì'
    m = re.search(r"\bcâu\s*(?:số\s*)?(\d+)\b(?!\s*[-–]\s*\d+)", qs)
    if m:
        return ("single", int(m.group(1)))

    # 4) so sánh hai câu
    m = re.search(r"\bso\s*sánh\s*câu\s*(\d+)\s*(?:và|&)\s*câu\s*(\d+)\b", qs)
    if m:
        return ("compare", int(m.group(1)), int(m.group(2)))

    return None
