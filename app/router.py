# -*- coding: utf-8 -*-
import re

def _parse_list_request(q: str):
    """
    Phát hiện yêu cầu liệt kê/tóm tắt ngắn.
    Trả về ("facts", n) với n = số mục nếu tìm thấy, ngược lại None.
    """
    qs = (q or "").lower().strip()

    # tín hiệu cứng: tiền tố 'liệt kê:' hoặc 'bullet:' từ UI
    if qs.startswith(("liệt kê:", "liet ke:", "bullet:", "tldr:", "tóm tắt:", "tom tat:")):
        # cố gắng bắt số mục phía sau, mặc định 5
        m = re.search(r"\b(\d{1,2})\b", qs)
        n = int(m.group(1)) if m else 5
        return ("facts", max(3, min(n, 12)))

    # từ khoá thường gặp
    KEYWORDS = [
        r"\bliệt\s*kê\b", r"\btóm\s*tắt\b", r"\btổng\s*hợp\b",
        r"\bdanh\s*sách\b", r"\bgạch\s*đầu\s*dòng\b", r"\bđiểm\s*chính\b",
        r"\bkey\s*takeaways?\b", r"\bbullet(s)?\b", r"\bnêu\s+(\d+)\s+ý\b",
        r"\b(\d+)\s*ý\s*chính\b",
    ]
    if any(re.search(pat, qs) for pat in KEYWORDS):
        m = re.search(r"\b(\d{1,2})\b", qs)
        n = int(m.group(1)) if m else 5
        return ("facts", max(3, min(n, 12)))
    if _parse_list_request(qs) is not None:
        return "facts"
    return None


def route_intent(q: str) -> str:
    qs = (q or "").lower().strip()
    if not qs:
        return "chitchat"

    # thơ: nếu parse được yêu cầu thơ => poem
    if parse_poem_request(qs) is not None:
        return "poem"

    # >>> NEW: nếu phát hiện liệt kê -> facts
    if _parse_list_request(qs) is not None:
        return "facts"

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
