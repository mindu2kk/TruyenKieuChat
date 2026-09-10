# -*- coding: utf-8 -*-
"""Deterministic intent routing for the chatbot harness.

The router uses cheap, explainable rules. Retrieval and the LLM remain
responsible for domain answers, but they do not decide which flow handles a
request.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    flow: str
    confidence: float
    reason: str
    requires_poem_evidence: bool = False
    requires_exact_quotes: bool = False


_NUMBER_WORDS = {
    "mot": 1,
    "hai": 2,
    "ba": 3,
    "bon": 4,
    "tu": 4,
    "nam": 5,
    "sau": 6,
    "bay": 7,
    "tam": 8,
    "chin": 9,
    "muoi": 10,
}

_DOMAIN_TERMS = (
    "truyen kieu",
    "doan truong tan thanh",
    "nguyen du",
    "thuy kieu",
    "thuy van",
    "kim trong",
    "tu hai",
    "hoan thu",
    "thuc sinh",
    "ma giam sinh",
    "dam tien",
    "gia dinh vuong vien ngoai",
)

_OUT_OF_SCOPE_TERMS = (
    "tong thong",
    "thu tuong",
    "thoi tiet",
    "bitcoin",
    "chung khoan",
    "bong da",
    "lap trinh",
    "python",
    "javascript",
    "tri tue nhan tao",
    "chien tranh the gioi",
)


def normalize_query(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D").lower()
    value = re.sub(r"[^0-9a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _number_value(token: str, default: int = 2) -> int:
    token = normalize_query(token)
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token, default)


def _parse_list_request(q: str) -> Optional[Tuple[str, int]]:
    qs = normalize_query(q)
    if not qs:
        return None

    prefixes = ("liet ke", "bullet", "tldr", "tom tat")
    keywords = (
        "liet ke",
        "tom tat",
        "tong hop",
        "danh sach",
        "gach dau dong",
        "diem chinh",
        "key takeaway",
        "bullet",
    )
    if any(qs.startswith(prefix) for prefix in prefixes) or any(term in qs for term in keywords):
        match = re.search(r"\b(\d{1,2})\b", qs)
        count = int(match.group(1)) if match else 5
        return "facts", max(3, min(count, 12))
    return None


def parse_poem_request(q: str):
    """Parse exact poem lookup requests without involving an LLM."""
    qs = normalize_query(q)
    number = r"(\d{1,4}|mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi)"

    # "hai câu đầu", "2 câu mở đầu", "trích hai câu mở đầu"
    match = re.search(rf"(?:trich\s+)?{number}\s+cau\s+(?:mo\s+)?dau\b", qs)
    if match:
        return "opening", _number_value(match.group(1))

    match = re.search(r"so sanh cau\s*(\d+)\s*(?:voi|vs|va)\s*cau\s*(\d+)", qs)
    if match:
        return "compare", int(match.group(1)), int(match.group(2))

    match = re.search(r"cau\s*(\d+)\s*-\s*(\d+)", qs)
    if match:
        return "range", int(match.group(1)), int(match.group(2))

    # normalize_query removes punctuation, so a hyphen/en-dash range becomes
    # two adjacent numbers separated by whitespace.
    match = re.search(r"cau\s*(\d+)\s+(\d+)\b", qs)
    if match:
        return "range", int(match.group(1)), int(match.group(2))

    match = re.search(r"tu\s*cau\s*(\d+)\s*den\s*cau\s*(\d+)", qs)
    if match:
        return "range", int(match.group(1)), int(match.group(2))

    match = re.search(r"(?:doc|trich|cho)?\s*cau(?:\s*so)?\s*(\d+)\b", qs)
    if match:
        return "single", int(match.group(1))

    return None


def requests_poem_explanation(q: str) -> bool:
    """Return whether the user positively asks for literary explanation.

    A negated phrase such as ``không giải thích thêm`` must not be treated as
    an explanation request merely because it contains the words
    ``giải thích``.
    """
    qs = normalize_query(q)
    no_explanation = (
        "khong giai thich",
        "khong phan tich",
        "khong binh",
        "chi trich",
        "chi chep",
    )
    if any(term in qs for term in no_explanation):
        return False
    return any(term in qs for term in ("giai thich", "y nghia", "phan tich", "cam nhan", "binh"))


def requests_poem_evidence(q: str) -> bool:
    """Detect poetry-specific requests without confusing character names.

    In particular, ``Thúy Vân`` normalises to ``thuy van``; the isolated word
    ``van`` therefore cannot safely mean poetic rhyme.
    """
    qs = normalize_query(q)
    exact_or_form = (
        "trich",
        "nguyen van",
        "cau tho",
        "luc bat",
        "the tho",
        "nhip tho",
        "nhip dieu",
        "gieo van",
        "van tho",
        "van dieu",
        "phep doi",
        "doi xung",
        "diep tu",
        "diep ngu",
    )
    return parse_poem_request(q) is not None or any(term in qs for term in exact_or_form)


def requests_exact_poem_quote(q: str) -> bool:
    qs = normalize_query(q)
    return parse_poem_request(q) is not None or any(term in qs for term in ("trich", "nguyen van", "cau tho"))


def route_query(q: str) -> RouteDecision:
    qs = normalize_query(q)
    if not qs:
        return RouteDecision("chitchat", "smalltalk", 1.0, "empty-or-whitespace")

    if parse_poem_request(q) is not None:
        if requests_poem_explanation(q):
            return RouteDecision(
                "poem",
                "grounded-poem-analysis",
                1.0,
                "exact-poem-with-explanation",
                True,
                True,
            )
        return RouteDecision("poem", "exact-poem-lookup", 1.0, "poem-range-pattern", True, True)

    if _parse_list_request(q) is not None:
        return RouteDecision("facts", "structured-domain", 0.95, "list-pattern")

    greeting = re.match(r"^(hi|hello|xin chao|chao)(\b|\s)", qs)
    capability = any(term in qs for term in ("ban co the giup", "ban lam duoc gi", "giup toi hoc"))
    if greeting or capability:
        return RouteDecision("chitchat", "smalltalk", 0.98, "greeting-or-capability")

    has_domain_term = any(term in qs for term in _DOMAIN_TERMS)
    if not has_domain_term and any(term in qs for term in _OUT_OF_SCOPE_TERMS):
        return RouteDecision("out_of_scope", "safe-refusal", 0.99, "explicit-out-of-scope-topic")

    if "tac gia" in qs and "truyen kieu" in qs:
        return RouteDecision("core_fact", "verified-core-fact", 1.0, "author-question")

    if "truyen kieu" in qs and "nhan vat" in qs and any(term in qs for term in ("chinh", "trung tam")):
        return RouteDecision("core_fact", "verified-core-fact", 1.0, "main-character-question")

    if re.fullmatch(r"\s*\d+\s*[+\-*x]\s*\d+\s*=\s*\??\s*", (q or "").lower()):
        return RouteDecision("generic", "generic", 0.98, "arithmetic-pattern")

    if requests_poem_evidence(q):
        exact_quotes = requests_exact_poem_quote(q)
        return RouteDecision(
            "poem_analysis",
            "verified-poem-analysis",
            0.9,
            "poem-analysis-pattern",
            True,
            exact_quotes,
        )

    if any(term in qs for term in ("phan tich", "cam nhan", "binh giang", "nghe thuat", "an du", "diem nhin")):
        return RouteDecision("analysis", "literary-analysis", 0.9, "analysis-pattern")

    if any(term in qs for term in ("gap", "ban minh", "chuoc cha", "su kien", "vi sao", "hoan canh")):
        return RouteDecision("plot", "plot-and-character", 0.82, "plot-pattern")

    if has_domain_term:
        return RouteDecision("domain", "domain-qa", 0.8, "domain-entity")

    return RouteDecision("domain", "domain-qa", 0.55, "domain-default")


def route_intent(q: str) -> str:
    return route_query(q).intent


_CHITCHAT_RESPONSES = {
    "hi": "Xin chào! Tôi là trợ lý về Truyện Kiều. Bạn muốn hỏi gì về tác phẩm này?",
    "hello": "Xin chào! Tôi là trợ lý về Truyện Kiều. Bạn muốn hỏi gì về tác phẩm này?",
    "xin chao": "Xin chào! Tôi là trợ lý về Truyện Kiều. Bạn muốn hỏi gì về tác phẩm này?",
    "chao": "Chào bạn! Tôi có thể giúp gì cho bạn về Truyện Kiều?",
    "chao ban": "Chào bạn! Tôi có thể giúp gì cho bạn về Truyện Kiều?",
}


def get_chitchat_response(q: str) -> str | None:
    qs = normalize_query(q)
    exact = _CHITCHAT_RESPONSES.get(qs)
    if exact:
        return exact
    if re.match(r"^(hi|hello|xin chao|chao)(\b|\s)", qs):
        return (
            "Xin chào! Tôi có thể giúp bạn tra câu thơ, tìm hiểu nhân vật, "
            "tóm tắt tình tiết hoặc phân tích Truyện Kiều. Bạn muốn bắt đầu từ đâu?"
        )
    return None


__all__ = [
    "RouteDecision",
    "get_chitchat_response",
    "normalize_query",
    "parse_poem_request",
    "requests_exact_poem_quote",
    "requests_poem_evidence",
    "requests_poem_explanation",
    "route_intent",
    "route_query",
]
