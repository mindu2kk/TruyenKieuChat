"""Shadow-only HyDE and CRAG primitives.

Nothing in the production request path imports this module. The evaluator uses
it to compare strategies before any feature can be considered for promotion.
"""

from __future__ import annotations

import re
import time
import unicodedata
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .retrieval_policy import extract_line_range, select_retrieval_policy
from .router import normalize_query


_STOPWORDS = {
    "cua", "cho", "trong", "truyen", "kieu", "hay", "nhung", "mot", "cac",
    "the", "nao", "tai", "sao", "voi", "va", "la", "gi", "ve", "doan",
}

_DIRECT_FACT_PATTERNS = (
    r"\bbao nhieu cau\b",
    r"\bthuoc the tho nao\b",
    r"\btac gia la ai\b",
    r"\bai la tac gia\b",
    r"\bsang tac nam nao\b",
    r"\bten goi khac\b",
)

_PROMOTION_CONFIG = Path(__file__).resolve().parents[1] / "config" / "advanced_retrieval.json"


@lru_cache(maxsize=1)
def load_promotion_config() -> Mapping[str, Any]:
    try:
        payload = json.loads(_PROMOTION_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def is_strategy_promoted(name: str) -> bool:
    return bool(strategy_promotion(name).get("enabled") is True)


def strategy_promotion(name: str) -> Mapping[str, Any]:
    strategies = load_promotion_config().get("strategies") or {}
    strategy = strategies.get(name) if isinstance(strategies, Mapping) else None
    return strategy if isinstance(strategy, Mapping) else {}


def _tokens(value: str) -> set[str]:
    text = unicodedata.normalize("NFD", (value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char)).replace("đ", "d")
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in _STOPWORDS}


def should_use_hyde(query: str) -> bool:
    """Use HyDE only for open-ended semantic retrieval, never exact lookup."""
    normalized = normalize_query(query)
    policy = select_retrieval_policy(query)
    if extract_line_range(query):
        return False
    if policy.lane in {"verse", "glossary"}:
        return False
    if any(term in normalized for term in ("trich", "nguyen van", "cau so", "nghia la gi")):
        return False
    if any(re.search(pattern, normalized) for pattern in _DIRECT_FACT_PATTERNS):
        return False
    return policy.lane in {"literary-analysis", "character", "timeline", "general"}


def build_hyde_prompt(query: str) -> str:
    return (
        "Viết một đoạn tài liệu giả định 80-140 từ để hỗ trợ TÌM KIẾM trong kho Truyện Kiều. "
        "Dùng tên nhân vật, sự kiện, khái niệm nghệ thuật và từ khóa có khả năng xuất hiện trong tài liệu liên quan. "
        "Không bịa hoặc trích nguyên văn câu thơ; không trả lời xã giao; chỉ xuất đoạn tìm kiếm.\n\n"
        f"Câu hỏi: {query.strip()}\nĐoạn tài liệu giả định:"
    )


def generate_hypothetical_document(
    query: str,
    *,
    generator: Callable[..., str],
    model: str | None = None,
) -> tuple[str, float]:
    started = time.perf_counter()
    # GPT-OSS spends completion tokens on reasoning before emitting content.
    # A tiny cap can therefore yield an apparently successful empty response.
    document = generator(build_hyde_prompt(query), model=model, long_answer=False, max_tokens=1024)
    return (document or "").strip(), (time.perf_counter() - started) * 1000.0


@dataclass(frozen=True)
class RetrievalGrade:
    label: str
    score: float
    query_coverage: float
    source_diversity: int


def grade_retrieval(query: str, contexts: Sequence[Mapping[str, Any]]) -> RetrievalGrade:
    """Lightweight CRAG evaluator used only for shadow routing decisions."""
    query_tokens = _tokens(query)
    if not contexts:
        return RetrievalGrade("incorrect", 0.0, 0.0, 0)

    top = list(contexts[:5])
    context_tokens = _tokens(" ".join(str(item.get("text") or "") for item in top))
    coverage = len(query_tokens & context_tokens) / max(1, len(query_tokens))
    sources = {
        str((item.get("meta") or item.get("metadata") or {}).get("source_id") or "")
        for item in top
    }
    sources.discard("")
    top_signal = max(
        float(item.get("re_score", item.get("_raw_score", item.get("score", 0.0))) or 0.0)
        for item in top
    )
    normalized_signal = min(1.0, top_signal if top_signal <= 1.0 else top_signal / 10.0)
    score = 0.72 * coverage + 0.2 * normalized_signal + 0.08 * min(1.0, len(sources) / 2.0)
    label = "correct" if score >= 0.56 else "ambiguous" if score >= 0.28 else "incorrect"
    return RetrievalGrade(label, round(score, 4), round(coverage, 4), len(sources))


def build_crag_grader_prompt(query: str, contexts: Sequence[Mapping[str, Any]]) -> str:
    excerpts = []
    for index, item in enumerate(contexts[:4], start=1):
        compact = " ".join(str(item.get("text") or "").split())[:900]
        excerpts.append(f"[{index}] {compact}")
    joined = "\n".join(excerpts) or "(không có tài liệu)"
    return (
        "Bạn là bộ chấm relevance cho retrieval của chatbot Truyện Kiều. "
        "Chỉ đánh giá liệu các đoạn tìm được có đủ bằng chứng trực tiếp để trả lời câu hỏi hay không. "
        "Trả đúng MỘT nhãn tiếng Anh: correct nếu đủ bằng chứng; ambiguous nếu có liên quan nhưng thiếu; "
        "incorrect nếu lạc đề hoặc không có bằng chứng. Không giải thích.\n\n"
        f"Câu hỏi: {query.strip()}\n\nCác đoạn tìm được:\n{joined}\n\nNhãn:"
    )


def grade_retrieval_with_model(
    query: str,
    contexts: Sequence[Mapping[str, Any]],
    *,
    generator: Callable[..., str],
    model: str,
) -> tuple[RetrievalGrade, float]:
    """CRAG critic for shadow evaluation; falls back by raising on invalid output."""
    started = time.perf_counter()
    output = generator(
        build_crag_grader_prompt(query, contexts),
        model=model,
        long_answer=False,
        max_tokens=512,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    normalized = (output or "").strip().casefold()
    match = re.search(r"\b(incorrect|ambiguous|correct)\b", normalized)
    if not match:
        raise ValueError(f"CRAG grader trả nhãn không hợp lệ: {output!r}")
    label = match.group(1)
    heuristic = grade_retrieval(query, contexts)
    score = {"incorrect": 0.0, "ambiguous": 0.5, "correct": 1.0}[label]
    return (
        RetrievalGrade(label, score, heuristic.query_coverage, heuristic.source_diversity),
        elapsed_ms,
    )


__all__ = [
    "RetrievalGrade",
    "build_hyde_prompt",
    "generate_hypothetical_document",
    "grade_retrieval",
    "grade_retrieval_with_model",
    "is_strategy_promoted",
    "load_promotion_config",
    "strategy_promotion",
    "should_use_hyde",
]
