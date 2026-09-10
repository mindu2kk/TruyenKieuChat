# -*- coding: utf-8 -*-
"""Response contracts and deterministic verification for Kiều Bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Optional, Tuple

from .router import RouteDecision, normalize_query


@dataclass(frozen=True)
class QualityReport:
    status: str
    grounded: bool
    quote_check: str
    issues: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        return payload


def route_metadata(decision: RouteDecision, quality: QualityReport) -> Dict[str, Any]:
    return {
        "flow": decision.flow,
        "route_confidence": decision.confidence,
        "route_reason": decision.reason,
        "quality": quality.as_dict(),
    }


def verified_core_answer(query: str) -> Optional[str]:
    """Return curated answers for small, stable facts that must never depend on retrieval."""
    qs = normalize_query(query)

    if "tac gia" in qs and "truyen kieu" in qs:
        if any(term in qs for term in ("tac pham goc", "chu han", "ten goc")):
            return (
                "Nguyễn Du là tác giả Truyện Kiều, còn có nhan đề Đoạn trường tân thanh và được viết bằng chữ Nôm. "
                "Tác phẩm được sáng tạo dựa trên cốt truyện Kim Vân Kiều truyện của Thanh Tâm Tài Nhân."
            )
        return "Nguyễn Du (1765–1820) là tác giả Truyện Kiều, còn có nhan đề Đoạn trường tân thanh."

    if "truyen kieu" in qs and "nhan vat" in qs and any(term in qs for term in ("chinh", "trung tam")):
        return "Thúy Kiều là nhân vật chính và là trung tâm của toàn bộ câu chuyện trong Truyện Kiều."

    return None


def out_of_scope_answer() -> str:
    return (
        "Câu hỏi này nằm ngoài phạm vi Truyện Kiều nên tôi không trả lời để tránh cung cấp thông tin thiếu kiểm chứng. "
        "Bạn có thể hỏi tôi về câu thơ, nhân vật, tình tiết hoặc nghệ thuật của tác phẩm."
    )


def curated_poem_explanation(poem_text: str) -> Optional[str]:
    """Return a reviewed explanation for high-frequency canonical excerpts."""
    lines = [normalize_query(line) for line in (poem_text or "").splitlines() if line.strip()]
    if (
        len(lines) >= 2
        and lines[0].startswith("tram nam trong coi nguoi ta")
        and lines[1].startswith("chu tai chu menh kheo la ghet nhau")
    ):
        return (
            "Nguyễn Du mở đầu bằng một nhận xét có tính khái quát về kiếp người, rồi nêu nghịch lý "
            "“tài mệnh tương đố”: người có tài thường phải chịu số phận éo le. Hai câu này đặt nền tư tưởng "
            "cho những biến cố và bi kịch của Thúy Kiều về sau."
        )
    return None


def deterministic_quality(status: str = "verified") -> QualityReport:
    return QualityReport(status=status, grounded=True, quote_check="not-required")


def grounded_quality(*, has_evidence: bool) -> QualityReport:
    if has_evidence:
        return QualityReport(status="grounded", grounded=True, quote_check="not-required")
    return QualityReport(
        status="insufficient-evidence",
        grounded=False,
        quote_check="not-required",
        issues=("no-retrieval-evidence",),
    )


_DANGLING_LIST_MARKER = re.compile(r"(?:^|\n)\s*(?:[-*+]\s*|(?:\*{0,2})\d{1,2}[.)](?:\*{0,2})\s*)$")
_LIST_ITEM = re.compile(r"(?m)^\s*(?:[-*+]\s+\S|(?:\*{0,2})\d{1,2}[.)](?:\*{0,2})\s+\S)")
_REQUESTED_ITEM_COUNT = re.compile(
    r"\b(?:dung|du|gom|liet ke|neu|trinh bay)\s+(\d{1,2})\s+"
    r"(?:y|ly do|nguyen nhan|pham chat|dac diem|luan diem|noi dung)\b"
)


def answer_completeness_issues(answer: str, query: str = "") -> Tuple[str, ...]:
    """Detect obvious truncation without pretending to semantically grade prose."""
    text = (answer or "").strip()
    if not text:
        return ("empty-answer",)

    issues = []
    if _DANGLING_LIST_MARKER.search(text):
        issues.append("dangling-list-marker")
    if text.endswith(":"):
        issues.append("dangling-ending")

    normalized_query = normalize_query(query)
    count_match = _REQUESTED_ITEM_COUNT.search(normalized_query)
    requested_items = int(count_match.group(1)) if count_match else 0
    if requested_items:
        detected_items = len(_LIST_ITEM.findall(text))
        if detected_items < requested_items:
            issues.append("requested-items-missing")

    return tuple(dict.fromkeys(issues))


_REFUSAL_PATTERNS = (
    "chua the xac minh",
    "khong the xac minh",
    "khong du bang chung",
    "khong co bang chung",
    "khong tim thay thong tin",
    "corpus khong chua",
    "du lieu khong chua",
)


def is_refusal_answer(answer: str) -> bool:
    """Identify answers that decline to answer instead of stating a fact."""
    normalized = normalize_query(answer)
    return any(pattern in normalized for pattern in _REFUSAL_PATTERNS)


def verify_generated_answer(
    answer: str,
    *,
    require_exact_quotes: bool,
    has_evidence: bool,
    query: str = "",
) -> tuple[str, Dict[str, object], QualityReport]:
    """Fail closed on incomplete prose and unverifiable poem quotations."""
    from .verifier import verify_and_autocorrect

    corrected, verification = verify_and_autocorrect(answer or "", threshold=92.0, autocorrect=True)
    completeness_issues = answer_completeness_issues(corrected, query)
    verification["completeness"] = {
        "status": "failed" if completeness_issues else "passed",
        "issues": list(completeness_issues),
    }
    if completeness_issues:
        safe_answer = (
            "Phản hồi vừa tạo bị thiếu ý hoặc kết thúc giữa chừng nên tôi chưa thể xác nhận. "
            "Bạn vui lòng thử lại để nhận câu trả lời hoàn chỉnh."
        )
        return (
            safe_answer,
            verification,
            QualityReport(
                status="incomplete",
                grounded=has_evidence,
                quote_check="not-checked",
                issues=completeness_issues,
            ),
        )

    quotes = list(verification.get("quotes", []))
    accepted = list(verification.get("accepted", []))
    rejected = [q for q in quotes if float(q.get("score", 0.0) or 0.0) < 92.0]

    if require_exact_quotes and (not quotes or rejected):
        issues = []
        if not quotes:
            issues.append("required-quote-missing")
        if rejected:
            issues.append("unverified-poem-quote")
        safe_answer = (
            "Tôi chưa thể xác minh nguyên văn trích dẫn cho câu trả lời này nên tạm thời không đưa ra câu thơ có thể sai. "
            "Bạn hãy cho biết số câu hoặc đoạn cần tra; tôi sẽ lấy trực tiếp từ bản thơ đã kiểm chứng."
        )
        quality = QualityReport(
            status="blocked-by-verifier",
            grounded=has_evidence,
            quote_check="failed",
            issues=tuple(issues),
        )
        return safe_answer, verification, quality

    quote_status = "passed" if quotes and len(accepted) == len(quotes) else "not-required"
    quality = QualityReport(
        status="verified" if has_evidence else "insufficient-evidence",
        grounded=has_evidence,
        quote_check=quote_status,
        issues=() if has_evidence else ("no-retrieval-evidence",),
    )
    return corrected, verification, quality


__all__ = [
    "QualityReport",
    "answer_completeness_issues",
    "curated_poem_explanation",
    "deterministic_quality",
    "grounded_quality",
    "is_refusal_answer",
    "out_of_scope_answer",
    "route_metadata",
    "verified_core_answer",
    "verify_generated_answer",
]
