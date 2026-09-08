# -*- coding: utf-8 -*-
"""Response contracts and deterministic verification for Kiều Bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def verify_generated_answer(
    answer: str,
    *,
    require_exact_quotes: bool,
    has_evidence: bool,
) -> tuple[str, Dict[str, object], QualityReport]:
    """Autocorrect near-exact poem quotes and fail closed on unverifiable quotes."""
    from .verifier import verify_and_autocorrect

    corrected, verification = verify_and_autocorrect(answer or "", threshold=92.0, autocorrect=True)
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
    "deterministic_quality",
    "grounded_quality",
    "out_of_scope_answer",
    "route_metadata",
    "verified_core_answer",
    "verify_generated_answer",
]
