# app/prompt_engineering.py
# -*- coding: utf-8 -*-
"""High level prompt engineering utilities for Kieu-Bot (robust & safe)."""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover
    from .poem_tools import PoemLine
except ImportError:  # pragma: no cover
    from poem_tools import PoemLine  # type: ignore

# ==== Token budgets (ints) ====================================================
DEFAULT_SHORT_TOKEN_BUDGET: int = 640
DEFAULT_LONG_TOKEN_BUDGET: int = 1152

# ==== Global style guardrails =================================================
_GLOBAL_STYLE_GUARDRAILS = (
    "- Giữ giọng học giả văn chương, mềm mại nhưng súc tích.\n"
    "- Tránh liệt kê khô, ưu tiên câu ghép cân đối 18–28 từ.\n"
    "- Tuyệt đối không nhắc tới nguồn tài liệu hay tài liệu tham khảo.\n"
    "- Trước khi trả lời, hãy lập dàn ý trong đầu (KHÔNG in ra)."
)

# ==== Helpers ================================================================
def _as_str(x: Any) -> str:
    """Ép an toàn về chuỗi (tránh .strip trên int / object)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""

def _normalise(x: Any) -> str:
    return _as_str(x).strip()

def _history_section(history_text: Optional[str]) -> str:
    s = _normalise(history_text)
    return f"[NHẬT KÝ HỘI THOẠI]\n{s}" if s else ""

def _compose_sections(sections: Iterable[Any]) -> str:
    parts: List[str] = []
    for s in sections:
        txt = _normalise(s)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)

def _format_context(ctx_list: Sequence[Dict[str, Any]]) -> str:
    if not ctx_list:
        return "(không có ngữ cảnh phù hợp – hãy dựa vào tri thức đã nạp, tránh suy đoán)."

    blocks: List[str] = []
    for idx, ctx in enumerate(ctx_list, start=1):
        # text
        text = _normalise(ctx.get("text") or ctx.get("content") or ctx.get("body") or ctx.get("snippet"))
        # meta
        meta = ctx.get("metadata") or ctx.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        label_bits: List[str] = []
        title = _normalise(meta.get("title") or meta.get("section_title"))
        if title:
            label_bits.append(title)
        doc_type = _normalise(meta.get("type"))
        if doc_type:
            label_bits.append(doc_type)
        source = _normalise(meta.get("source"))
        if source:
            label_bits.append(source)

        ls = meta.get("line_start")
        le = meta.get("line_end")
        if isinstance(ls, int) and isinstance(le, int):
            label_bits.append(f"L{ls}" if ls == le else f"L{ls}–L{le}")
        elif isinstance(ls, int):
            label_bits.append(f"L{ls}")

        score = ctx.get("score")
        if isinstance(score, (int, float)):
            label_bits.append(f"score={float(score):.3f}")

        header = f"[ĐOẠN {idx}" + (": " + " | ".join(label_bits) if label_bits else "") + "]"
        blocks.append(f"{header}\n{_truncate(text)}")

    return "\n\n---\n\n".join(blocks)

def _truncate(s: str, limit: int = 1200) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    head = s[:limit]
    # cắt gọn theo biên từ cho đẹp
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head + "…"

def _compose_prompt(
    *,
    system: Any,
    briefing: Any,
    user_request: Any,
    history_text: Optional[str] = None,
    context_block: Optional[str] = None,
    response_plan: Optional[str] = None,
) -> str:
    sections: List[str] = [f"[SYSTEM]\n{_normalise(system)}"]

    hist_section = _history_section(history_text)
    if hist_section:
        sections.append(hist_section)

    if context_block:
        sections.append(f"[NGỮ CẢNH]\n{_normalise(context_block)}")

    sections.append(f"[CHIẾN LƯỢC]\n{_GLOBAL_STYLE_GUARDRAILS}\n{_normalise(briefing)}")

    if response_plan:
        sections.append(f"[KẾ HOẠCH TRẢ LỜI]\n{_normalise(response_plan)}")

    sections.append(f"[YÊU CẦU NGƯỜI DÙNG]\n{_normalise(user_request)}")

    return _compose_sections(sections)

# ==== Builders ===============================================================
def build_smalltalk_prompt(message: str, history_text: Optional[str] = None) -> str:
    return _compose_prompt(
        system="Bạn là người bạn tri kỷ am hiểu thơ ca và Truyện Kiều.",
        briefing=(
            "- Ưu tiên 3–5 câu thân mật, gợi hình ảnh từ Truyện Kiều khi tự nhiên.\n"
            "- Giữ thái độ nâng đỡ cảm xúc, tránh phán xét.\n"
            "- Có thể kết câu bằng một gợi ý mở rộng hội thoại."
        ),
        history_text=history_text,
        user_request=message,
        response_plan=(
            "1) Nhìn lại lịch sử trò chuyện để nắm tâm trạng.\n"
            "2) Chọn 1 hình ảnh hoặc câu thơ phù hợp (không cần trích dẫn nếu không tự nhiên).\n"
            "3) Trả lời ấm áp, tối đa một câu hỏi ngắn để tiếp tục trò chuyện."
        ),
    )

def build_generic_prompt(
    message: str,
    history_text: Optional[str] = None,
    *,
    depth: str = "balanced",
) -> str:
    depth_line = {
        "concise": "- Độ dài 6–8 câu, tập trung ý chính.",
        "expanded": "- Độ dài 10–12 câu, triển khai từng bước lập luận.",
    }.get(_normalise(depth).lower(), "- Độ dài 8–10 câu, triển khai từng luận điểm rõ ràng.")

    return _compose_prompt(
        system="Bạn là học giả bách khoa giàu trải nghiệm giáo dục.",
        briefing=(
            "- Giải thích khái niệm bằng ví dụ gần gũi rồi khái quát hóa.\n"
            "- Chia câu trả lời thành các đoạn ngắn với câu chủ đề rõ ràng.\n"
            f"{depth_line}\n"
            "- Nhấn mạnh ứng dụng/thông điệp rút ra cuối cùng.\n"
            "- Không thêm mục 'Nguồn'."
        ),
        history_text=history_text,
        user_request=message,
        response_plan=(
            "1) Xác định 2–3 ý chính.\n"
            "2) Minh họa mỗi ý bằng ví dụ ngắn.\n"
            "3) Kết thúc bằng lời khuyên hoặc tổng kết."
        ),
    )

def build_poem_disambiguation_prompt(message: str, history_text: Optional[str] = None) -> str:
    return _compose_prompt(
        system="Bạn phụ trách tra cứu Truyện Kiều, giúp người dùng xác định rõ câu cần trích.",
        briefing=(
            "- Giữ câu trả lời tối đa 2 câu.\n"
            "- Nếu thiếu thông tin (số câu/bối cảnh), hỏi lại thật cụ thể.\n"
            "- Không đưa ra câu thơ khi chưa rõ yêu cầu."
        ),
        history_text=history_text,
        user_request=message,
        response_plan=(
            "1) Kiểm tra lịch sử xem người dùng đã nêu chỉ số câu chưa.\n"
            "2) Nếu chưa đủ, hỏi lại theo dạng \"Bạn muốn trích ...?\" với lựa chọn cụ thể.\n"
            "3) Cảm ơn và chờ phản hồi."
        ),
    )

def build_rag_synthesis_prompt(
    query: str,
    contexts: Sequence[Dict[str, Any]],
    *,
    history_text: Optional[str] = None,
    long_answer: bool = False,
) -> str:
    ctx_block = _format_context(contexts)
    length_line = (
        "- Độ dài mục tiêu: 320–420 từ, chia 3 đoạn rõ mở–thân–kết."
        if long_answer
        else "- Độ dài mục tiêu: 220–300 từ, vẫn đảm bảo mở–thân–kết."
    )
    briefing = (
        "- Sử dụng dữ kiện trong NGỮ CẢNH để lập luận, tránh suy đoán.\n"
        "- Khi trích thơ, đặt trong ngoặc kép; nếu biết số câu, có thể ghi (câu X–Y).\n"
        "- Kết nối các đoạn bằng câu chuyển ý giàu hình ảnh.\n"
        "- Không thêm mục 'Nguồn'.\n"
        f"{length_line}"
    )
    plan = (
        "1) Chọn 3 luận điểm then chốt.\n"
        "2) Mỗi luận điểm: ngữ liệu -> phân tích -> tiểu kết.\n"
        "3) Kết luận khái quát tư tưởng và mở rộng hiện đại."
    )
    return _compose_prompt(
        system="Bạn là giám khảo kỳ cựu môn Văn học trung đại, chuyên sâu Truyện Kiều.",
        briefing=briefing,
        user_request=query,
        history_text=history_text,
        context_block=ctx_block,
        response_plan=plan,
    )

def _format_line(line: PoemLine | Dict[str, Any]) -> str:
    if isinstance(line, dict):
        num = line.get("number")
        txt = _normalise(line.get("text"))
        motifs = line.get("motifs") or []
    else:
        num = getattr(line, "number", None)
        txt = _normalise(getattr(line, "text", ""))
        motifs = getattr(line, "motifs", []) or []
    motif_note = f" · motif: {', '.join(motifs)}" if motifs else ""
    if num is not None:
        return f"(câu {num}) {txt}{motif_note}"
    return txt + motif_note

def build_poem_compare_prompt(
    query: str,
    *,
    line_a: PoemLine | Dict[str, Any],
    line_b: PoemLine | Dict[str, Any],
    history_text: Optional[str] = None,
) -> str:
    side_by_side = f"{_format_line(line_a)}\n{_format_line(line_b)}"
    briefing = (
        "- So sánh hai câu thơ dựa trên ngữ cảnh Truyện Kiều.\n"
        "- Phân tích: hình ảnh, nhạc tính, tư tưởng, cảm xúc.\n"
        "- Liên hệ motif để lý giải tương đồng/khác biệt.\n"
        "- Không thêm mục 'Nguồn'."
    )
    plan = (
        "1) Nhắc lại yêu cầu và giới thiệu hai câu thơ.\n"
        "2) So sánh theo các bình diện đã nêu.\n"
        "3) Kết luận giá trị nghệ thuật và thông điệp."
    )
    return _compose_prompt(
        system="Bạn là chuyên gia chấm thi HSG Văn, nắm vững Truyện Kiều.",
        briefing=briefing,
        user_request=query,
        history_text=history_text,
        context_block=f"[TRÍCH DẪN]\n{side_by_side}",
        response_plan=plan,
    )

__all__ = [
    "DEFAULT_LONG_TOKEN_BUDGET",
    "DEFAULT_SHORT_TOKEN_BUDGET",
    "build_generic_prompt",
    "build_poem_disambiguation_prompt",
    "build_rag_synthesis_prompt",
    "build_smalltalk_prompt",
    "build_poem_compare_prompt",
]
