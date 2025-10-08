# -*- coding: utf-8 -*-
"""High level prompt engineering utilities for Kieu-Bot."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - allow both package/script imports
    from .poem_tools import PoemLine
except ImportError:  # pragma: no cover
    from poem_tools import PoemLine

DEFAULT_SHORT_TOKEN_BUDGET = 640
DEFAULT_LONG_TOKEN_BUDGET = 1152

_GLOBAL_STYLE_GUARDRAILS = (
    "- Giữ giọng học giả văn chương, mềm mại nhưng súc tích.\n"
    "- Tránh liệt kê khô, ưu tiên câu ghép cân đối 18–28 từ.\n"
    "- Tuyệt đối không nhắc tới nguồn tài liệu hay tài liệu tham khảo.\n"
    "- Trước khi trả lời, hãy lập dàn ý trong đầu (KHÔNG in ra)."
)


def _normalise(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.strip()


def _history_section(history_text: Optional[str]) -> str:
    history_text = _normalise(history_text)
    if not history_text:
        return ""
    return f"[NHẬT KÝ HỘI THOẠI]\n{history_text}"


def _compose_sections(sections: Iterable[str]) -> str:
    return "\n\n".join(filter(None, (_normalise(s) for s in sections)))


def _format_context(ctx_list: Sequence[Dict[str, Any]]) -> str:
    if not ctx_list:
        return (
            "(không có ngữ cảnh phù hợp – hãy dựa vào kiến thức chung trong kho dữ liệu đã học, tránh bịa đặt)."
        )

    blocks: List[str] = []
    for idx, ctx in enumerate(ctx_list, start=1):
        text = _normalise(ctx.get("text"))
        meta = ctx.get("metadata") or {}

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

        line_start = meta.get("line_start")
        line_end = meta.get("line_end")
        if isinstance(line_start, int) and isinstance(line_end, int):
            if line_start == line_end:
                label_bits.append(f"L{line_start}")
            else:
                label_bits.append(f"L{line_start}–{line_end}")
        elif isinstance(line_start, int):
            label_bits.append(f"L{line_start}")

        score = ctx.get("score")
        if isinstance(score, (int, float)):
            label_bits.append(f"score={score:.3f}")

        if label_bits:
            header = f"[ĐOẠN {idx}: {' | '.join(label_bits)}]"
        else:
            header = f"[ĐOẠN {idx}]"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks)


def _compose_prompt(
    *,
    system: str,
    briefing: str,
    user_request: str,
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
    }.get(depth, "- Độ dài 8–10 câu, triển khai từng luận điểm rõ ràng.")

    return _compose_prompt(
        system="Bạn là học giả bách khoa giàu trải nghiệm giáo dục.",
        briefing=(
            "- Giải thích khái niệm bằng ví dụ gần gũi rồi khái quát hóa.\n"
            "- Chia câu trả lời thành các đoạn ngắn với câu chủ đề rõ ràng.\n"
            f"{depth_line}\n"
            "- Nhấn mạnh ứng dụng/thông điệp rút ra cuối cùng."
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
    context_block = _format_context(contexts)

    length_line = (
        "- Độ dài mục tiêu: 320–420 từ, chia 3 đoạn rõ mở–thân–kết."
        if long_answer
        else "- Độ dài mục tiêu: 220–300 từ, vẫn đảm bảo mở–thân–kết."
    )

    briefing = (
        "- Sử dụng dữ kiện trong NGỮ CẢNH để lập luận sâu, tránh suy đoán.\n"
        "- Khi trích thơ, đặt trong ngoặc kép và ghi rõ (câu X–Y) theo metadata.\n"
        "- Kết nối các đoạn bằng câu chuyển ý giàu hình ảnh.\n"
        f"{length_line}"
    )

    response_plan = (
        "1) Tìm 3 luận điểm then chốt liên quan yêu cầu.\n"
        "2) Với mỗi luận điểm: trích dẫn/ngữ liệu -> phân tích -> tiểu kết.\n"
        "3) Kết luận khái quát tư tưởng và mở rộng ý nghĩa hiện đại."
    )

    return _compose_prompt(
        system="Bạn là giám khảo kỳ cựu môn Văn học trung đại, chuyên sâu Truyện Kiều.",
        briefing=briefing,
        user_request=query,
        history_text=history_text,
        context_block=context_block,
        response_plan=response_plan,
    )


def _format_line(line: PoemLine) -> str:
    motifs = ", ".join(line.motifs) if line.motifs else ""
    motif_note = f" · motif: {motifs}" if motifs else ""
    return f"(câu {line.number}) {line.text}{motif_note}"


def build_poem_compare_prompt(
    query: str,
    *,
    line_a: PoemLine,
    line_b: PoemLine,
    history_text: Optional[str] = None,
) -> str:
    side_by_side = f"{_format_line(line_a)}\n{_format_line(line_b)}"
    briefing = (
        "- So sánh hai câu thơ dựa trên ngữ cảnh Truyện Kiều.\n"
        "- Phân tích theo trục: hình ảnh, nhạc tính, tư tưởng, cảm xúc.\n"
        "- Liên hệ motif để lý giải sự tương đồng và khác biệt."
    )
    response_plan = (
        "1) Nhắc lại yêu cầu và giới thiệu hai câu thơ.\n"
        "2) So sánh từng bình diện (hình ảnh, âm điệu, nội dung).\n"
        "3) Kết luận giá trị nghệ thuật và thông điệp."
    )
    return _compose_prompt(
        system="Bạn là chuyên gia chấm thi HSG Văn, nắm vững Truyện Kiều.",
        briefing=briefing,
        user_request=query,
        history_text=history_text,
        context_block=f"[TRÍCH DẪN]\n{side_by_side}",
        response_plan=response_plan,
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