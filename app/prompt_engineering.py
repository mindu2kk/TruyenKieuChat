# app/prompt_engineering.py
# -*- coding: utf-8 -*-
"""
Prompt templates cho Kiểu bot RAG văn học – Truyện Kiều.
Bao gồm:
- Small talk / Generic factual
- Poem disambiguation & compare
- RAG synthesis (citation-aware) + essay_mode="hsg"
- Literature Review + các prompt tiện ích citation-aware
"""

from typing import List, Dict, Any, Optional

# =========================================================
# Token budgets (cứ để rộng rãi, có thể sửa qua orchestrator)
# =========================================================
DEFAULT_SHORT_TOKEN_BUDGET = 600
DEFAULT_LONG_TOKEN_BUDGET = 1400


# =========================================================
# Tiện ích dựng nhãn trích dẫn [SOURCE: ...]
# =========================================================
def _cite_tag(meta: Dict[str, Any]) -> str:
    """
    Sinh nhãn trích dẫn gọn, ưu tiên info dòng thơ / offset.
    """
    src = meta.get("source") or meta.get("title") or "unknown"
    if meta.get("type") == "poem" and meta.get("line_start") and meta.get("line_end"):
        return f"[SOURCE: {src} L{meta['line_start']}-{meta['line_end']}]"
    if meta.get("char_start") is not None and meta.get("char_end") is not None:
        return f"[SOURCE: {src} {meta['char_start']}-{meta['char_end']}]"
    if meta.get("chunk_index") is not None:
        return f"[SOURCE: {src}#chunk{meta['chunk_index']}]"
    return f"[SOURCE: {src}]"


# =========================================================
# Small talk
# =========================================================
def build_smalltalk_prompt(user_msg: str, *, history_text: Optional[str] = None, **kwargs) -> str:
    hist = f"\n[HISTORY]\n{history_text.strip()}" if history_text else ""
    return (
        "Bạn là một trợ lý thân thiện, lịch sự, trả lời ngắn gọn, tự nhiên bằng tiếng Việt.\n"
        "Tránh bịa đặt sự kiện. Nếu người dùng chuyển chủ đề học thuật, mời họ cung cấp thêm chi tiết.\n\n"
        f"[USER]\n{user_msg.strip()}{hist}\n\n"
        "Trả lời:"
    )


# =========================================================
# Generic factual (không RAG)
# =========================================================
def build_generic_prompt(
    query: str,
    *,
    history_text: Optional[str] = None,
    depth: str = "balanced",
    **kwargs,
) -> str:
    hist = f"\n[HISTORY]\n{history_text.strip()}" if history_text else ""
    style = {
        "brief": "ngắn gọn, trọng tâm",
        "balanced": "cân bằng giữa ngắn gọn và đầy đủ",
        "expanded": "đầy đủ, có ví dụ minh hoạ khi cần",
    }.get(depth, "cân bằng")
    return (
        "Bạn là một trợ lý cung cấp thông tin chính xác. "
        f"Hãy trả lời bằng tiếng Việt, văn phong {style}. Nếu thiếu dữ liệu, nói rõ 'tôi chưa đủ dữ liệu'.\n\n"
        f"[USER QUESTION]\n{query.strip()}{hist}\n\n"
        "Trả lời:"
    )


# =========================================================
# Poem disambiguation – khi user hỏi mơ hồ về thơ
# =========================================================
def build_poem_disambiguation_prompt(
    user_query: str,
    *,
    history_text: Optional[str] = None,
    **kwargs,
) -> str:
    hist = f"\n[HISTORY]\n{history_text.strip()}" if history_text else ""
    return (
        "Bạn đang hỗ trợ tra cứu 'Truyện Kiều' (mỗi câu 1 dòng). "
        "Câu hỏi hiện chưa đủ rõ. Hãy đặt 1–2 câu hỏi làm rõ (rất ngắn), "
        "ví dụ: 'bạn cần khoảng câu số mấy?' hoặc 'bạn muốn trích nguyên văn hay phân tích?'\n\n"
        f"[USER]\n{user_query.strip()}{hist}\n\n"
        "Câu hỏi làm rõ:"
    )


# =========================================================
# Poem compare – so sánh hai câu thơ đã xác định
# =========================================================
def build_poem_compare_prompt(
    query: str,
    *,
    line_a: Any,
    line_b: Any,
    history_text: Optional[str] = None,
    **kwargs,
) -> str:
    """
    line_a, line_b: object từ poem_tools.compare_lines(...) có .number, .text
    """
    hist = f"\n[HISTORY]\n{history_text.strip()}" if history_text else ""
    a_no = getattr(line_a, "number", None)
    b_no = getattr(line_b, "number", None)
    a_tx = getattr(line_a, "text", "")
    b_tx = getattr(line_b, "text", "")
    return (
        "So sánh hai câu thơ trong Truyện Kiều theo các trục: nhịp–vần–điệp–đối–ngữ nghĩa–tu từ.\n"
        "Ưu tiên phân tích kỹ close-reading, trích đúng nguyên văn trong ngoặc kép, giữ dấu câu.\n\n"
        f"[CÂU A – {a_no}]\n{a_tx}\n"
        f"[CÂU B – {b_no}]\n{b_tx}\n\n"
        f"[YÊU CẦU]\n{query.strip()}{hist}\n\n"
        "Phân tích:"
    )


# =========================================================
# RAG synthesis (citation-aware) — hỗ trợ essay_mode="hsg"
# =========================================================
def build_rag_synthesis_prompt(
    query: str,
    contexts: List[Dict[str, Any]],
    *,
    history_text: Optional[str] = None,
    long_answer: bool = False,
    essay_mode: Optional[str] = None,   # bổ sung để tương thích orchestrator/rag_pipeline
    **kwargs,                            # nuốt an toàn tham số mới
) -> str:
    # Gói evidence (giới hạn 12 block cho gọn)
    blocks = []
    for i, ctx in enumerate(contexts[:12], start=1):
        text = (ctx.get("text") or "").strip()
        meta = dict(ctx.get("meta") or {})
        cite = _cite_tag(meta)
        if text:
            blocks.append(f"- EXCERPT {i}: {text}\n  {cite}")
    context_dump = "\n".join(blocks) if blocks else "(no evidence)"

    # Quy ước chung
    common_rules = (
        "YÊU CẦU:\n"
        "1) Bám sát trích dẫn trong [EVIDENCE]; không bịa.\n"
        "2) Mọi luận điểm quan trọng phải gắn nhãn trích dẫn dạng [SOURCE: …].\n"
        "3) Nếu trích câu thơ, đặt trong ngoặc kép và giữ nguyên văn.\n"
        "4) Nếu không đủ bằng chứng, hãy nói rõ: 'Không đủ bằng chứng từ corpus.'\n"
    )

    # Khung cấu trúc
    if (essay_mode or "").lower() == "hsg":
        structure = (
            "CẤU TRÚC BÀI (HSG):\n"
            "- Mở bài: nêu vấn đề/ngữ cảnh ngắn gọn.\n"
            "- Luận điểm 1 → Dẫn chứng → Phân tích → Tiểu kết. [SOURCE bắt buộc]\n"
            "- Luận điểm 2 → Dẫn chứng → Phân tích → Tiểu kết. [SOURCE bắt buộc]\n"
            "- (Có thể thêm luận điểm 3 nếu đủ chứng cứ.)\n"
            "- Kết luận: khái quát giá trị nghệ thuật/ý nghĩa.\n"
        )
        task = "Viết bài phân tích theo dàn ý HSG, súc tích nhưng có dẫn chứng cụ thể."
    else:
        structure = (
            "CẤU TRÚC TRẢ LỜI:\n"
            "- Trả lời trực tiếp câu hỏi.\n"
            "- Chèn dẫn chứng theo mạch lập luận, mỗi ý chính kèm [SOURCE].\n"
            "- Một đoạn kết ngắn tổng kết phát hiện chính.\n"
        )
        task = "Trả lời trực tiếp, có dẫn chứng và nhãn [SOURCE]."

    history_section = f"\n[HISTORY]\n{history_text.strip()}" if history_text else ""

    prompt = (
        "Bạn là nhà nghiên cứu văn học Việt Nam, chuyên về Truyện Kiều. "
        "Hãy tổng hợp và lập luận dựa trên chứng cứ trong corpus nội bộ dưới đây.\n\n"
        f"{common_rules}{structure}\n"
        f"NHIỆM VỤ: {task}\n\n"
        "[USER QUESTION]\n"
        f"{query.strip()}\n\n"
        "[EVIDENCE]\n"
        f"{context_dump}\n"
        f"{history_section}\n\n"
        "BẮT ĐẦU TRẢ LỜI:\n"
    )
    return prompt


# =========================================================
# Literature Review (citations required)
# =========================================================
def build_lit_review_prompt(
    topic: str,
    evidence_blocks: List[Dict[str, Any]],
    *,
    min_citations: int = 5,
    history_text: Optional[str] = None,
    **kwargs,
) -> str:
    lines = []
    for b in evidence_blocks[: max(min_citations, 12)]:
        txt = (b.get("text") or "").strip()
        meta = dict(b.get("meta") or {})
        if txt:
            lines.append(f"- {txt}\n  {_cite_tag(meta)}")
    ev_dump = "\n".join(lines) if lines else "(no evidence)"
    history_section = f"\n[HISTORY]\n{history_text.strip()}" if history_text else ""

    return f"""
Bạn là nhà nghiên cứu văn học. Hãy viết **tổng quan tài liệu** về chủ đề: "{topic}".

YÊU CẦU CHUNG:
- Cấu trúc: (1) Bối cảnh & phạm vi; (2) Nhóm chủ đề chính (gộp & đặt nhan đề con);
  (3) So sánh/đối chiếu quan điểm; (4) Khoảng trống nghiên cứu; (5) Hướng mở.
- **BẮT BUỘC**: Mọi luận điểm phải gắn trích dẫn dạng [SOURCE: …] từ [EVIDENCE].
- Không bịa. Nếu thiếu bằng chứng, nêu rõ: "Không đủ bằng chứng từ corpus."
- Văn phong học thuật, súc tích, liền mạch; tránh trích dẫn thừa.

[USER TOPIC]
{topic}

[EVIDENCE]
{ev_dump}
{history_section}

BẮT ĐẦU VIẾT:
""".strip()


# =========================================================
# Claim–Evidence Matrix (bảng luận điểm–chứng cứ)
# =========================================================
def build_claim_evidence_prompt(
    question: str,
    evidence_blocks: List[Dict[str, Any]],
    *,
    max_rows: int = 8,
    **kwargs,
) -> str:
    rows = []
    for b in evidence_blocks[:max_rows]:
        txt = (b.get("text") or "").strip()
        meta = dict(b.get("meta") or {})
        if txt:
            rows.append(f"- EVID: {txt}\n  {_cite_tag(meta)}")
    ev_dump = "\n".join(rows) if rows else "(no evidence)"
    return f"""
Nhiệm vụ: Lập **bảng luận điểm–chứng cứ** (Claim–Evidence Matrix) cho câu hỏi sau,
bảo đảm mỗi luận điểm có trích dẫn [SOURCE: …] và 1 câu bình luận phương pháp (vì sao dẫn chứng phù hợp).

[CÂU HỎI]
{question}

[DỮ LIỆU]
{ev_dump}

Định dạng đầu ra (markdown):

| Luận điểm | Dẫn chứng (trích ngắn) | Nguồn | Bình luận phương pháp |
|---|---|---|---|
| ... | "..." | [SOURCE: ...] | ... |
""".strip()


# =========================================================
# Counterargument mode (phản biện – “hai vế”)
# =========================================================
def build_counterargument_prompt(
    thesis: str,
    evidence_blocks: List[Dict[str, Any]],
    *,
    min_pairs: int = 2,
    **kwargs,
) -> str:
    ev = []
    for b in evidence_blocks[:min_pairs * 3]:
        t = (b.get("text") or "").strip()
        if t:
            ev.append(f"- {t}  {_cite_tag(dict(b.get('meta') or {}))}")
    ev_dump = "\n".join(ev) if ev else "(no evidence)"
    return f"""
Đề bài: Viết phân tích hai vế cho luận đề sau, mỗi vế **đều phải** dùng [SOURCE: …].
- Vế A (Ủng hộ luận đề) → Dẫn chứng → Lập luận.
- Vế B (Phản biện luận đề) → Dẫn chứng → Lập luận.
- Kết đoạn: đánh giá cân bằng (khi nào A đúng hơn, khi nào B thuyết phục).

[LUẬN ĐỀ]
{thesis}

[EVIDENCE]
{ev_dump}

Bắt đầu:
""".strip()


# =========================================================
# Quote verification checklist (kiểm tra trích dẫn)
# =========================================================
def build_quote_verification_prompt(
    answer_draft: str,
    **kwargs,
) -> str:
    return f"""
Bạn là biên tập viên khoa học. Hãy kiểm tra bản thảo dưới đây bằng **checklist 6 mục**:
1) Có trích nguyên văn câu thơ? 2) Có sai chữ/dấu? 3) Có cắt ghép làm đổi nghĩa?
4) Có gắn [SOURCE] đúng vị trí (Lstart–Lend) hoặc offset? 5) Có suy diễn vượt bằng chứng?
6) Các kết luận chính đều có ít nhất 1 [SOURCE]?

[ANSWER DRAFT]
{answer_draft}

Trả về: Danh sách mục đạt/chưa đạt + đề xuất sửa ngắn gọn.
""".strip()


# =========================================================
# Evidence-first outline (từ chứng cứ → dàn ý)
# =========================================================
def build_evidence_outline_prompt(
    question: str,
    evidence_blocks: List[Dict[str, Any]],
    *,
    max_points: int = 6,
    **kwargs,
) -> str:
    ev = []
    for b in evidence_blocks[:max_points * 2]:
        t = (b.get("text") or "").strip()
        if t:
            ev.append(f"- {t}  {_cite_tag(dict(b.get('meta') or {}))}")
    ev_dump = "\n".join(ev) if ev else "(no evidence)"

    return f"""
Nhiệm vụ: Sắp xếp các **evidence** thành dàn ý trả lời cho câu hỏi, mỗi mục trong dàn ý gắn [SOURCE: …].
Sau đó ghi 1 câu **tiểu kết** cho từng mục.

[CÂU HỎI]
{question}

[EVIDENCE]
{ev_dump}

Đầu ra (markdown):
- I. Luận điểm 1 — [SOURCE: …]
  - Dẫn chứng: …
  - Tiểu kết: …
- II. Luận điểm 2 — [SOURCE: …]
  ...
""".strip()
