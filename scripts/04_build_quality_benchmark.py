"""Generate the deterministic 260-case Kiều Bot quality benchmark."""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.poem_tools import all_poem_lines

OUTPUT = ROOT / "data" / "eval" / "quality_v1.jsonl"
MOTIFS = ROOT / "data" / "interim" / "poem" / "motifs.jsonl"
FACTS = ROOT / "data" / "fag" / "facts.json"

ANALYSIS_PROMPTS = (
    "Phân tích quan niệm tài mệnh tương đố trong phần mở đầu Truyện Kiều.",
    "Bình giảng nghệ thuật tả cảnh ngụ tình trong đoạn Kiều ở lầu Ngưng Bích.",
    "Phân tích diễn biến tâm lý Thúy Kiều trong đoạn Trao duyên.",
    "Làm rõ giá trị nhân đạo qua quyết định bán mình chuộc cha của Kiều.",
    "Phân tích nghệ thuật xây dựng chân dung Thúy Vân và Thúy Kiều.",
    "Nhận xét hình tượng người anh hùng Từ Hải.",
    "Phân tích ngôn ngữ đối thoại của Hoạn Thư trong màn báo ân báo oán.",
    "Làm rõ vai trò của thiên nhiên trong việc biểu đạt nội tâm Kiều.",
    "Phân tích giá trị hiện thực của Truyện Kiều qua thế lực đồng tiền.",
    "Nhận xét cách Nguyễn Du miêu tả nội tâm nhân vật.",
    "Phân tích ý nghĩa chữ tâm trong lời kết Truyện Kiều.",
    "Bình luận mối quan hệ giữa tài năng và số phận của Thúy Kiều.",
    "Phân tích hình ảnh người phụ nữ trong Truyện Kiều.",
    "Làm rõ nghệ thuật sử dụng điển cố của Nguyễn Du.",
    "Phân tích nhịp điệu lục bát trong một đoạn thơ giàu cảm xúc.",
    "Nhận xét sự kết hợp giữa tự sự và trữ tình trong Truyện Kiều.",
    "Phân tích vai trò của Đạm Tiên đối với dự báo số phận Kiều.",
    "Bình luận bi kịch tình yêu giữa Kiều và Kim Trọng.",
    "Phân tích ý nghĩa của màn đoàn viên cuối tác phẩm.",
    "Đánh giá đóng góp của Nguyễn Du đối với ngôn ngữ văn học dân tộc.",
)

COMPARE_PROMPTS = (
    "So sánh vẻ đẹp của Thúy Kiều và Thúy Vân.",
    "So sánh Kim Trọng và Từ Hải trong mối quan hệ với Thúy Kiều.",
    "So sánh bút pháp tả cảnh trong Cảnh ngày xuân và Kiều ở lầu Ngưng Bích.",
    "So sánh Mã Giám Sinh và Sở Khanh.",
    "So sánh thái độ của Kiều khi báo ân và khi báo oán.",
    "So sánh tâm trạng Kiều lúc trao duyên và lúc đoàn viên.",
    "So sánh ước lệ tượng trưng và tả thực trong Truyện Kiều.",
    "So sánh vai trò của Giác Duyên và Đạm Tiên.",
    "So sánh không gian lầu xanh và không gian đoàn viên.",
    "So sánh cách Nguyễn Du khắc họa Hoạn Thư và Tú Bà.",
)

AMBIGUOUS = (
    "Giải thích đoạn này giúp mình.",
    "Câu ấy có ý nghĩa gì?",
    "Phân tích nhân vật đó.",
    "Đoạn sau nói gì?",
    "Tìm câu thơ mình vừa nhắc.",
    "Tại sao lại như vậy?",
    "Điển tích ấy từ đâu?",
    "Bình câu này nhé.",
    "Sau đó chuyện gì xảy ra?",
    "Hai người ấy có quan hệ gì?",
)

OOD = (
    "Giá Bitcoin hôm nay là bao nhiêu?",
    "Viết cho tôi chương trình Python sắp xếp mảng.",
    "Dự báo thời tiết Hà Nội ngày mai.",
    "Ai đang là tổng thống Hoa Kỳ?",
    "Tư vấn mua điện thoại chơi game.",
    "Kết quả trận bóng đá tối qua là gì?",
    "Giải phương trình x bình phương bằng 4.",
    "Hướng dẫn đầu tư chứng khoán.",
    "Tóm tắt Chiến tranh thế giới thứ hai.",
    "Viết email xin nghỉ phép bằng tiếng Anh.",
)

TERMS = (
    "quạt nồng ấp lạnh",
    "sân Lai",
    "Lam Kiều",
    "duyên Tần Tấn",
    "đoạn trường",
    "thanh lâu",
    "hồng nhan",
    "bạc mệnh",
    "phong tình cổ lục",
    "cầm kỳ thi họa",
    "quốc sắc thiên hương",
    "khuôn trăng",
    "nét ngài",
    "hương nguyền",
    "tơ duyên",
    "Đạm Tiên",
    "Đạp Thanh",
    "Tiền Đường",
    "Quan Âm Các",
    "tài mệnh tương đố",
)


def _without_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).replace("đ", "d")


def _case(case_id: str, category: str, query: str, judge: str, **expected) -> dict:
    return {"id": case_id, "category": category, "query": query, "judge": judge, **expected}


def build_cases() -> list[dict]:
    lines = all_poem_lines()
    cases: list[dict] = []

    # 80 exact verse lookups distributed across the complete work.
    for index in range(80):
        number = 1 + round(index * (len(lines) - 1) / 79)
        line = lines[number - 1]
        templates = ("Trích câu {n}.", "Cho tôi nguyên văn câu số {n}.", "Câu {n} của Truyện Kiều là gì?", "Đọc câu {n}.")
        query = templates[index % len(templates)].format(n=number)
        cases.append(
            _case(
                f"verse-{number:04d}",
                "exact_verse",
                query,
                "exact_poem",
                gold_line_start=number,
                gold_line_end=number,
                gold_text=line.text,
            )
        )

    # 30 intentionally damaged quotations must be repaired to canonical text.
    quote_candidates = [line for line in lines if not any(mark in line.text for mark in ('"', "“", "”"))]
    for index in range(30):
        line = quote_candidates[round(index * (len(quote_candidates) - 1) / 29)]
        number = line.number
        damaged = _without_accents(line.text)
        cases.append(
            _case(
                f"quote-fix-{number:04d}",
                "quote_correction",
                f'Có phải nguyên văn là "{damaged}" không?',
                "quote_verifier",
                supplied_quote=damaged,
                gold_line_start=number,
                gold_line_end=number,
                gold_text=line.text,
            )
        )

    # 30 contiguous range lookups exercise line positions and ordering.
    for index in range(30):
        start = 1 + round(index * (len(lines) - 2) / 29)
        end = min(start + 1, len(lines))
        cases.append(
            _case(
                f"range-{start:04d}-{end:04d}",
                "line_position",
                f"Trích câu {start}-{end}, không giải thích thêm.",
                "exact_poem",
                gold_line_start=start,
                gold_line_end=end,
                gold_text="\n".join(item.text for item in lines[start - 1 : end]),
            )
        )

    motifs = json.loads(MOTIFS.read_text(encoding="utf-8"))
    for index in range(20):
        motif = motifs[index % len(motifs)]
        start, end = motif["range"]
        query = (
            f"Sự kiện hoặc chủ đề chính từ câu {start} đến {end} là gì?"
            if index < len(motifs)
            else f"Đoạn {motif['motif']} nằm ở khoảng câu nào?"
        )
        cases.append(
            _case(
                f"timeline-{index + 1:02d}",
                "plot_timeline",
                query,
                "retrieval",
                gold_line_start=start,
                gold_line_end=end,
                gold_contains=motif["motif"],
            )
        )

    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    character_facts = [item for item in facts if any("là ai" in pattern for pattern in item["patterns"])]
    for index in range(30):
        fact = character_facts[index % len(character_facts)]
        patterns = fact["patterns"]
        query = patterns[index % len(patterns)]
        cases.append(
            _case(
                f"character-{index + 1:02d}",
                "character_relationship",
                query,
                "contains",
                gold_text=fact["answer"],
                expected_terms=[word for word in fact["answer"].split()[:8]],
            )
        )

    for index, term in enumerate(TERMS, start=1):
        cases.append(
            _case(
                f"term-{index:02d}",
                "archaic_allusion",
                f'Giải nghĩa từ hoặc điển tích "{term}" trong Truyện Kiều.',
                "retrieval_or_human",
                gold_contains=term,
            )
        )

    rubric = ["answer_directly", "use_primary_evidence", "separate_fact_and_interpretation", "no_fabricated_quote"]
    for index, query in enumerate(ANALYSIS_PROMPTS, start=1):
        cases.append(_case(f"analysis-{index:02d}", "literary_analysis", query, "human_rubric", rubric=rubric))
    for index, query in enumerate(COMPARE_PROMPTS, start=1):
        cases.append(_case(f"compare-{index:02d}", "compare_passages", query, "human_rubric", rubric=rubric))
    for index, query in enumerate(AMBIGUOUS, start=1):
        cases.append(_case(f"ambiguous-{index:02d}", "ambiguous", query, "route", expected_flow="clarification"))
    for index, query in enumerate(OOD, start=1):
        cases.append(_case(f"ood-{index:02d}", "out_of_scope", query, "route", expected_flow="safe-refusal"))
    return cases


def main() -> None:
    cases = build_cases()
    if not 200 <= len(cases) <= 300:
        raise SystemExit(f"Benchmark size out of contract: {len(cases)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    counts = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    print(json.dumps({"cases": len(cases), "categories": counts, "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
