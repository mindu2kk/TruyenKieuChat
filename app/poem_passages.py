"""Small, reviewed narrative anchors for close reading of canonical poem lines.

These records are not generated answers. They are compact story facts checked
against the canonical poem so a follow-up about a numbered passage does not
invent a character, location, or chronology from semantically similar chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PassageContext:
    line_start: int
    line_end: int
    summary: str
    close_reading: str


PASSAGE_CONTEXTS = (
    PassageContext(
        1789,
        1802,
        (
            "Mạch kể chuyển sang Thúc Sinh ở Lâm Truy. Sau vụ hỏa hoạn giả do Hoạn Thư sắp đặt, "
            "Thúc Sinh tưởng Thúy Kiều đã chết; thực ra nàng bị bắt về nhà Hoạn Thư và phải làm thị tì. "
            "Một năm trôi qua, Thúc Sinh vẫn thương nhớ Kiều trong cảnh phòng không. Đến câu 1799-1800, "
            "chàng nhớ quê và lên đường về Vô Tích, chưa biết rằng sắp gặp lại Kiều ngay trong nhà Hoạn Thư. "
            "Vì vậy, 'sầu dài' trước hết chỉ nỗi sầu và thời gian chờ đợi của Thúc Sinh, không phải tâm trạng Kiều."
        ),
        (
            "**Bối cảnh:** Mạch kể đang nói về Thúc Sinh ở Lâm Truy. Sau vụ hỏa hoạn giả do Hoạn Thư "
            "sắp đặt, chàng tưởng Thúy Kiều đã chết, trong khi nàng thực ra bị bắt về làm thị tì. "
            "Một năm xa cách và thương nhớ đã trôi qua; ngay sau hai câu này, Thúc Sinh nhớ quê rồi trở về "
            "Vô Tích. Vì thế, **sầu dài** trước hết là nỗi sầu kéo dài của Thúc Sinh, không phải tâm trạng của Kiều.\n\n"
            "**Nghệ thuật:** Nguyễn Du nén cả vòng vận động của một năm vào hai câu lục bát: sen tàn gợi hạ qua, "
            "cúc nở báo thu tới, ngày ngắn gợi đông và rồi đông chuyển sang xuân. Nhịp 2/2/2 ở câu lục "
            "(**Sen tàn / cúc lại / nở hoa**) tạo cảm giác từng mùa nối bước; các thế đối **tàn – nở**, "
            "**sầu dài – ngày ngắn**, **đông – xuân** đặt thời gian tuần hoàn của thiên nhiên bên cạnh thời gian "
            "tâm lý bị kéo giãn bởi nhớ thương. Hai chữ **lại** và **đà** vừa chỉ vòng quay đều đặn, vừa gợi sự bất lực: "
            "cảnh vật đổi thay mà nỗi sầu vẫn chưa dứt."
        ),
    ),
)


def context_for_range(line_start: int, line_end: int) -> Optional[PassageContext]:
    start, end = sorted((int(line_start), int(line_end)))
    for passage in PASSAGE_CONTEXTS:
        if passage.line_start <= start and end <= passage.line_end:
            return passage
    return None


__all__ = ["PassageContext", "PASSAGE_CONTEXTS", "context_for_range"]
