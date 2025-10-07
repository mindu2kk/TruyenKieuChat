# scripts/debug_poem.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path

# Thêm project root (thư mục chứa folder "app/") vào sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.router import parse_poem_request, route_intent
from app.poem_tools import poem_ready, get_opening, get_range

def main():
    tests = [
        "Cho tôi 10 câu đầu Truyện Kiều",
        "trích 30 câu đầu",
        "câu 241-260",
        "từ câu 100 đến câu 110",
        "xin mười câu đầu",
    ]
    print("poem_ready:", poem_ready())
    for t in tests:
        print("Q:", t)
        print("  intent:", route_intent(t))
        print("  spec:", parse_poem_request(t))

    if poem_ready():
        print("\nSample opening(5):")
        for i, ln in enumerate(get_opening(5), 1):
            print(f"{i:>4}: {ln}")
        print("\nSample range 241–246:")
        for i, ln in enumerate(get_range(241, 246), 241):
            print(f"{i:>4}: {ln}")
    else:
        print("\n⚠️  Chưa có data/interim/poem/poem.txt (mỗi câu 1 dòng).")

if __name__ == "__main__":
    main()
