#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test generation.py trực tiếp với API thật."""
import os, sys, time
os.environ.setdefault("GOOGLE_API_KEY", open(".env").read().split("GOOGLE_API_KEY=")[1].split()[0])

from app.generation import generate_answer_gemini, GenerationError

print("=== GENERATION TEST (real API) ===")
print(f"  Model: gemini-2.5-flash")

# Test 1: câu đơn giản
try:
    t0 = time.time()
    ans = generate_answer_gemini("Trả lời ngắn gọn: Truyện Kiều do ai sáng tác?", long_answer=False)
    elapsed = time.time() - t0
    ok = "Nguyễn Du" in ans or "nguyễn du" in ans.lower()
    print(f"  {'PASS' if ok else 'FAIL'} | T={elapsed:.1f}s | ans={ans[:80]!r}")
except GenerationError as e:
    print(f"  FAIL | GenerationError: {e}")
except Exception as e:
    print(f"  FAIL | Exception: {e}")

# Test 2: empty prompt guard
try:
    ans = generate_answer_gemini("   ")
    print(f"  WARN | empty prompt returned: {ans!r}")
except GenerationError as e:
    print(f"  PASS | empty prompt raised GenerationError (expected): {str(e)[:60]}")
except Exception as e:
    print(f"  FAIL | unexpected: {e}")

print("\nDone.")
