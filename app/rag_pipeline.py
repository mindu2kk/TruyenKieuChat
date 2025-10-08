# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional
import importlib.util, sys
from pathlib import Path
from rerank import rerank
from generation import generate_answer_gemini
from prompt_engineering import (
    DEFAULT_LONG_TOKEN_BUDGET,
    DEFAULT_SHORT_TOKEN_BUDGET,
    build_rag_synthesis_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
retr_path = ROOT / "scripts" / "04_retrieve.py"
spec = importlib.util.spec_from_file_location("retrieve_mod", retr_path)
retrieve_mod = importlib.util.module_from_spec(spec)
sys.modules["retrieve_mod"] = retrieve_mod
assert spec and spec.loader
spec.loader.exec_module(retrieve_mod)  # type: ignore
retrieve_context = retrieve_mod.retrieve_context

def answer_question(
    query: str,
    k: int = 5,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 120,
    synthesize: str | bool = "single",
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = True,
    long_answer: bool = False,
    history_text: str | None = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:

    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    if max_tokens is None:
        max_tokens = DEFAULT_LONG_TOKEN_BUDGET if long_answer else DEFAULT_SHORT_TOKEN_BUDGET

    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = build_rag_synthesis_prompt(query, [], history_text=history_text, long_answer=long_answer)
        return {"query": query, "prompt": prompt, "contexts": []}

    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = build_rag_synthesis_prompt(query, [], history_text=history_text, long_answer=long_answer)
        return {"query": query, "prompt": prompt, "contexts": []}

    hits = rerank(query, hits, top_k=k)
    prompt = build_rag_synthesis_prompt(query, hits, history_text=history_text, long_answer=long_answer)

    if synthesize and synthesize != "mapreduce":
        if force_quote:
            prompt += "\n\n[LƯU Ý] Nếu có câu thơ phù hợp, hãy trích 1–2 câu trong ngoặc kép."
        ans = generate_answer_gemini(
            prompt,
            model=gen_model,
            long_answer=long_answer,
            max_tokens=max_tokens,
        )
    else:
        ans = None

    out = {"query": query, "prompt": prompt, "contexts": hits}
    if ans:
        out["answer"] = ans
    return out