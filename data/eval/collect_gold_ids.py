# -*- coding: utf-8 -*-
import json, importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
retr_path = ROOT / "scripts" / "04_retrieve.py"
spec = importlib.util.spec_from_file_location("retr", retr_path)
retr = importlib.util.module_from_spec(spec); sys.modules["retr"]=retr
assert spec and spec.loader
spec.loader.exec_module(retr)  # type: ignore

RETR_FILE = ROOT / "data" / "eval" / "retrieval.jsonl"

def main():
    items = [json.loads(l) for l in open(RETR_FILE,"r",encoding="utf-8") if l.strip()]
    for it in items:
        q = it["query"]
        hits = retr.retrieve_context(q, k=10, num_candidates=100)
        print("\n=== QUERY:", q)
        for i,h in enumerate(hits):
            mid = (h.get("meta") or {}).get("id")
            src = (h.get("meta") or {}).get("source")
            sc  = h.get("score")
            print(f"{i+1:>2}. id={mid}  score={sc:.3f}  src={src}")
        print("-> Lấy 1-2 id phù hợp và thay vào gold_ctx_ids trong retrieval.jsonl")

if __name__=="__main__":
    main()
