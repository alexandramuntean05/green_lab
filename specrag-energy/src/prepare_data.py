"""Samples the questions for each benchmark.

Input: Self-RAG eval files (documents already retrieved with Contriever, field `ctxs`).
Output: data/prepared/<benchmark>.jsonl, <benchmark>_warmup.jsonl, sampling_log.json
Random sample with a fixed seed. Questions need a gold answer and at least R documents.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import standard_rag_prompt, format_docs  # noqa: E402


def load_jsonl(path):
    with open(path) as fh:
        text = fh.read().strip()
    if text.startswith("["):  # json array instead of jsonl
        return json.loads(text)
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def select_ctxs(ctxs, mode: str):
    # scored_only: only documents with a Contriever score, sorted by score
    # (the TriviaQA file has extra unscored documents at ranks 6-10)
    # file_order: keep the order of the file
    if mode == "file_order":
        return ctxs
    scored = [c for c in ctxs if c.get("score") is not None]
    return sorted(scored, key=lambda c: float(c["score"]), reverse=True)


def normalise(item: dict, benchmark: str, idx: int, ctx_mode: str = "scored_only"):
    question = item.get("question") or item.get("claim") or item.get("instruction") or item.get("input")
    golds = item.get("answers") or item.get("golds") or item.get("output") or item.get("label")
    if isinstance(golds, str):
        golds = [golds]
    if benchmark == "pubhealth" and golds:
        golds = [str(g).strip().lower() for g in golds]
    ctxs = select_ctxs(item.get("ctxs") or item.get("top_contexts") or [], ctx_mode)
    ctxs = [{"title": c.get("title", ""), "text": c.get("text", "")} for c in ctxs]
    qid = str(item.get("id", item.get("question_id", f"{benchmark}-{idx}")))
    return {"qid": qid, "benchmark": benchmark, "question": question, "golds": golds, "ctxs": ctxs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF tokenizer (verifier) used for the context-window check")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    R = cfg["retrieval"]["R"]
    Q = cfg["sampling"]["questions_per_benchmark"]
    W = cfg["sampling"]["warmup_questions_per_benchmark"]
    seed = cfg["sampling"]["seed"]
    out_dir = cfg["data"]["prepared_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tok, max_ctx = None, None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        max_ctx = tok.model_max_length
    max_new = cfg["generation"]["max_new_tokens"]

    ctx_mode = cfg["retrieval"].get("ctx_mode", "scored_only")
    log = {"seed": seed, "R": R, "Q": Q, "warmup": W, "ctx_mode": ctx_mode, "benchmarks": {}}
    for bench, key in (("pubhealth", "pubhealth_raw"), ("triviaqa", "triviaqa_raw")):
        raw = load_jsonl(cfg["data"][key])
        pool, excluded = [], {"no_gold": 0, "too_few_docs": 0, "too_long": 0, "duplicate_id": 0}
        seen = set()
        for i, item in enumerate(raw):
            it = normalise(item, bench, i, ctx_mode)
            if it["qid"] in seen:
                excluded["duplicate_id"] += 1
                continue
            seen.add(it["qid"])
            if not it["golds"] or not it["question"]:
                excluded["no_gold"] += 1
                continue
            if len(it["ctxs"]) < R:
                excluded["too_few_docs"] += 1
                continue
            it["ctxs"] = it["ctxs"][:R]
            if tok is not None:
                # check that the longest prompt (standard RAG) fits
                prompt = standard_rag_prompt(it["question"], format_docs(it["ctxs"]), bench)
                n = len(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                add_generation_prompt=True))
                if n + max_new > max_ctx:
                    excluded["too_long"] += 1
                    continue
            pool.append(it)

        rng = random.Random(f"{seed}-{bench}")
        if len(pool) < Q + W:
            raise SystemExit(f"{bench}: only {len(pool)} eligible questions, {Q + W} needed "
                             f"(excluded: {excluded}). With ctx_mode='scored_only' every passage "
                             f"needs a 'score' field.")
        chosen = rng.sample(pool, Q + W)
        sample, warm = chosen[:Q], chosen[Q:]
        for name, rows in ((bench, sample), (f"{bench}_warmup", warm)):
            with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log["benchmarks"][bench] = {"raw": len(raw), "eligible": len(pool), "excluded": excluded,
                                    "sampled_qids": [r["qid"] for r in sample],
                                    "warmup_qids": [r["qid"] for r in warm]}
        print(f"{bench}: raw={len(raw)} eligible={len(pool)} sampled={len(sample)} warmup={len(warm)} excluded={excluded}")

    with open(os.path.join(out_dir, "sampling_log.json"), "w") as fh:
        json.dump(log, fh, indent=2)


if __name__ == "__main__":
    main()
