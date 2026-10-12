"""Tries prompt variants (one-shot example yes/no, prefill yes/no) on the warm-up
questions and prints how often the output format is followed.

    CUDA_VISIBLE_DEVICES=0 python src/format_probe.py
Stop the rag_server first.
"""
import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import drafter_prompt, format_docs, parse_output, standard_rag_prompt  # noqa: E402
from rag_pipeline import RagSystem  # noqa: E402


def main():
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "config.json"))
    S = RagSystem(cfg, device=os.environ.get("DEVICE", "cuda:0"))
    items = []
    for b in ("pubhealth", "triviaqa"):
        with open(os.path.join(cfg["data"]["prepared_dir"], f"{b}_warmup.jsonl")) as fh:
            items += [json.loads(l) for l in fh if l.strip()]
    max_new = cfg["generation"]["max_new_tokens"]
    for one_shot, prefill in itertools.product([False, True], ["", "## Rationale:"]):
        S.one_shot, S.prefill = one_shot, prefill
        for role in ("drafter", "verifier"):
            ok, n_gen, trunc, example = [], [], [], None
            for it in items:
                if role == "drafter":   # 2 documents
                    p = drafter_prompt(it["question"], format_docs(it["ctxs"][:2]), it["benchmark"], one_shot)
                    g = S._generate(S.drafter, S.d_tok, [p])[0]
                else:                   # all R documents
                    p = standard_rag_prompt(it["question"], format_docs(it["ctxs"][:S.R]), it["benchmark"], one_shot)
                    g = S._generate(S.verifier, S.v_tok, [p])[0]
                good = parse_output(prefill + g["text"])[4]
                ok.append(good)
                n_gen.append(g["n_gen"])
                trunc.append(g["n_gen"] >= max_new)
                if example is None:
                    example = (prefill + g["text"])[:300].replace("\n", " | ")
            print(f"one_shot={one_shot!s:5} prefill={prefill!r:16} {role:8} format_ok={statistics.mean(ok):.2f} "
                  f"mean_gen={statistics.mean(n_gen):5.1f} truncated={statistics.mean(trunc):.2f}")
            print(f"    e.g. {example}")


if __name__ == "__main__":
    main()
