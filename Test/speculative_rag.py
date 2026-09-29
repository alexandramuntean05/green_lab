#!/usr/bin/env python3
"""
Minimal reproduction of Speculative RAG (Wang et al., ICLR 2025, arXiv:2407.08223)
for a single 8GB-class GPU (e.g. RTX 3070).

What this does, and where it deviates from the paper (documented, not hidden):
  - Verifier / Standard-RAG generator: Mistral-7B-Instruct-v0.2, 4-bit (NF4) quantized.
  - Drafter: Gemma-2-2B-it, 4-bit (NF4) quantized. The paper fine-tunes a "specialist"
    drafter on ~40k instances; no checkpoint is public, so this uses the off-the-shelf
    instruction-tuned model instead, prompted to emit an answer + rationale.
  - Document clustering: the paper uses InBedder-Roberta embeddings + K-Means. This
    script uses sentence-transformers/all-MiniLM-L6-v2 (much smaller, runs on the same
    GPU/CPU) + K-Means. Same algorithm shape, different (documented) embedding model.
  - The paper launches m drafts as PARALLEL endpoints. On a single 8GB card that does
    not fit once m is more than a few, so this script runs the m drafts SEQUENTIALLY
    (a for-loop). Functionally equivalent output, not literally parallel -- this is the
    deviation you should mention when reporting m/k feasibility limits back to the team.
  - Verification: the paper trains a scoring mechanism (self-consistency + confidence).
    This script asks the verifier LLM to read all m drafts and pick/synthesize the final
    answer in one pass, which is a reasonable stand-in for a sanity-check reproduction
    but is NOT the paper's exact scoring method.

Usage:
  # Fast, no GPU / no model download -- checks the control flow and prompt building only.
  python speculative_rag.py --mock

  # Real run on the cluster node (needs a CUDA GPU with bitsandbytes support).
  python speculative_rag.py --m 5 --k 2

Run with --mock first (works anywhere, including this sandbox) to catch bugs before
burning GPU time on glg4.
"""
import argparse
import dataclasses
import json
import sys
import time
from typing import List


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Query:
    qid: str
    question: str
    # For PubHealth-style claim verification the "question" is the claim; for
    # TriviaQA-style it's a real question. docs is the pre-retrieved top-N context.
    docs: List[str]
    gold_answer: str = ""


# A tiny hardcoded set so the script is runnable standalone without first wiring up
# the real PubHealth / TriviaQA loaders. Swap load_queries() for a real loader once
# the retrieval cache described in Section 4.1 of the assignment write-up exists.
SAMPLE_QUERIES: List[Query] = [
    Query(
        qid="ph_001",
        question="Drinking bleach cures COVID-19.",
        docs=[
            "Bleach and other disinfectants are toxic when ingested and can cause severe "
            "internal injury, chemical burns, and death. Health authorities have "
            "repeatedly warned against consuming any form of bleach.",
            "There is no scientific evidence that ingesting bleach treats or cures "
            "COVID-19 or any other viral infection. The CDC and WHO explicitly warn "
            "against this practice.",
            "COVID-19 is caused by the SARS-CoV-2 virus and is treated with antiviral "
            "medication, supportive care, and vaccination for prevention; disinfectants "
            "are for surfaces, not the human body.",
        ],
        gold_answer="false",
    ),
    Query(
        qid="tq_001",
        question="What is the largest ocean on Earth by surface area?",
        docs=[
            "The Pacific Ocean is the largest and deepest of Earth's five oceanic "
            "divisions, covering about 63 million square miles.",
            "The Atlantic Ocean is the second-largest ocean, separating the Americas "
            "from Europe and Africa.",
            "The Indian Ocean is the third-largest of the world's oceanic divisions.",
        ],
        gold_answer="Pacific Ocean",
    ),
]


def load_queries(mock: bool) -> List[Query]:
    if mock:
        return SAMPLE_QUERIES
    # TODO(team): replace with the real cached retrieval output described in
    # Section 4.1 of sections_4_5.tex (100 PubHealth + 100 TriviaQA queries,
    # top-10 docs from Contriever-MS MARCO). Keeping this as a stub on purpose --
    # wiring up the actual dataset loaders is a separate task from proving the
    # pipeline fits on an 8GB GPU.
    return SAMPLE_QUERIES


# ---------------------------------------------------------------------------
# Clustering: partition k retrieved docs into m subsets (paper: Sec 3, InBedder+K-Means)
# ---------------------------------------------------------------------------

def cluster_documents(docs: List[str], m: int, k: int, mock: bool) -> List[List[str]]:
    """Return m subsets of (up to) k documents each, clustered by semantic similarity.

    Falls back to a deterministic round-robin split in --mock mode (no embedding
    model loaded) or when there are fewer documents than m*k.
    """
    if mock or len(docs) < 2:
        return _round_robin_split(docs, m, k)

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError as e:
        print(f"[warn] embedding/clustering deps missing ({e}); falling back to "
              f"round-robin split", file=sys.stderr)
        return _round_robin_split(docs, m, k)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs)
    n_clusters = min(m, len(docs))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(embeddings)

    clusters: List[List[str]] = [[] for _ in range(n_clusters)]
    for doc, label in zip(docs, labels):
        clusters[label].append(doc)

    # Pad/truncate each cluster to k docs by borrowing round-robin from the full pool
    # so every draft gets a comparable amount of context, and pad out to m clusters
    # if K-Means produced fewer than m (can happen when len(docs) < m).
    while len(clusters) < m:
        clusters.append([])
    for c in clusters:
        i = 0
        while len(c) < k and docs:
            c.append(docs[i % len(docs)])
            i += 1
    return [c[:k] for c in clusters[:m]]


def _round_robin_split(docs: List[str], m: int, k: int) -> List[List[str]]:
    if not docs:
        return [[] for _ in range(m)]
    subsets = []
    for i in range(m):
        subset = [docs[(i * k + j) % len(docs)] for j in range(k)]
        subsets.append(subset)
    return subsets


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = (
    "You are a careful assistant. Read the provided documents and answer the "
    "question using only the evidence they contain. Then give a one-sentence "
    "rationale citing which document(s) support your answer."
)

VERIFY_SYSTEM = (
    "You are a strict verifier. You will see a question and several candidate "
    "answers, each with its own rationale, produced by a smaller assistant model. "
    "Pick the single best-supported answer, or synthesize a corrected answer if "
    "all candidates are flawed. Reply with just the final answer, nothing else."
)


def build_draft_prompt(question: str, docs: List[str]) -> str:
    ctx = "\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(docs))
    return (
        f"{DRAFT_SYSTEM}\n\nDocuments:\n{ctx}\n\nQuestion: {question}\n"
        f"Answer (with a one-sentence rationale):"
    )


def build_verify_prompt(question: str, drafts: List[str]) -> str:
    listing = "\n".join(f"[Draft {i+1}] {d}" for i, d in enumerate(drafts))
    return (
        f"{VERIFY_SYSTEM}\n\nQuestion: {question}\n\nCandidate drafts:\n{listing}\n\n"
        f"Final answer:"
    )


# ---------------------------------------------------------------------------
# Model wrapper -- real HF models, or a mock that echoes deterministic text
# ---------------------------------------------------------------------------

class LLM:
    """Thin wrapper so the pipeline code is identical in --mock and real runs."""

    def __init__(self, model_name: str, mock: bool, label: str):
        self.model_name = model_name
        self.mock = mock
        self.label = label
        self.model = None
        self.tokenizer = None
        if not mock:
            self._load_real()

    def _load_real(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[{self.label}] loading {self.model_name} (4-bit NF4)...", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        if self.mock:
            # Deterministic stand-in so --mock is useful for testing prompt
            # plumbing and control flow without any GPU or network access.
            return f"[MOCK OUTPUT from {self.label} for prompt of {len(prompt)} chars]"

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding, matches the paper (T=0)
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        return text.strip()


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def run_standard_rag(query: Query, verifier: LLM) -> dict:
    t0 = time.perf_counter()
    ctx = "\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(query.docs))
    prompt = (
        f"Answer the question using only the documents below.\n\nDocuments:\n{ctx}\n\n"
        f"Question: {query.question}\nAnswer:"
    )
    answer = verifier.generate(prompt)
    return {
        "qid": query.qid,
        "mode": "standard",
        "answer": answer,
        "latency_s": time.perf_counter() - t0,
    }


def run_speculative_rag(query: Query, drafter: LLM, verifier: LLM,
                         m: int, k: int, mock: bool) -> dict:
    t0 = time.perf_counter()
    clusters = cluster_documents(query.docs, m, k, mock)

    # Sequential drafting (see module docstring: paper does this in parallel across
    # m endpoints; a single 8GB card can't host that many endpoints at once).
    drafts = []
    for cluster in clusters:
        prompt = build_draft_prompt(query.question, cluster)
        drafts.append(drafter.generate(prompt, max_new_tokens=120))

    verify_prompt = build_verify_prompt(query.question, drafts)
    final_answer = verifier.generate(verify_prompt, max_new_tokens=60)

    return {
        "qid": query.qid,
        "mode": f"speculative_m{m}_k{k}",
        "num_drafts": len(drafts),
        "drafts": drafts,
        "answer": final_answer,
        "latency_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def report_gpu_memory(mock: bool):
    if mock:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            }
    except ImportError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true",
                     help="Run with fake LLM outputs; no GPU or model download needed. "
                          "Use this to sanity-check the pipeline before a real GPU run.")
    ap.add_argument("--m", type=int, default=5, help="number of speculative drafts")
    ap.add_argument("--k", type=int, default=2, help="documents per draft")
    ap.add_argument("--verifier-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--drafter-model", default="google/gemma-2-2b-it")
    ap.add_argument("--out", default="results.jsonl", help="output file (JSON lines)")
    args = ap.parse_args()

    queries = load_queries(args.mock)
    print(f"Loaded {len(queries)} queries. mock={args.mock} m={args.m} k={args.k}")

    verifier = LLM(args.verifier_model, args.mock, label="verifier")
    drafter = LLM(args.drafter_model, args.mock, label="drafter")

    results = []
    for q in queries:
        std = run_standard_rag(q, verifier)
        spec = run_speculative_rag(q, drafter, verifier, args.m, args.k, args.mock)
        results.append(std)
        results.append(spec)
        print(f"[{q.qid}] standard: {std['answer'][:80]!r} "
              f"({std['latency_s']:.2f}s)")
        print(f"[{q.qid}] speculative(m={args.m},k={args.k}): "
              f"{spec['answer'][:80]!r} ({spec['latency_s']:.2f}s)")

    mem = report_gpu_memory(args.mock)
    if mem:
        print(f"Peak GPU memory: {mem}")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} records to {args.out}")


if __name__ == "__main__":
    main()
