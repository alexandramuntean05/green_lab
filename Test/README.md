# Speculative RAG — minimal reproduction (for glg4, RTX 3070 8GB)

This is a feasibility check, not the final experiment code: it exists to answer
Winston's question ("what's possible on this GPU") before the team commits to a
model/parameter configuration in `sections_4_5.tex`.

## Deviations from Wang et al. (documented on purpose, see docstring in the script)

1. Drafter is the off-the-shelf Gemma-2-2B-it, not the paper's fine-tuned specialist
   (no checkpoint is public).
2. Document clustering uses `all-MiniLM-L6-v2` + K-Means instead of the paper's
   InBedder-Roberta + K-Means (same algorithm shape, smaller embedding model).
3. The `m` drafts run **sequentially**, not as `m` parallel endpoints — a single
   8GB GPU can't host many parallel endpoints. This is the deviation to flag back
   to the team once you know what m the GPU can actually sustain.
4. Verification is "ask the verifier LLM to pick/synthesize the best draft" rather
   than the paper's trained self-consistency scorer. Good enough for a feasibility
   check, not for the final accuracy numbers.

## Step 1 — sanity-check the pipeline (works anywhere, no GPU, no downloads)

```bash
pip install -r requirements.txt   # only needed for the real run, see below
python speculative_rag.py --mock
```

This uses fake model outputs to verify the control flow (clustering, prompt
building, timing, result writing) without touching a GPU. I already ran this in
my own sandbox and it works.

## Step 2 — real run on glg4

```bash
ssh -J gl_thegreenteam@145.108.225.3:42224 gl_thegreenteam@glg4
# on glg4:
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python speculative_rag.py --m 5 --k 2
```

The first run downloads Mistral-7B-Instruct-v0.2 and Gemma-2-2B-it from Hugging
Face (a few GB each) — make sure the node has internet access and enough disk.
Watch `nvidia-smi` in another terminal while it runs.

## Step 3 — find the m ceiling

Run with increasing `m` (e.g. `--m 1`, `--m 3`, `--m 5`, `--m 8`) and note where
you hit an out-of-memory error. Report that number back to Winston — that's the
answer to "can the 3070 afford m>4".

## What I could NOT verify myself

I don't have a GPU in my own environment, so I could only test `--mock` mode.
I have not confirmed that Mistral-7B-Instruct-v0.2 + Gemma-2-2B-it actually fit
together in 8GB with 4-bit quantization — that's exactly what running this on
glg4 will tell you. If it OOMs, the first things to try are: drop `--k`, reduce
`max_new_tokens` in `LLM.generate`, or load the drafter in 8-bit only when the
verifier isn't active.
