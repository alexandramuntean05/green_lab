"""Prompt templates and output parsing.

Drafter prompt   : adapted from Wang et al. [3], Appendix H (Figure 6).
Standard RAG     : adapted from Wang et al. [3], Appendix I (Figure 8, prompt for
                   instruction-tuned LMs), extended with the same
                   "## Rationale / ## Response" output format as the drafter so
                   that both architectures deliver a response (alpha) and a
                   rationale (beta) and N_final is defined identically.
PubHealth task   : task instruction from Self-RAG (TASK_INST["fever"],
                   retrieval_lm/utils.py), as used for this benchmark by [3].
Verifier text    : [Q, alpha, beta, R, "Yes"] as in Wang et al. [3], Sec. 3.3.
"""
from __future__ import annotations

import re

PUBHEALTH_INST = ("Is the following statement correct or not? "
                  "Say true if it's correct; otherwise say false.")

OUTPUT_FORMAT = ("Answer in exactly this format:\n"
                 "## Rationale: <one or two sentences grounded in the evidence>\n"
                 "## Response: <the answer only>")


def task_instruction(question: str, benchmark: str) -> str:
    if benchmark == "pubhealth":
        return f"{PUBHEALTH_INST}\n\n### Input:\n{question}"
    return question


def format_docs(docs) -> str:
    return "\n".join(f"[{i + 1}] {d['title']}\n{d['text']}" for i, d in enumerate(docs))


def drafter_prompt(question: str, docs_text: str, benchmark: str) -> str:
    return ("Response to the instruction. Also provide rationale for your response.\n"
            f"## Instruction: {task_instruction(question, benchmark)}\n"
            f"## Evidence:\n{docs_text}\n\n{OUTPUT_FORMAT}")


def standard_rag_prompt(question: str, docs_text: str, benchmark: str) -> str:
    return ("Below is an instruction that describes a task. Write a response for it and "
            "state your explanation supporting your response.\n"
            f"### Instruction: {task_instruction(question, benchmark)}\n"
            f"### Evidence:\n{docs_text}\n\n{OUTPUT_FORMAT}")


_RAT = re.compile(r"##\s*Rationale\s*:?(.*?)(?=##\s*Response|\Z)", re.S | re.I)
_RESP = re.compile(r"##\s*Response\s*:?(.*)", re.S | re.I)


def parse_output(text: str):
    """Return (rationale, response, char_span_rationale, char_span_response, ok).

    Spans are character offsets in `text` used to attribute generated tokens to
    beta (rationale) and alpha (response). If the format is not followed, the
    whole output is treated as the response and ok=False is recorded.
    """
    m_rat, m_resp = _RAT.search(text), _RESP.search(text)
    if m_resp is None:
        return "", text.strip(), (0, 0), (0, len(text)), False
    rat_span = (m_rat.start(1), m_rat.end(1)) if m_rat else (0, 0)
    resp_span = (m_resp.start(1), m_resp.end(1))
    return (text[rat_span[0]:rat_span[1]].strip(), text[resp_span[0]:resp_span[1]].strip(),
            rat_span, resp_span, m_rat is not None)


# ------------------------------------------------------------------ correctness
def match(prediction: str, golds) -> int:
    """Self-RAG metrics.match: 1 if any gold string is contained in the prediction."""
    return int(any(g in prediction for g in golds))


def pubhealth_label(prediction: str):
    m = re.search(r"\b(true|false)\b", prediction.lower())
    return m.group(1) if m else None


def is_correct(prediction: str, golds, benchmark: str) -> dict:
    """`correct` follows the metric of the benchmark as used by [3] and Self-RAG:
    TriviaQA: gold alias contained in the response (case-sensitive `match`);
    PubHealth: accuracy of the extracted true/false label.
    `correct_norm` is a case-insensitive variant reported for robustness only."""
    if benchmark == "pubhealth":
        label = pubhealth_label(prediction)
        c = int(label is not None and label == golds[0])
        return {"correct": c, "correct_norm": c, "pred_label": label}
    return {"correct": match(prediction, golds),
            "correct_norm": match(prediction.lower(), [g.lower() for g in golds]),
            "pred_label": None}
