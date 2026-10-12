"""Prompts and output parsing.

Drafter and standard RAG prompts are based on the ones in the Speculative RAG
paper; both use the same Rationale/Response output format.
PubHealth instruction is taken from Self-RAG.
"""
from __future__ import annotations

import re

PUBHEALTH_INST = ("Is the following statement correct or not? "
                  "Say true if it's correct; otherwise say false.")

OUTPUT_FORMAT = ("Answer in exactly this format:\n"
                 "## Rationale: <one or two sentences grounded in the evidence>\n"
                 "## Response: <the answer only>")


# one-shot examples (written by us) to show the output format
DEMOS = {
    "triviaqa": ("### Example\n"
                 "## Instruction: Which river flows through the city of Vienna?\n"
                 "## Evidence:\n[1] Vienna\nVienna is the capital of Austria and lies on the river Danube.\n"
                 "## Rationale: The evidence states that Vienna lies on the Danube.\n"
                 "## Response: The Danube\n### End of example\n\n"),
    "pubhealth": ("### Example\n"
                  "## Instruction: Is the following statement correct or not? Say true if it's correct; "
                  "otherwise say false.\n\n### Input:\nDrinking water is necessary for human survival.\n"
                  "## Evidence:\n[1] Water\nHumans cannot survive more than a few days without drinking water.\n"
                  "## Rationale: The evidence says humans cannot survive without water, which supports the statement.\n"
                  "## Response: true\n### End of example\n\n"),
}


def task_instruction(question: str, benchmark: str) -> str:
    if benchmark == "pubhealth":
        return f"{PUBHEALTH_INST}\n\n### Input:\n{question}"
    return question


def format_docs(docs) -> str:
    return "\n".join(f"[{i + 1}] {d['title']}\n{d['text']}" for i, d in enumerate(docs))


def drafter_prompt(question: str, docs_text: str, benchmark: str, one_shot: bool = False) -> str:
    return ("Response to the instruction. Also provide rationale for your response.\n"
            + (DEMOS[benchmark] if one_shot else "") +
            f"## Instruction: {task_instruction(question, benchmark)}\n"
            f"## Evidence:\n{docs_text}\n\n{OUTPUT_FORMAT}")


def standard_rag_prompt(question: str, docs_text: str, benchmark: str, one_shot: bool = False) -> str:
    return ("Below is an instruction that describes a task. Write a response for it and "
            "state your explanation supporting your response.\n"
            + (DEMOS[benchmark] if one_shot else "") +
            f"### Instruction: {task_instruction(question, benchmark)}\n"
            f"### Evidence:\n{docs_text}\n\n{OUTPUT_FORMAT}")


# accepts "## Rationale:", "**Rationale:**", "Rationale:" etc.
_H = r"(?:^|\n)[ \t]*(?:#+[ \t]*|\*\*)?"
_RAT = re.compile(_H + r"Rationale[ \t]*:?(?:\*\*)?[ \t]*:?(.*?)(?=" + _H + r"Response\b|\Z)", re.S | re.I)
_RESP = re.compile(_H + r"Response[ \t]*:?(?:\*\*)?[ \t]*:?(.*)", re.S | re.I)


def parse_output(text: str):
    # returns rationale, response, their character spans, ok
    # if the format is not followed, the whole text counts as the response
    m_rat, m_resp = _RAT.search(text), _RESP.search(text)
    if m_resp is None:
        return "", text.strip(), (0, 0), (0, len(text)), False
    rat_span = (m_rat.start(1), m_rat.end(1)) if m_rat else (0, 0)
    resp_span = (m_resp.start(1), m_resp.end(1))
    return (text[rat_span[0]:rat_span[1]].strip(), text[resp_span[0]:resp_span[1]].strip(),
            rat_span, resp_span, m_rat is not None)


def match(prediction: str, golds) -> int:
    # same as match() in Self-RAG
    return int(any(g in prediction for g in golds))


def pubhealth_label(prediction: str):
    m = re.search(r"\b(true|false)\b", prediction.lower())
    return m.group(1) if m else None


def is_correct(prediction: str, golds, benchmark: str) -> dict:
    # TriviaQA: gold answer contained in the response, PubHealth: true/false label
    if benchmark == "pubhealth":
        label = pubhealth_label(prediction)
        c = int(label is not None and label == golds[0])
        return {"correct": c, "correct_norm": c, "pred_label": label}
    return {"correct": match(prediction, golds),
            "correct_norm": match(prediction.lower(), [g.lower() for g in golds]),
            "pred_label": None}
