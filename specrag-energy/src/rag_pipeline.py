"""Standard RAG and Speculative RAG pipelines.

Speculative RAG (Wang et al., ICLR 2025):
  1. embed the R documents with InBedder, cluster them into k groups (k-means)
  2. build m subsets with one document per cluster
  3. drafter writes a rationale + answer for every subset (one batch)
  4. verifier scores all drafts in one forward pass
  5. return the draft with the best score
Scores are added as log probabilities.

Standard RAG: the verifier model answers from all R documents in one prompt.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn.functional import gelu
from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer,
                          GenerationConfig, LogitsProcessor, LogitsProcessorList)

from prompts import (drafter_prompt, format_docs, is_correct, parse_output,
                     standard_rag_prompt, task_instruction)


def _dtype_kw(dtype):
    # newer transformers use dtype=, older ones torch_dtype=
    import inspect
    from transformers import PreTrainedModel
    params = inspect.signature(PreTrainedModel.from_pretrained).parameters
    src = inspect.getsource(PreTrainedModel.from_pretrained)
    return {"dtype": dtype} if ('kwargs.pop("dtype"' in src or "dtype" in params) else {"torch_dtype": dtype}


class ChosenTokenLogProb(LogitsProcessor):
    # log prob of the greedy token at each step. Keeping all logits
    # (output_logits=True) ran out of memory with m=10.

    def __init__(self):
        self.steps = []

    def __call__(self, input_ids, scores):
        s = scores.float()
        self.steps.append(s.max(-1).values - torch.logsumexp(s, -1))
        return scores


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class InBedder:
    # usage as on the model card of KomeijiForce/inbedder-roberta-large

    def __init__(self, path: str, device, dtype):
        mlm = AutoModelForMaskedLM.from_pretrained(path, **_dtype_kw(dtype))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = mlm.roberta.to(device).eval()
        self.dense = mlm.lm_head.dense.to(device)
        self.layer_norm = mlm.lm_head.layer_norm.to(device)
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: List[str], instruction: str, n_mask: int = 3):
        prompts = [instruction + self.tok.mask_token * n_mask for _ in texts]
        inputs = self.tok(texts, prompts, padding=True, truncation="only_first",
                          max_length=512, return_tensors="pt").to(self.device)
        mask = inputs.input_ids.eq(self.tok.mask_token_id)
        h = self.model(**inputs).last_hidden_state[mask]
        h = self.layer_norm(gelu(self.dense(h))).float()
        h = h.reshape(len(texts), n_mask, -1).mean(1)
        h = (h - h.mean(1, keepdim=True)) / h.std(1, keepdim=True)
        return h.cpu().numpy(), int(inputs.attention_mask.sum())


class RagSystem:
    def __init__(self, cfg: dict, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        mc = cfg["models"]
        dtype = getattr(torch, mc["dtype"]) if self.device.type == "cuda" else torch.float32
        kw = _dtype_kw(dtype)
        if mc.get("attn_implementation"):
            kw["attn_implementation"] = mc["attn_implementation"]
        # both models stay on the GPU
        self.drafter = AutoModelForCausalLM.from_pretrained(mc["drafter"], **kw).to(self.device).eval()
        self.verifier = AutoModelForCausalLM.from_pretrained(mc["verifier"], **kw).to(self.device).eval()
        self.d_tok = AutoTokenizer.from_pretrained(mc["drafter"])
        self.v_tok = AutoTokenizer.from_pretrained(mc["verifier"])
        for t in (self.d_tok, self.v_tok):
            if t.pad_token is None:
                t.pad_token = t.eos_token
        self.embedder = InBedder(mc["embedder"], self.device, dtype)
        self.max_new = cfg["generation"]["max_new_tokens"]
        self.R = cfg["retrieval"]["R"]
        self.seed = cfg["sampling"]["seed"]
        self.sc = cfg["speculative"]
        # prefill + one-shot example, added after the pilot (format problems)
        self.prefill = cfg["generation"].get("prefill", "")
        self.one_shot = bool(cfg["generation"].get("one_shot", False))

    def _gen_config(self, model):
        # greedy, overrides sampling/repetition_penalty from the model's own config
        eos = model.generation_config.eos_token_id
        return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=self.max_new,
                                return_dict_in_generate=True,
                                temperature=None, top_p=None, top_k=None,
                                repetition_penalty=1.0, eos_token_id=eos,
                                pad_token_id=model.generation_config.pad_token_id
                                if model.generation_config.pad_token_id is not None else
                                (eos[0] if isinstance(eos, list) else eos))

    @torch.inference_mode()
    def _generate(self, model, tok, prompts: List[str]) -> List[dict]:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) + self.prefill for p in prompts]
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        gcfg = self._gen_config(model)
        rec = ChosenTokenLogProb()
        out = model.generate(**enc, generation_config=gcfg, logits_processor=LogitsProcessorList([rec]))
        eos_ids = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
        gen = out.sequences[:, enc.input_ids.shape[1]:]
        lp = torch.stack(rec.steps, dim=1).cpu()
        results = []
        for b in range(len(prompts)):
            ids = gen[b].tolist()
            n_steps = len(ids)
            for i, t in enumerate(ids):
                if t in eos_ids:
                    n_steps = i + 1          # incl. eos
                    ids = ids[:i]
                    break
            text = tok.decode(ids, skip_special_tokens=True)
            # end position of each token in the text
            ends = [len(tok.decode(ids[:i + 1], skip_special_tokens=True)) for i in range(len(ids))]
            results.append({"text": text, "ids": ids, "ends": ends,
                            "logprobs": lp[b, :len(ids)].tolist(),
                            "n_in": int(enc.attention_mask[b].sum()), "n_gen": n_steps})
        return results

    @staticmethod
    def _span_tokens(ends: List[int], span) -> List[int]:
        s, e = span
        idx, start = [], 0
        for i, end in enumerate(ends):
            if end > s and start < e:
                idx.append(i)
            start = end
        return idx

    def _draft_fields(self, g: dict) -> dict:
        # spans are shifted by the prefill length
        off = len(self.prefill)
        rationale, response, rs, ps, ok = parse_output(self.prefill + g["text"])
        rs = (max(rs[0] - off, 0), max(rs[1] - off, 0))
        ps = (max(ps[0] - off, 0), max(ps[1] - off, 0))
        a_idx = self._span_tokens(g["ends"], ps)
        b_idx = self._span_tokens(g["ends"], rs) if ok else []
        lp = g["logprobs"]
        log_pa = float(sum(lp[i] for i in a_idx))
        log_pb = float(sum(lp[i] for i in b_idx)) if b_idx else -math.inf
        return {"alpha": response, "beta": rationale, "parse_ok": ok,
                "n_alpha": len(a_idx), "log_p_alpha": log_pa, "log_p_beta": log_pb}

    def _retrieve(self, item: dict) -> List[dict]:
        # documents are pre-retrieved (Contriever), same for all treatments
        return item["ctxs"][: self.R]

    def run_standard(self, item: dict) -> dict:
        tm = {}
        t = time.perf_counter()
        docs = self._retrieve(item)
        tm["t_retrieve_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        prompt = standard_rag_prompt(item["question"], format_docs(docs), item["benchmark"], self.one_shot)
        g = self._generate(self.verifier, self.v_tok, [prompt])[0]
        _sync(self.device)
        tm["t_generate_ms"] = (time.perf_counter() - t) * 1000

        f = self._draft_fields(g)
        res = {"architecture": "standard", "m": None, "k": None,
               "pred": f["alpha"], "rationale": f["beta"], "raw_output": g["text"],
               "parse_ok_frac": float(f["parse_ok"]),
               "N_in": g["n_in"], "N_emb": 0, "N_gen": g["n_gen"],
               "N_final": f["n_alpha"], "N_int": g["n_gen"] - f["n_alpha"]}
        res.update(tm)
        res.update(is_correct(f["alpha"], item["golds"], item["benchmark"]))
        return res

    def _subsets(self, labels: np.ndarray, qid: str, m: int, k: int):
        # one document per cluster until m different subsets exist; if fewer are
        # possible, the subsets are reused so there are still m drafts
        clusters = [sorted(np.where(labels == c)[0].tolist()) for c in range(k)]
        possible = math.prod(len(c) for c in clusters)
        rng = random.Random(f"{self.seed}-{qid}-k{k}")
        uniq, seen = [], set()
        target = min(m, possible)
        while len(uniq) < target:
            s = tuple(sorted(rng.choice(c) for c in clusters))
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        subsets = [uniq[i % len(uniq)] for i in range(m)]
        return subsets, len(uniq)

    def run_speculative(self, item: dict, m: int, k: int) -> dict:
        tm = {}
        t = time.perf_counter()
        docs = self._retrieve(item)
        tm["t_retrieve_ms"] = (time.perf_counter() - t) * 1000

        # embedding + clustering
        t = time.perf_counter()
        emb, n_emb = self.embedder.encode([f"{d['title']}\n{d['text']}" for d in docs],
                                          item["question"], self.sc["embed_n_mask"])
        labels = KMeans(n_clusters=k, n_init=self.sc["kmeans_n_init"],
                        random_state=self.seed).fit_predict(emb)
        subsets, n_unique = self._subsets(labels, item["qid"], m, k)
        _sync(self.device)
        tm["t_cluster_ms"] = (time.perf_counter() - t) * 1000

        # drafting
        t = time.perf_counter()
        prompts = [drafter_prompt(item["question"], format_docs([docs[i] for i in s]),
                                  item["benchmark"], self.one_shot) for s in subsets]
        gens = self._generate(self.drafter, self.d_tok, prompts)
        _sync(self.device)
        tm["t_draft_ms"] = (time.perf_counter() - t) * 1000
        drafts = [self._draft_fields(g) for g in gens]

        # verification
        t = time.perf_counter()
        v = self._verify(item, drafts)
        _sync(self.device)
        tm["t_verify_ms"] = (time.perf_counter() - t) * 1000

        scores = []
        for d, (log_sc, log_sr) in zip(drafts, v["scores"]):
            log_draft = float(np.logaddexp(d["log_p_beta"], d["log_p_alpha"]))
            scores.append(log_draft + log_sc + log_sr)
        best = int(np.argmax(scores))
        chosen = drafts[best]
        n_gen = sum(g["n_gen"] for g in gens)
        res = {"architecture": "speculative", "m": m, "k": k,
               "pred": chosen["alpha"], "rationale": chosen["beta"], "raw_output": gens[best]["text"],
               "parse_ok_frac": float(np.mean([d["parse_ok"] for d in drafts])),
               "N_in": sum(g["n_in"] for g in gens) + v["n_in"], "N_emb": n_emb,
               "N_in_drafter": sum(g["n_in"] for g in gens), "N_in_verifier": v["n_in"],
               "N_gen": n_gen, "N_final": chosen["n_alpha"], "N_int": n_gen - chosen["n_alpha"],
               "n_unique_subsets": n_unique, "selected_draft": best,
               "draft_log_scores": [round(s, 4) for s in scores],
               "subsets": [list(s) for s in subsets]}
        res.update(tm)
        res.update(is_correct(chosen["alpha"], item["golds"], item["benchmark"]))
        return res

    @torch.inference_mode()
    def _verify(self, item: dict, drafts: List[dict]) -> dict:
        tok = self.v_tok
        q = task_instruction(item["question"], item["benchmark"])
        R = self.sc["reflection_statement"]
        enc_ = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
        seqs, spans = [], []
        for d in drafts:
            parts = [("q", f"Question: {q}\nAnswer: "), ("a", d["alpha"]),
                     ("x", "\nExplanation: "), ("b", d["beta"]),
                     ("x", f"\n{R}\n"), ("y", "Yes")]
            ids, sp = [], {}
            for tag, s in parts:
                t_ids = enc_(s) if s else []
                if tag in ("a", "b", "y"):
                    sp[tag] = (len(ids), len(ids) + len(t_ids))
                ids += t_ids
            seqs.append(ids)
            spans.append(sp)
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
        attn = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
        logits = self.verifier(input_ids=input_ids, attention_mask=attn).logits
        # per sequence to save memory
        lp = torch.stack([
            torch.log_softmax(logits[b, :-1].float(), -1).gather(-1, input_ids[b, 1:].unsqueeze(-1)).squeeze(-1)
            for b in range(logits.shape[0])]).cpu()
        del logits

        def span_sum(b, s):
            a, e = s
            return float(lp[b, max(a - 1, 0):e - 1].sum()) if e > a and a > 0 else 0.0

        scores = []
        for b, sp in enumerate(spans):
            log_sc = span_sum(b, sp["a"]) + span_sum(b, sp["b"])
            log_sr = span_sum(b, sp["y"])
            scores.append((log_sc, log_sr))
        return {"scores": scores, "n_in": int(attn.sum())}

    def run(self, item: dict, treatment: dict) -> dict:
        if treatment["architecture"] == "standard":
            return self.run_standard(item)
        return self.run_speculative(item, int(treatment["m"]), int(treatment["k"]))


TREATMENTS: Dict[str, dict] = {"T0": {"architecture": "standard"}}
for _i, (_m, _k) in enumerate(itertools.product([2, 5, 10], [1, 2, 4]), start=1):
    TREATMENTS[f"T{_i}"] = {"architecture": "speculative", "m": _m, "k": _k}
