"""Standard RAG and Speculative RAG pipelines (the system under test).

Speculative RAG follows Wang et al. [3], Algorithm 1 and Sections 3.2-3.3:
  1. instruction-aware embeddings of the R retrieved documents (InBedder, n_mask=3,
     instruction = the question),
  2. K-Means into k clusters, one document per cluster per subset, repeated until
     m unique subsets exist,
  3. the drafter generates (rationale beta, response alpha) for the m subsets as one
     batch  (rho_Draft = P(beta|Q,delta) + P(alpha|Q,delta,beta)),
  4. the verifier scores all m drafts in ONE batched forward pass over
     [Q, alpha, beta, R, "Yes"]  (rho_SC = P(alpha,beta|Q), rho_SR = P("Yes"|...)),
  5. the draft with the highest rho = rho_Draft * rho_SC * rho_SR is returned.
All probabilities are combined in log space to avoid floating-point underflow;
this is mathematically identical to the product/sum in [3].

Standard RAG: the verifier model generates the answer from all R documents in one
prompt (Wang et al. [3], Appendix I, Figure 8).
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
                          GenerationConfig)

from prompts import (drafter_prompt, format_docs, is_correct, parse_output,
                     standard_rag_prompt, task_instruction)


def _dtype_kw(dtype):
    """`dtype=` replaced `torch_dtype=` in newer Transformers releases; older
    releases silently ignore `dtype=` and load float32 weights (2x memory,
    which does not fit two models in 8 GB). Pick the keyword this version uses."""
    import inspect
    from transformers import PreTrainedModel
    params = inspect.signature(PreTrainedModel.from_pretrained).parameters
    src = inspect.getsource(PreTrainedModel.from_pretrained)
    return {"dtype": dtype} if ('kwargs.pop("dtype"' in src or "dtype" in params) else {"torch_dtype": dtype}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class InBedder:
    """Instruction-aware embedder, usage as documented on the model card of
    KomeijiForce/inbedder-roberta-large (Peng et al., 2024, arXiv:2402.09642)."""

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
        # Both LMs stay resident in GPU memory for the whole experiment (Section 4.2).
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

    # ---------------------------------------------------------------- helpers
    def _gen_config(self, model):
        """Pure greedy decoding. Model-shipped generation_config.json files may set
        sampling or a repetition penalty (e.g. Qwen2.5-Instruct); these are
        overridden so that decoding is identical for all models and treatments."""
        eos = model.generation_config.eos_token_id
        return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=self.max_new,
                                output_logits=True, return_dict_in_generate=True,
                                temperature=None, top_p=None, top_k=None,
                                repetition_penalty=1.0, eos_token_id=eos,
                                pad_token_id=model.generation_config.pad_token_id
                                if model.generation_config.pad_token_id is not None else
                                (eos[0] if isinstance(eos, list) else eos))

    @torch.inference_mode()
    def _generate(self, model, tok, prompts: List[str]) -> List[dict]:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        gcfg = self._gen_config(model)
        out = model.generate(**enc, generation_config=gcfg)
        eos_ids = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
        # raw (unprocessed) logits -> log-probabilities of the chosen tokens
        logits = torch.stack(out.logits, dim=1).float()            # [B, T, V]
        gen = out.sequences[:, enc.input_ids.shape[1]:]             # [B, T]
        lp = torch.log_softmax(logits, -1).gather(-1, gen.unsqueeze(-1)).squeeze(-1).cpu()
        results = []
        for b in range(len(prompts)):
            ids = gen[b].tolist()
            n_steps = len(ids)
            for i, t in enumerate(ids):
                if t in eos_ids:
                    n_steps = i + 1          # decoding steps incl. the EOS step
                    ids = ids[:i]
                    break
            text = tok.decode(ids, skip_special_tokens=True)
            # character end offset of every generated token, for span attribution
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
            if end > s and start < e:   # token [start, end) overlaps span [s, e)
                idx.append(i)
            start = end
        return idx

    def _draft_fields(self, g: dict) -> dict:
        rationale, response, rs, ps, ok = parse_output(g["text"])
        a_idx = self._span_tokens(g["ends"], ps)
        b_idx = self._span_tokens(g["ends"], rs) if ok else []
        lp = g["logprobs"]
        log_pa = float(sum(lp[i] for i in a_idx))
        log_pb = float(sum(lp[i] for i in b_idx)) if b_idx else -math.inf
        return {"alpha": response, "beta": rationale, "parse_ok": ok,
                "n_alpha": len(a_idx), "log_p_alpha": log_pa, "log_p_beta": log_pb}

    def _retrieve(self, item: dict) -> List[dict]:
        # Cached retrieval: the top-R documents were retrieved offline (Contriever-MS
        # MARCO) and are identical for every treatment (Section 4.1).
        return item["ctxs"][: self.R]

    # ------------------------------------------------------------ Standard RAG
    def run_standard(self, item: dict) -> dict:
        tm = {}
        t = time.perf_counter()
        docs = self._retrieve(item)
        tm["t_retrieve_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        prompt = standard_rag_prompt(item["question"], format_docs(docs), item["benchmark"])
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

    # --------------------------------------------------------- Speculative RAG
    def _subsets(self, labels: np.ndarray, qid: str, m: int, k: int):
        """Algorithm 1, lines 3-9 of [3]: sample one document per cluster until m
        unique subsets exist. If fewer than m unique subsets are possible
        (prod of cluster sizes < m), the unique subsets are reused cyclically so
        that exactly m drafts are generated; n_unique is recorded."""
        clusters = [sorted(np.where(labels == c)[0].tolist()) for c in range(k)]
        possible = math.prod(len(c) for c in clusters)
        rng = random.Random(f"{self.seed}-{qid}-k{k}")   # same stream for every m -> nested subsets
        uniq, seen = [], set()
        target = min(m, possible)
        while len(uniq) < target:
            s = tuple(sorted(rng.choice(c) for c in clusters))  # retrieval-rank order
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

        # 1-2: instruction-aware embedding + K-Means (clustering is part of the measured pipeline)
        t = time.perf_counter()
        emb, n_emb = self.embedder.encode([f"{d['title']}\n{d['text']}" for d in docs],
                                          item["question"], self.sc["embed_n_mask"])
        labels = KMeans(n_clusters=k, n_init=self.sc["kmeans_n_init"],
                        random_state=self.seed).fit_predict(emb)
        subsets, n_unique = self._subsets(labels, item["qid"], m, k)
        _sync(self.device)
        tm["t_cluster_ms"] = (time.perf_counter() - t) * 1000

        # 3: parallel drafting = one batch of m prompts to the drafter
        t = time.perf_counter()
        prompts = [drafter_prompt(item["question"], format_docs([docs[i] for i in s]),
                                  item["benchmark"]) for s in subsets]
        gens = self._generate(self.drafter, self.d_tok, prompts)
        _sync(self.device)
        tm["t_draft_ms"] = (time.perf_counter() - t) * 1000
        drafts = [self._draft_fields(g) for g in gens]

        # 4: verification = one batched forward pass of the verifier over all drafts
        t = time.perf_counter()
        v = self._verify(item, drafts)
        _sync(self.device)
        tm["t_verify_ms"] = (time.perf_counter() - t) * 1000

        # 5: selection
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
        logits = self.verifier(input_ids=input_ids, attention_mask=attn).logits.float()
        # log P(t_i | t_<i): logits at position i-1 predict token i
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()

        def span_sum(b, s):
            a, e = s
            return float(lp[b, max(a - 1, 0):e - 1].sum()) if e > a and a > 0 else 0.0

        scores = []
        for b, sp in enumerate(spans):
            log_sc = span_sum(b, sp["a"]) + span_sum(b, sp["b"])
            log_sr = span_sum(b, sp["y"])
            scores.append((log_sc, log_sr))
        return {"scores": scores, "n_in": int(attn.sum())}

    # ------------------------------------------------------------------ entry
    def run(self, item: dict, treatment: dict) -> dict:
        if treatment["architecture"] == "standard":
            return self.run_standard(item)
        return self.run_speculative(item, int(treatment["m"]), int(treatment["k"]))


TREATMENTS: Dict[str, dict] = {"T0": {"architecture": "standard"}}
for _i, (_m, _k) in enumerate(itertools.product([2, 5, 10], [1, 2, 4]), start=1):
    TREATMENTS[f"T{_i}"] = {"architecture": "speculative", "m": _m, "k": _k}
