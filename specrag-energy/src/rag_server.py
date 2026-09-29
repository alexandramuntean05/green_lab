"""Long-lived inference server (system under test).

Experiment Runner executes every run in a fresh child process
(ExperimentController.do_experiment -> multiprocessing.Process), so models that
live in the runner process would be re-loaded for every run and CUDA contexts
cannot be shared across fork(). The models therefore live in this separate
process, are loaded once, stay resident in GPU memory, and are called over
HTTP on localhost. The server is single-threaded: exactly one query is processed
at a time (serving batch size of one query, Section 4.2).

Endpoints
  GET  /health                           -> {"status": "ok", ...}
  POST /query  {"benchmark","qid","treatment"} -> result of one RAG query
  POST /warmup {"n": int}                -> n unmeasured warm-up queries
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import TREATMENTS, RagSystem  # noqa: E402


def load_items(prepared_dir: str):
    items = {}
    for name in ("pubhealth", "triviaqa", "pubhealth_warmup", "triviaqa_warmup"):
        path = os.path.join(prepared_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            items[name] = {r["qid"]: r for r in (json.loads(l) for l in fh if l.strip())}
    return items


class Handler(BaseHTTPRequestHandler):
    system: RagSystem = None
    items: dict = None
    info: dict = None

    def log_message(self, fmt, *args):  # keep stdout quiet during measurements
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", **self.info})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/query":
                item = self.items[req["benchmark"]][req["qid"]]
                t0 = time.perf_counter()
                res = self.system.run(item, TREATMENTS[req["treatment"]])
                res["server_wall_ms"] = (time.perf_counter() - t0) * 1000
                res.update({"qid": req["qid"], "benchmark": req["benchmark"], "treatment": req["treatment"]})
                self._send(200, res)
            elif self.path == "/warmup":
                rng = random.Random(req.get("seed", 0))
                pools = [(b, q) for b in ("pubhealth_warmup", "triviaqa_warmup")
                         for q in self.items.get(b, {})]
                n = int(req.get("n", 10))
                for i in range(n):
                    b, q = pools[i % len(pools)]
                    self.system.run(self.items[b][q], TREATMENTS[rng.choice(list(TREATMENTS))])
                self._send(200, {"warmup_queries": n})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": repr(exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    t = time.perf_counter()
    Handler.system = RagSystem(cfg, device=args.device)
    Handler.items = load_items(cfg["data"]["prepared_dir"])
    import torch, transformers  # noqa: E401
    Handler.info = {"models": cfg["models"], "device": args.device,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "load_s": round(time.perf_counter() - t, 1),
                    "n_items": {k: len(v) for k, v in Handler.items.items()}}
    srv = HTTPServer((cfg["server_host"], cfg["server_port"]), Handler)
    print(f"READY {json.dumps(Handler.info)}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
