"""Pilot helper: how long does the GPU need to return to idle after the heaviest
treatment?  Used to choose the cool-down (Section 4.4 assumes 20 s) from data.

Requires the rag_server to be running (e.g. started by a pilot run, or manually:
  CUDA_VISIBLE_DEVICES=0 python src/rag_server.py --config config.json &).

    python src/cooldown_probe.py --benchmark pubhealth --treatment T9 --trials 3

Procedure per trial: record 30 s of idle power/temperature as the reference,
run one query, then sample at 1 Hz for `--observe` seconds and report the first
time after which power stays within `--tol` of the idle mean and temperature is
within 1 C of the idle temperature.
"""
import argparse
import json
import statistics
import time

import requests

from energy import GpuCounter


def sample(gpu, seconds):
    out = []
    for _ in range(int(seconds)):
        out.append((gpu.power_w(), gpu.temperature_c()))
        time.sleep(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--benchmark", default="pubhealth")
    ap.add_argument("--treatment", default="T9")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--observe", type=int, default=120)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    url = f"http://{cfg['server_host']}:{cfg['server_port']}"
    gpu = GpuCounter(cfg["gpu_index"])
    qids = [json.loads(l)["qid"] for l in open(f"{cfg['data']['prepared_dir']}/{a.benchmark}.jsonl")]
    results = []
    for trial in range(a.trials):
        idle = sample(gpu, 30)
        p_idle = statistics.mean(p for p, _ in idle)
        t_idle = max(t for _, t in idle)
        requests.post(url + "/query", json={"benchmark": a.benchmark, "qid": qids[trial % len(qids)],
                                            "treatment": a.treatment}, timeout=600)
        after = sample(gpu, a.observe)
        settle = None
        for i in range(len(after)):
            if all(abs(p - p_idle) <= a.tol * p_idle and t <= t_idle + 1 for p, t in after[i:]):
                settle = i
                break
        results.append({"trial": trial, "idle_power_w": round(p_idle, 1), "idle_temp_c": t_idle,
                        "settle_s": settle})
        print(results[-1], flush=True)
    ok = [r["settle_s"] for r in results if r["settle_s"] is not None]
    print(json.dumps({"max_settle_s": max(ok) if ok else None, "trials": results}, indent=2))


if __name__ == "__main__":
    main()
