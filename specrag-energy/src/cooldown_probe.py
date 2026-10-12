"""Measures how long the GPU needs to get back to idle after one heavy query,
used to pick the cool-down time. Needs a running rag_server:

    python src/cooldown_probe.py --benchmark pubhealth --treatment T9 --trials 3

Idle band = idle mean +/- max(tol * mean, 3 * stdev, 1 W), 5 s moving average.
"""
import argparse
import json
import statistics
import sys
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
    try:
        info = requests.get(url + "/health", timeout=5).json()
    except requests.exceptions.RequestException:
        sys.exit("rag_server is not reachable - start it first (see docstring).")
    print("server:", info.get("gpu"), "| models:", info.get("models", {}).get("drafter"), "+",
          info.get("models", {}).get("verifier"))
    gpu = GpuCounter(cfg["gpu_index"])
    qids = [json.loads(l)["qid"] for l in open(f"{cfg['data']['prepared_dir']}/{a.benchmark}_warmup.jsonl")]
    results = []
    for trial in range(a.trials):
        idle = sample(gpu, 30)
        p = [x for x, _ in idle]
        p_idle, p_sd = statistics.mean(p), statistics.stdev(p)
        band = max(a.tol * p_idle, 3 * p_sd, 1.0)
        t_idle = max(t for _, t in idle)
        r = requests.post(url + "/query", json={"benchmark": f"{a.benchmark}_warmup", "qid": qids[trial % len(qids)],
                                                "treatment": a.treatment}, timeout=600)
        if r.status_code != 200:
            sys.exit(f"query failed: {r.text[:300]}")
        after = sample(gpu, a.observe)
        pw = [x for x, _ in after]
        ma = [statistics.mean(pw[max(0, i - 4):i + 1]) for i in range(len(pw))]
        ok = [abs(ma[i] - p_idle) <= band and after[i][1] <= t_idle + 1 for i in range(len(after))]
        settle = next((i for i in range(len(ok)) if all(ok[i:])), None)
        results.append({"trial": trial, "idle_power_w": round(p_idle, 1), "idle_sd_w": round(p_sd, 2),
                        "band_w": round(band, 2), "idle_temp_c": t_idle, "peak_power_w": round(max(pw), 1),
                        "peak_temp_c": max(t for _, t in after), "settle_s": settle})
        print(results[-1], flush=True)
        print("   power (W) every 5 s after the query:", [round(x) for x in pw[::5]], flush=True)
    ok_s = [r["settle_s"] for r in results if r["settle_s"] is not None]
    print(json.dumps({"max_settle_s": max(ok_s) if ok_s else None,
                      "unsettled_trials": sum(r["settle_s"] is None for r in results)}, indent=2))


if __name__ == "__main__":
    main()
