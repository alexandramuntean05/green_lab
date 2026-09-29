"""Pilot checks (Section 5.5) on an Experiment Runner results folder.

    python src/pilot_check.py experiments/specrag_pubhealth_pilot

Reports per treatment: mean T_q (to recompute the total duration), coefficient of
variation of E_q over repetitions, drafter format-compliance, and the agreement
between counter-based GPU energy (NVML total-energy counter) and the trapezoidal
integral of the 100 ms power samples.
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

exp = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(exp, "run_table.csv"))))
ok = [r for r in rows if r.get("status") == "ok" and r["E_q_J"] not in ("", "None")]
print(f"runs: {len(rows)}  ok: {len(ok)}")

by_t = defaultdict(list)
cell = defaultdict(list)
for r in ok:
    by_t[r["treatment"]].append(r)
    cell[(r["treatment"], r["qid"])].append(float(r["E_q_J"]))

print(f"\n{'treat':6}{'T_q mean (s)':>14}{'E_q mean (J)':>14}{'parse_ok':>10}")
total_s = 0.0
for t in sorted(by_t, key=lambda x: int(x[1:])):
    rs = by_t[t]
    tq = statistics.mean(float(r["T_q_s"]) for r in rs)
    eq = statistics.mean(float(r["E_q_J"]) for r in rs)
    po = statistics.mean(float(r["parse_ok_frac"] or 0) for r in rs)
    total_s += tq
    print(f"{t:6}{tq:14.2f}{eq:14.1f}{po:10.2f}")

cvs = [statistics.stdev(v) / statistics.mean(v) for v in cell.values() if len(v) > 1 and statistics.mean(v) > 0]
if cvs:
    print(f"\nCV of E_q per question x treatment: median {statistics.median(cvs):.3f}  max {max(cvs):.3f}")

# counter vs integrated samples
diffs = []
for d in os.listdir(exp):
    trace, resp = os.path.join(exp, d, "power_trace.csv"), os.path.join(exp, d, "response.json")
    if not (os.path.exists(trace) and os.path.exists(resp)):
        continue
    e_counter = json.load(open(resp))["measurement"]["E_gpu_J"]
    pts = [(float(a["t_ms"]) / 1000, float(a["gpu_power_w"])) for a in csv.DictReader(open(trace))
           if a["gpu_power_w"] not in ("", "None")]
    if e_counter is None or len(pts) < 2:
        continue
    e_int = sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(pts, pts[1:]))
    diffs.append((e_int - e_counter) / e_counter)
if diffs:
    print(f"GPU energy, integrated samples vs counter: median rel. diff {statistics.median(diffs):+.3f}, "
          f"range [{min(diffs):+.3f}, {max(diffs):+.3f}]  (n={len(diffs)})")

# projected duration of the full experiment (one benchmark block)
cool = float(os.environ.get("ER_COOLDOWN_MS", 20000)) / 1000
q, reps = 30, 10
mean_tq = total_s / max(1, len(by_t))
per_run = mean_tq + cool + 2.0
print(f"\nprojected block duration: {q}*{len(by_t)}*{reps} runs x {per_run:.1f} s "
      f"= {q * len(by_t) * reps * per_run / 3600:.1f} h (assumes 2 s overhead per run)")
