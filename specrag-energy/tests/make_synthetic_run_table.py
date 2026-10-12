"""Random run_table.csv for testing analysis.R (fake numbers)."""
import csv
import random
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/synthetic_run_table.csv"
rng = random.Random(1)
cols = ["__run_id", "__done", "treatment", "qid", "status", "E_q_J", "E_gpu_J", "E_cpu_J", "T_q_s", "P_avg_W",
        "E_tok_J", "N_in", "N_int", "N_final", "N_gen", "N_emb", "correct", "correct_norm"]
MK = {"T0": (None, None)}
for i, (m, k) in enumerate([(m, k) for m in (2, 5, 10) for k in (1, 2, 4)], 1):
    MK[f"T{i}"] = (m, k)
rows = []
for q in range(30):
    difficulty = rng.uniform(0.7, 1.4)
    for t, (m, k) in MK.items():
        base_t = 2.0 if m is None else 0.8 + 0.05 * m + 0.1 * k
        base_p = 250 if m is None else 230 + 4 * m + 3 * k
        corr = int(rng.random() < (0.6 if m is None else 0.65))
        for r in range(10):
            T = base_t * difficulty * rng.gauss(1, 0.03)
            P = base_p * rng.gauss(1, 0.02)
            E = T * P
            n_final = rng.randint(5, 30)
            rows.append({"__run_id": f"run_{t}_{q}_{r}", "__done": "DONE", "treatment": t, "qid": f"q{q}",
                         "status": "ok", "E_q_J": round(E, 3), "E_gpu_J": round(0.85 * E, 3),
                         "E_cpu_J": round(0.15 * E, 3), "T_q_s": round(T, 4), "P_avg_W": round(P, 3),
                         "E_tok_J": round(E / n_final, 4), "N_in": 1800 if m is None else m * (k * 150 + 120),
                         "N_int": 60 if m is None else m * 90, "N_final": n_final, "N_gen": 80,
                         "N_emb": 0 if m is None else 1500, "correct": corr, "correct_norm": corr})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print("wrote", len(rows), "synthetic rows to", out)
