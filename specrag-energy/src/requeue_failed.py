"""Set failed runs back to TODO so Experiment Runner runs them again.

    python src/requeue_failed.py experiments/specrag_pubhealth/run_table.csv
"""
import csv
import shutil
import sys

path = sys.argv[1]
shutil.copy(path, path + ".bak")
with open(path, newline="") as fh:
    rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
n = 0
for r in rows:
    if r["__done"] == "DONE" and r.get("status") != "ok":
        r["__done"] = "TODO"
        n += 1
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"re-queued {n} runs (backup: {path}.bak)")
