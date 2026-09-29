"""Mark failed or incomplete runs of an Experiment Runner run table as TODO so that
re-invoking Experiment Runner on the same config re-executes only those runs
(Experiment Runner "Restarting" feature; Section 4.5: failed runs are repeated).

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
