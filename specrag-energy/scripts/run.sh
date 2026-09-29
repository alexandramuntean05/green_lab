#!/usr/bin/env bash
# Runs one phase of the experiment inside tmux on the GPU node so that it
# survives SSH disconnects:
#   bash scripts/run.sh smoke       # tiny random models on CPU, no GPU needed (≈2 min)
#   bash scripts/run.sh pilot       # 3 questions x 10 treatments x 2 reps per benchmark
#   bash scripts/run.sh pubhealth   # full block: 30 x 10 x 10 = 3000 runs
#   bash scripts/run.sh triviaqa
#   bash scripts/run.sh analyse     # R analysis of both benchmarks
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
ER=../experiment-runner
[ -d .venv ] && source .venv/bin/activate
phase=${1:?usage: run.sh smoke|pilot|pubhealth|triviaqa|analyse}

case "$phase" in
  smoke)
    python tests/make_fixtures.py
    python src/prepare_data.py --config tests/fixtures/config_smoke.json
    (cd "$ER" && ER_CONFIG="$ROOT/tests/fixtures/config_smoke.json" ER_DEVICE=cpu ER_PILOT=1 \
       ER_BENCHMARK=pubhealth python experiment-runner/ "$ROOT/RunnerConfig.py")
    python tests/make_synthetic_run_table.py tests/fixtures/synthetic_run_table.csv
    Rscript analysis/analysis.R tests/fixtures/synthetic_run_table.csv tests/fixtures/results_synthetic
    ;;
  pilot)
    for b in pubhealth triviaqa; do
      (cd "$ER" && ER_PILOT=1 ER_BENCHMARK=$b python experiment-runner/ "$ROOT/RunnerConfig.py")
    done
    ;;
  pubhealth|triviaqa)
    (cd "$ER" && ER_BENCHMARK=$phase python experiment-runner/ "$ROOT/RunnerConfig.py")
    python src/requeue_failed.py "experiments/specrag_${phase}/run_table.csv"
    # re-invoke: Experiment Runner restarts and executes only the re-queued runs
    (cd "$ER" && ER_BENCHMARK=$phase python experiment-runner/ "$ROOT/RunnerConfig.py") || true
    ;;
  analyse)
    for b in pubhealth triviaqa; do
      Rscript analysis/analysis.R "experiments/specrag_${b}/run_table.csv" "results/${b}"
    done
    ;;
  *) echo "unknown phase $phase"; exit 1 ;;
esac
