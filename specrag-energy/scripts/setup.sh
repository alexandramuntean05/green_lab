#!/usr/bin/env bash
# setup on the GPU node: bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# python env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# experiment runner, commit 56620fa (v1.0.1 fails in tmux because of os.getlogin())
if [ ! -d ../experiment-runner ]; then
  git clone https://github.com/S2-group/experiment-runner.git ../experiment-runner
  git -C ../experiment-runner checkout 56620fa6e48ef99180824a24369dc2f6ec3d6910
fi

# download models
python - <<'EOF'
import json
from huggingface_hub import snapshot_download
cfg = json.load(open("config.json"))
for role in ("drafter", "verifier", "embedder"):
    print("downloading", cfg["models"][role])
    snapshot_download(cfg["models"][role])
EOF

# Self-RAG eval data (copy on Hugging Face, the original Google Drive link is dead)
python - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ("eval_datasets/PubHealth/health_claims_processed.jsonl",
          "eval_datasets/TriviaQA/triviaqa_test_w_gs.jsonl"):
    print(hf_hub_download("RAGLAB/data", f, repo_type="dataset", local_dir="data/raw"))
EOF
# data/prepared/ already contains the sampled questions used in the experiment;
# to sample again: python src/prepare_data.py

# save software versions
mkdir -p experiments
{ date -Is; uname -a; nvidia-smi; python --version; pip freeze; } > experiments/environment_setup.txt 2>&1
echo "environment recorded in experiments/environment_setup.txt"
