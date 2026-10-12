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

# Self-RAG eval data (if the Google Drive link is dead, use huggingface RAGLAB/data)
mkdir -p data/raw
if [ ! -f data/raw/eval_data.zip ]; then
  gdown 1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb -O data/raw/eval_data.zip
fi
unzip -o -q data/raw/eval_data.zip -d data/raw
ls -R data/raw | head -40
echo
echo ">> Check that config.json -> data.pubhealth_raw / data.triviaqa_raw point to the unzipped files,"
echo ">> then run:  python src/prepare_data.py --tokenizer \$(python -c \"import json;print(json.load(open('config.json'))['models']['verifier'])\")"

# save software versions
mkdir -p experiments
{ date -Is; uname -a; nvidia-smi; python --version; pip freeze; } > experiments/environment_setup.txt 2>&1
echo "environment recorded in experiments/environment_setup.txt"
