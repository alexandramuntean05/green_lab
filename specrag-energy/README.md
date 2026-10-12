# Speculative RAG energy experiment

Green Lab 2026/2027, The Green Team.

Compares the energy per query of Standard RAG (T0) and Speculative RAG with
m in {2, 5, 10} and k in {1, 2, 4} (T1-T9) on PubHealth and TriviaQA
(30 questions x 10 treatments x 10 repetitions per benchmark).

Models: Qwen2.5-0.5B-Instruct (drafter), Qwen2.5-1.5B-Instruct (verifier / standard RAG),
InBedder-RoBERTa-large (embeddings). Hardware: glg4, RTX 3070 8 GB, i9-11900KF.

## Files

| Path | |
|---|---|
| `RunnerConfig.py` | Experiment Runner config (run table, cool-down, measurement) |
| `src/rag_server.py` | HTTP server on localhost that keeps the models loaded |
| `src/rag_pipeline.py` | Standard RAG and Speculative RAG |
| `src/prompts.py` | prompts, output parsing, correctness |
| `src/energy.py` | GPU energy (NVML), CPU energy (RAPL / EnergiBridge), power sampler |
| `src/prepare_data.py` | samples the questions |
| `src/requeue_failed.py` | sets failed runs back to TODO |
| `src/pilot_check.py`, `src/cooldown_probe.py`, `src/format_probe.py`, `src/counter_validation.py` | pilot scripts |
| `analysis/analysis.R` | statistical analysis (RQ1-RQ3) |
| `scripts/setup.sh`, `scripts/run.sh` | setup and running |
| `tests/` | tiny models and fake data for a smoke test |
| `data/prepared/` | the sampled questions with their documents |
| `experiments/specrag_*/run_table.csv` | final run tables |
| `results/` | output of `analysis.R` |

## Measurement

- `E_gpu_J`: difference of the NVML total energy counter.
- `E_cpu_J`: CPU package energy. RAPL is not readable directly on the cluster, so it is read
  through EnergiBridge (50 ms interval, counter interpolated at t0 and t1). No DRAM domain.
- `E_q_J = E_gpu_J + E_cpu_J`, `T_q_s = t1 - t0`, `P_avg_W = E_q / T_q`.
- `power_trace.csv` (100 ms GPU power) is only for monitoring.

## Running

```bash
bash scripts/setup.sh
python src/prepare_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct
tmux new -s green
bash scripts/run.sh pilot
bash scripts/run.sh pubhealth
bash scripts/run.sh triviaqa
Rscript analysis/analysis.R experiments/specrag_pubhealth/run_table.csv results/pubhealth
Rscript analysis/analysis.R experiments/specrag_triviaqa/run_table.csv results/triviaqa
```

R needs the packages `afex` and `ARTool`.

The data comes from the Self-RAG eval files. The Google Drive link in `setup.sh` no longer
works; we used the copy on Hugging Face (`RAGLAB/data`, `eval_datasets/PubHealth` and
`eval_datasets/TriviaQA`).

If Triton fails with `Python.h: No such file or directory` (missing python3-dev on the node),
install Python 3.12 headers in your home folder and set
`export C_INCLUDE_PATH=$HOME/pyhdr/include/python3.12` before running.

## Differences from Wang et al. (ICLR 2025)

- The drafter is not fine-tuned.
- Retrieval is pre-computed (Contriever documents from Self-RAG), so it is not measured.
- InBedder embeddings are computed during the measured run.
- If fewer than m different subsets exist, subsets are reused so there are always m drafts.
- Drafts are generated in one batch on one GPU.
- Greedy decoding for all models.
