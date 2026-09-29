# Energy Efficiency of Speculative RAG — experiment package

Code for Section 5 (Experiment Execution) of *Energy Efficiency of Speculative
Retrieval-Augmented Generation: A Controlled Experiment* (The Green Team, Green Lab 2026/2027).

It runs Standard RAG (T0) and Speculative RAG with m ∈ {2,5,10}, k ∈ {1,2,4} (T1–T9)
on PubHealth and TriviaQA, and measures energy per query with hardware counters.

## Layout

| Path | Role |
|---|---|
| `RunnerConfig.py` | Experiment Runner config: run table, randomisation, cool-down, measurement hooks |
| `src/rag_server.py` | Long-lived HTTP server on localhost holding all models (system under test) |
| `src/rag_pipeline.py` | Standard RAG and Speculative RAG (Wang et al., ICLR 2025, Alg. 1) |
| `src/prompts.py` | Prompts (Wang et al. App. H/I), output parsing, correctness metrics |
| `src/energy.py` | NVML GPU energy counter, RAPL CPU energy counter, 100 ms power sampler |
| `src/prepare_data.py` | Stratified random sampling of 30 questions per benchmark (+5 warm-up) |
| `src/requeue_failed.py` | Marks failed runs TODO so Experiment Runner re-executes them |
| `src/cooldown_probe.py` | Pilot: seconds until GPU power/temperature return to idle after T9 |
| `src/pilot_check.py` | Pilot: T_q per treatment, CV of E_q, format compliance, counter vs sampled energy, projected duration |
| `paper/section5.tex`, `paper/section5.bib` | Section 5 text (LaTeX, acmart) and its new references |
| `analysis/analysis.R` | Section 4.5 analysis (RQ1–RQ3, Holm, effect sizes, plots) |
| `scripts/setup.sh`, `scripts/run.sh` | Setup and execution phases |
| `tests/` | Tiny random models + synthetic data for an offline smoke test |

## Why a separate server process

Experiment Runner executes every run in a fresh child process
(`ExperimentController.do_experiment` → `multiprocessing.Process`). Models loaded in the
runner would be reloaded per run, and a CUDA context cannot be inherited across `fork()`.
The models therefore live in `rag_server.py`, are loaded once, stay resident in GPU memory
(Section 4.2), and are called over HTTP on `127.0.0.1`. The server is single-threaded, so
exactly one query is processed at a time.

## What is measured per run

`start_measurement` reads the counters and then t0; `stop_measurement` records t1 and
then reads the counters again.

- `E_gpu_J`: NVML `nvmlDeviceGetTotalEnergyConsumption` difference (mJ counter, Volta+).
- `E_cpu_J`: RAPL `energy_uj` difference, package + dram zones, wrap-around handled.
- `E_q_J = E_gpu_J + E_cpu_J`, `T_q_s = t1 − t0`, `P_avg_W = E_q/T_q`, `E_tok_J = E_q/N_final`.
- `N_in` (all LLM input tokens), `N_gen`, `N_final` (tokens of the delivered response α),
  `N_int = N_gen − N_final`, `N_emb` (embedder tokens), stage timings, `correct`.

Raw data per run: `response.json` (answer, rationale, draft scores, subsets) and
`power_trace.csv` (100 ms GPU power/temperature; for plots only, not used to compute E).

**RAPL permission:** since Linux 5.10, `energy_uj` is root-readable only. If the cluster
does not grant read access, `E_cpu_J` is NA and `energy_sources` = `gpu`. Ask the course
staff (e.g. a udev rule / `chmod o+r` on `energy_uj`) *before* the pilot.

## Steps on the cluster

```bash
git clone <this repo> specrag-energy && cd specrag-energy
bash scripts/setup.sh                       # venv, Experiment Runner @56620fa, models, data
python src/prepare_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct
bash scripts/run.sh smoke                   # optional, CPU only
tmux new -s green
bash scripts/run.sh pilot                   # decide cool-down, check CV, estimate duration
python src/pilot_check.py experiments/specrag_pubhealth_pilot
python src/cooldown_probe.py --treatment T9  # needs a running rag_server
bash scripts/run.sh pubhealth               # ≈ half of the full experiment
bash scripts/run.sh triviaqa
bash scripts/run.sh analyse
```

The analysis needs R with the packages `afex` and `ARTool`
(`install.packages(c("afex", "ARTool"))`). `setup.sh` does not install R; if R is not
available on the node, copy `experiments/` to a laptop and run `analysis/analysis.R` there.

Environment variables: `ER_BENCHMARK`, `ER_PILOT=1`, `ER_REPS`, `ER_COOLDOWN_MS`,
`ER_CONFIG`, `ER_DEVICE`.

## Values the team must confirm (TBD)

- GPU: NVIDIA GeForce RTX 3070, 8 GB (NVML energy counter verified to work).
  Models: Qwen2.5-1.5B-Instruct verifier (~3.1 GB bf16) + Qwen2.5-0.5B-Instruct drafter (~1.0 GB)
  + InBedder (~0.7 GB). Wang et al.'s Mistral-7B / Mixtral-8x7B do not fit in 8 GB; quantisation
  is avoided because it changes the computation and would confound energy.
- Exact TriviaQA / PubHealth file names inside the Self-RAG `eval_data` archive.
- Cool-down (default 20 s) and warm-up count (default 10) → set from the pilot.

## Deviations from Wang et al. [3] (report as threats to validity)

1. The drafter is an off-the-shelf instruction-tuned model; the drafter instruction tuning
   of [3] (40k examples with Gemini-Ultra rationales) is not reproduced.
2. Retrieval is cached (Self-RAG pre-retrieved Contriever-MS MARCO top-10); retrieval energy
   is therefore not measured. It is identical across treatments.
3. InBedder embeddings are computed online inside the measured interval ([3] pre-computes them).
4. If fewer than m unique subsets exist (∏ cluster sizes < m, e.g. k=2 with a 1/9 split),
   the unique subsets are reused cyclically so that exactly m drafts are generated
   (`n_unique_subsets` is logged). Algorithm 1 of [3] would not terminate in this case.
5. Drafts are generated as one batch on one GPU instead of m separate drafter endpoints.
6. Probabilities are combined in log space (numerically stable, mathematically identical).
7. Greedy decoding is enforced explicitly (model-shipped sampling / repetition penalty overridden).
