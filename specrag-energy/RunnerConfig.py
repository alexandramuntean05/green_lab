"""Experiment Runner configuration for "Energy Efficiency of Speculative RAG".

Usage (from the experiment-runner checkout, see README.md):
    ER_BENCHMARK=pubhealth python experiment-runner/ ../specrag-energy/RunnerConfig.py
    ER_BENCHMARK=triviaqa  python experiment-runner/ ../specrag-energy/RunnerConfig.py
    ER_PILOT=1 ER_BENCHMARK=pubhealth python experiment-runner/ ../specrag-energy/RunnerConfig.py

One invocation = one benchmark (one block of the design, analysed separately,
Section 4.5): 10 treatments x Q questions x `repetitions` runs in randomised order.

Per run (hooks called by Experiment Runner in this order, all inside one child
process per run):
  before_run        (parent) thermal gate: wait until the GPU is back near idle temperature
  start_run         record GPU temperature at run start
  start_measurement read energy counters, then t0; start 100 ms power sampler
  interact          one blocking HTTP request -> rag_server processes ONE query
  stop_measurement  t1, then read energy counters
  populate_run_data derived metrics -> run_table.csv; raw response + power trace -> run dir
Between runs Experiment Runner sleeps `time_between_runs_in_ms` (cool-down).
"""
import json
import os
import random
import subprocess
import sys
import time
from os.path import dirname, realpath
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.OperationType import OperationType
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from EventManager.EventSubscriptionController import EventSubscriptionController
from EventManager.Models.RunnerEvents import RunnerEvents
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

ROOT = Path(dirname(realpath(__file__)))
sys.path.insert(0, str(ROOT / "src"))
from energy import EnergyMeter, GpuCounter  # noqa: E402

CFG = json.load(open(os.environ.get("ER_CONFIG", ROOT / "config.json")))
BENCHMARK = os.environ.get("ER_BENCHMARK", "pubhealth")
PILOT = os.environ.get("ER_PILOT", "0") == "1"
TREATMENT_IDS = [f"T{i}" for i in range(10)]   # T0 = Standard RAG, T1..T9 = Speculative (Table 5)

DATA_COLUMNS = [
    "status", "E_q_J", "E_gpu_J", "E_cpu_J", "T_q_s", "P_avg_W", "E_tok_J",
    "N_in", "N_int", "N_final", "N_gen", "N_emb",
    "correct", "correct_norm", "parse_ok_frac", "n_unique_subsets",
    "t_retrieve_ms", "t_cluster_ms", "t_draft_ms", "t_verify_ms", "t_generate_ms", "server_wall_ms",
    "gpu_temp_start_c", "energy_sources",
]


def _read_qids(bench: str):
    path = ROOT / CFG["data"]["prepared_dir"] / f"{bench}.jsonl"
    with open(path) as fh:
        return [json.loads(l)["qid"] for l in fh if l.strip()]


class RunnerConfig:
    ROOT_DIR = ROOT

    # ================================ USER SPECIFIC CONFIG ================================
    name: str = f"specrag_{BENCHMARK}" + ("_pilot" if PILOT else "")
    results_output_path: Path = ROOT / "experiments"
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = int(os.environ.get("ER_COOLDOWN_MS", CFG["experiment"]["cooldown_ms"]))

    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
        ])
        self.run_table_model = None
        self.exp = CFG["experiment"]
        self.url = f"http://{CFG['server_host']}:{CFG['server_port']}"
        self.server: Optional[subprocess.Popen] = None
        self.idle_temp_c: Optional[int] = None
        output.console_log(f"Config loaded: benchmark={BENCHMARK} pilot={PILOT}")

    def create_run_table_model(self) -> RunTableModel:
        qids = _read_qids(BENCHMARK)
        reps = int(os.environ.get("ER_REPS", self.exp["repetitions"]))
        if PILOT:
            qids, reps = qids[:3], 2
        random.seed(f"{CFG['sampling']['seed']}-{BENCHMARK}-order")   # reproducible run order
        self.run_table_model = RunTableModel(
            factors=[FactorModel("treatment", TREATMENT_IDS), FactorModel("qid", qids)],
            repetitions=reps,
            data_columns=DATA_COLUMNS,
            shuffle=True,
        )
        return self.run_table_model

    # ------------------------------------------------------------ experiment
    def before_experiment(self) -> None:
        # Idle reference for the thermal gate is taken BEFORE the server starts and
        # before any warm-up, i.e. while the GPU is genuinely idle. (Reading it right
        # after the warm-up would give a warm reference and make the gate too lenient.)
        gpu = GpuCounter(CFG["gpu_index"])
        if gpu.available:
            self.idle_temp_c = gpu.temperature_c()
            output.console_log(f"GPU '{gpu.name}' idle reference temperature: {self.idle_temp_c} C")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=str(CFG["gpu_index"]))
        device = os.environ.get("ER_DEVICE", "cuda:0")
        (ROOT / "experiments").mkdir(exist_ok=True)
        log = open(ROOT / "experiments" / f"server_{BENCHMARK}.log", "a")
        self.server = subprocess.Popen(
            [sys.executable, str(ROOT / "src" / "rag_server.py"),
             "--config", os.environ.get("ER_CONFIG", str(ROOT / "config.json")), "--device", device],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + 1800
        while time.time() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError("rag_server exited during start-up, see experiments/server_*.log")
            try:
                info = requests.get(self.url + "/health", timeout=5).json()
                output.console_log_OK(f"Server ready: {info}")
                with open(ROOT / "experiments" / f"environment_{BENCHMARK}.json", "w") as fh:
                    json.dump(info, fh, indent=2)
                break
            except requests.exceptions.RequestException:
                time.sleep(5)
        else:
            raise RuntimeError("rag_server did not become ready in 30 min")

        n_warm = self.exp["warmup_queries"]
        output.console_log(f"Running {n_warm} unmeasured warm-up queries ...")
        requests.post(self.url + "/warmup", json={"n": n_warm, "seed": CFG["sampling"]["seed"]},
                      timeout=self.exp["request_timeout_s"] * max(1, n_warm))
        time.sleep(self.time_between_runs_in_ms / 1000)   # cool-down before the first measured run

    def before_run(self) -> None:
        """Thermal gate: start a run only when the GPU has cooled to within
        `thermal_gate_margin_c` of the idle reference (or below a fixed limit)."""
        if not self.exp["thermal_gate_enabled"]:
            return
        gpu = GpuCounter(CFG["gpu_index"])
        if not gpu.available:
            return
        limit = self.exp["thermal_gate_max_temp_c"]
        if limit is None and self.idle_temp_c is not None:
            limit = self.idle_temp_c + self.exp["thermal_gate_margin_c"]
        if limit is None:
            return
        deadline = time.time() + self.exp["thermal_gate_timeout_s"]
        while gpu.temperature_c() > limit and time.time() < deadline:
            time.sleep(1)

    # -------------------------------------------------------------------- run
    def start_run(self, context: RunnerContext) -> None:
        gpu = GpuCounter(CFG["gpu_index"])
        self.gpu_temp_start = gpu.temperature_c()
        self.response, self.error = None, None

    def start_measurement(self, context: RunnerContext) -> None:
        self.meter = EnergyMeter(CFG["gpu_index"], self.exp["power_sample_period_s"])
        self.meter.start()

    def interact(self, context: RunnerContext) -> None:
        run = context.execute_run
        try:
            r = requests.post(self.url + "/query",
                              json={"benchmark": BENCHMARK, "qid": run["qid"], "treatment": run["treatment"]},
                              timeout=self.exp["request_timeout_s"])
            if r.status_code != 200:
                self.error = r.text
            else:
                self.response = r.json()
        except requests.exceptions.RequestException as exc:
            self.error = repr(exc)

    def stop_measurement(self, context: RunnerContext) -> None:
        self.meter.stop()

    def stop_run(self, context: RunnerContext) -> None:
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        m = self.meter.result()
        self.meter.sampler.write_csv(str(context.run_dir / "power_trace.csv"))
        with open(context.run_dir / "response.json", "w") as fh:
            json.dump({"measurement": m, "response": self.response, "error": self.error}, fh,
                      indent=1, ensure_ascii=False)
        row = {c: "" for c in DATA_COLUMNS}
        row.update({"gpu_temp_start_c": self.gpu_temp_start, "energy_sources": m["energy_sources"],
                    "E_q_J": m["E_q_J"], "E_gpu_J": m["E_gpu_J"], "E_cpu_J": m["E_cpu_J"],
                    "T_q_s": m["T_q_s"], "P_avg_W": m["P_avg_W"]})
        if self.response is None:
            row["status"] = "error"          # re-queued with requeue_failed.py (Section 4.5)
            return row
        r = self.response
        row["status"] = "ok"
        for c in ("N_in", "N_int", "N_final", "N_gen", "N_emb", "correct", "correct_norm",
                  "parse_ok_frac", "n_unique_subsets", "t_retrieve_ms", "t_cluster_ms",
                  "t_draft_ms", "t_verify_ms", "t_generate_ms", "server_wall_ms"):
            v = r.get(c, "")
            row[c] = round(v, 3) if isinstance(v, float) else ("" if v is None else v)
        if m["E_q_J"] is not None and r.get("N_final"):
            row["E_tok_J"] = round(m["E_q_J"] / r["N_final"], 5)
        return row

    def after_experiment(self) -> None:
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.server.kill()

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None
