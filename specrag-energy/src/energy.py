"""Energy measurement for the Speculative-RAG energy experiment.

Two energy sources are read as *cumulative hardware counters* at t0 and t1;
E = counter(t1) - counter(t0). A counter difference integrates power over the
whole interval, so it does not depend on the sampling rate of a power reading.

* GPU  : NVML nvmlDeviceGetTotalEnergyConsumption (mJ since driver load,
         Volta and newer).  NVIDIA NVML API Reference, "Device Queries".
* CPU  : Linux powercap / RAPL, /sys/class/powercap/intel-rapl:*/energy_uj
         (package domains and their dram sub-domains, uJ, wraps at
         max_energy_range_uj).  Linux kernel docs, "Power Capping Framework".

A background PowerSampler additionally records instantaneous GPU power every
`period_s` for plotting P(t) and for sanity checks; it is NOT used to compute E.
"""
from __future__ import annotations

import csv
import glob
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import pynvml  # provided by the `nvidia-ml-py` package
except ImportError:  # pragma: no cover - allows CPU-only smoke tests
    pynvml = None


# --------------------------------------------------------------------------- GPU
class GpuCounter:
    """Reads the NVML total-energy counter of one GPU (by NVML index)."""

    def __init__(self, gpu_index: int):
        self.available = False
        self.handle = None
        self.name = "n/a"
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            name = pynvml.nvmlDeviceGetName(self.handle)
            self.name = name.decode() if isinstance(name, bytes) else name
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)  # probe support
            self.available = True
        except Exception as exc:  # NVML missing, no GPU, or unsupported arch
            print(f"[energy] GPU energy counter unavailable: {exc}")

    def read_mj(self) -> Optional[int]:
        if not self.available:
            return None
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def power_w(self) -> Optional[float]:
        if not self.available:
            return None
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W

    def temperature_c(self) -> Optional[int]:
        if not self.available:
            return None
        return int(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))


# --------------------------------------------------------------------------- CPU
@dataclass
class _RaplZone:
    name: str
    path: str
    max_range_uj: int


class RaplCounter:
    """Reads all RAPL package zones and their dram sub-zones.

    Core/uncore sub-zones are skipped because they are already contained in the
    package zone; dram is outside the package and is added separately.
    """

    def __init__(self, root: str = "/sys/class/powercap"):
        self.zones: List[_RaplZone] = []
        for zone_dir in sorted(glob.glob(os.path.join(root, "intel-rapl:*"))):
            base = os.path.basename(zone_dir)
            depth = base.count(":")
            try:
                with open(os.path.join(zone_dir, "name")) as fh:
                    name = fh.read().strip()
            except OSError:
                continue
            is_package = depth == 1 and name.startswith("package")
            is_dram = depth == 2 and name == "dram"
            if not (is_package or is_dram):
                continue
            energy_path = os.path.join(zone_dir, "energy_uj")
            try:
                with open(energy_path) as fh:
                    fh.read()
                with open(os.path.join(zone_dir, "max_energy_range_uj")) as fh:
                    max_range = int(fh.read().strip())
            except OSError as exc:  # typically EACCES: energy_uj is root-only since Linux 5.10
                print(f"[energy] RAPL zone {base} not readable: {exc}")
                continue
            self.zones.append(_RaplZone(f"{base}/{name}", energy_path, max_range))
        self.available = len(self.zones) > 0
        if not self.available:
            print("[energy] No readable RAPL zones: CPU energy will be reported as NA.")

    def read_uj(self) -> Optional[Dict[str, int]]:
        if not self.available:
            return None
        values = {}
        for z in self.zones:
            with open(z.path) as fh:
                values[z.name] = int(fh.read().strip())
        return values

    def delta_j(self, start: Dict[str, int], end: Dict[str, int]) -> float:
        total_uj = 0
        for z in self.zones:
            d = end[z.name] - start[z.name]
            if d < 0:  # counter wrapped around once during the interval
                d += z.max_range_uj
            total_uj += d
        return total_uj / 1e6


# ---------------------------------------------------------------------- sampler
class PowerSampler(threading.Thread):
    """Samples instantaneous GPU power and temperature at a fixed period."""

    def __init__(self, gpu: GpuCounter, period_s: float = 0.1):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.period_s = period_s
        self.samples: List[tuple] = []
        self._halt = threading.Event()

    def run(self) -> None:
        if not self.gpu.available:
            return
        t_start = time.perf_counter()
        next_t = t_start
        while not self._halt.is_set():
            now = time.perf_counter()
            self.samples.append((round((now - t_start) * 1000, 1), self.gpu.power_w(), self.gpu.temperature_c()))
            next_t += self.period_s
            self._halt.wait(max(0.0, next_t - time.perf_counter()))

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2)

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_ms", "gpu_power_w", "gpu_temp_c"])
            w.writerows(self.samples)


# ----------------------------------------------------------------- measurement
@dataclass
class Measurement:
    t0_ns: int = 0
    t1_ns: int = 0
    gpu0_mj: Optional[int] = None
    gpu1_mj: Optional[int] = None
    cpu0: Optional[Dict[str, int]] = None
    cpu1: Optional[Dict[str, int]] = None
    extra: dict = field(default_factory=dict)


class EnergyMeter:
    """start() -> [workload] -> stop() -> result().  Counters are read as close as
    possible to the time stamps: counters, then t0 at the start; t1, then
    counters at the end, so the counter interval encloses [t0, t1]."""

    def __init__(self, gpu_index: int, sample_period_s: float = 0.1):
        self.gpu = GpuCounter(gpu_index)
        self.cpu = RaplCounter()
        self.sample_period_s = sample_period_s
        self.m = Measurement()
        self.sampler: Optional[PowerSampler] = None

    def start(self) -> None:
        self.m = Measurement()
        self.sampler = PowerSampler(self.gpu, self.sample_period_s)
        self.sampler.start()
        self.m.gpu0_mj = self.gpu.read_mj()
        self.m.cpu0 = self.cpu.read_uj()
        self.m.t0_ns = time.perf_counter_ns()

    def stop(self) -> None:
        self.m.t1_ns = time.perf_counter_ns()
        self.m.gpu1_mj = self.gpu.read_mj()
        self.m.cpu1 = self.cpu.read_uj()
        if self.sampler is not None:
            self.sampler.stop()

    def result(self) -> dict:
        t_s = (self.m.t1_ns - self.m.t0_ns) / 1e9
        e_gpu = None if self.m.gpu0_mj is None else (self.m.gpu1_mj - self.m.gpu0_mj) / 1000.0
        e_cpu = None if self.m.cpu0 is None else self.cpu.delta_j(self.m.cpu0, self.m.cpu1)
        parts = [e for e in (e_gpu, e_cpu) if e is not None]
        e_total = sum(parts) if parts else None
        return {
            "T_q_s": round(t_s, 4),
            "E_gpu_J": None if e_gpu is None else round(e_gpu, 4),
            "E_cpu_J": None if e_cpu is None else round(e_cpu, 4),
            "E_q_J": None if e_total is None else round(e_total, 4),
            "P_avg_W": None if (e_total is None or t_s <= 0) else round(e_total / t_s, 3),
            "energy_sources": "+".join(
                s for s, v in (("gpu", e_gpu), ("cpu_rapl", e_cpu)) if v is not None) or "none",
        }
