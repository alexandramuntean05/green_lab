"""Energy measurement.

Energy = counter at t1 - counter at t0.
GPU: NVML total energy counter (mJ). CPU: RAPL package counter, read from
powercap if possible, otherwise through EnergiBridge.
The power sampler is only used for monitoring, not for computing energy.
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
    import pynvml  # nvidia-ml-py
except ImportError:  # CPU-only smoke test
    pynvml = None


class GpuCounter:
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
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.available = True
        except Exception as exc:
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


@dataclass
class _RaplZone:
    name: str
    path: str
    max_range_uj: int


class RaplCounter:
    # package zones + dram sub-zones (core/uncore are part of the package)

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
            except OSError as exc:  # usually permission denied
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
            if d < 0:  # counter wrap
                d += z.max_range_uj
            total_uj += d
        return total_uj / 1e6


class CpuUtilization:
    # CPU utilisation from /proc/stat, used for the rough E_cpu_est only

    @staticmethod
    def read():
        try:
            with open("/proc/stat") as fh:
                f = [int(x) for x in fh.readline().split()[1:]]
        except OSError:
            return None
        idle = f[3] + (f[4] if len(f) > 4 else 0)          # idle + iowait
        total = sum(f[:8])
        return total - idle, total

    @staticmethod
    def utilisation(start, end):
        if start is None or end is None or end[1] <= start[1]:
            return None
        return (end[0] - start[0]) / (end[1] - start[1])


class PowerSampler(threading.Thread):
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


class EnergiBridgeCpu:
    # CPU package energy via EnergiBridge (on the cluster RAPL is only readable this way).
    # energibridge runs `sleep` as child; we kill the sleep after t1 so it writes the csv.
    # Energy = PACKAGE_ENERGY interpolated at t1 minus at t0.

    COL = "PACKAGE_ENERGY (J)"

    def __init__(self, binary: str = "energibridge", interval_ms: int = 50, startup_s: float = 1.0):
        import shutil
        self.binary = shutil.which(binary)
        self.interval_ms = interval_ms
        self.startup_s = startup_s          # first sample comes ~0.6 s after start on glg4
        self.last_error = None
        self.available = self.binary is not None
        self.proc = None
        self.csv_path = None
        if not self.available:
            print(f"[energy] '{binary}' not found: CPU energy via EnergiBridge unavailable.")

    def start(self, csv_path: str) -> None:
        import subprocess
        self.csv_path = csv_path
        self.proc = subprocess.Popen([self.binary, "-i", str(self.interval_ms), "-o", csv_path,
                                      "--", "sleep", "3600"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(self.startup_s)

    def stop(self) -> None:
        if self.proc is None:
            return
        time.sleep(max(0.2, 4 * self.interval_ms / 1000))
        try:
            import psutil
            for child in psutil.Process(self.proc.pid).children(recursive=True):
                child.terminate()
        except Exception:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def delta_j(self, t0_epoch: float, t1_epoch: float) -> Optional[float]:
        try:
            with open(self.csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
        except OSError:
            return None
        if len(rows) < 2 or self.COL not in rows[0]:
            self.last_error = f"csv has {len(rows)} rows / missing column"
            return None
        e = [float(r[self.COL]) for r in rows]
        times = [float(r["Time"]) / 1000.0 for r in rows]
        if times[-1] > times[0]:
            t = times
        else:                             # Time not increasing: use Delta
            t, acc = [], times[0]
            for i, r in enumerate(rows):
                if i > 0:
                    acc += float(r["Delta"]) / 1000.0
                t.append(acc)
        if t0_epoch < t[0] or t1_epoch > t[-1]:
            self.last_error = (f"window not covered: t0-first={1000 * (t0_epoch - t[0]):.0f} ms, "
                               f"last-t1={1000 * (t[-1] - t1_epoch):.0f} ms")
            return None

        def at(x):
            for j in range(1, len(t)):
                if t[j] >= x:
                    if t[j] == t[j - 1]:
                        return e[j]
                    return e[j - 1] + (e[j] - e[j - 1]) * (x - t[j - 1]) / (t[j] - t[j - 1])
            return e[-1]

        d = at(t1_epoch) - at(t0_epoch)
        if d < 0:
            self.last_error = "negative difference (counter wrap)"
            return None
        self.last_error = None
        return d


@dataclass
class Measurement:
    t0_ns: int = 0
    t1_ns: int = 0
    t0_epoch: float = 0.0
    t1_epoch: float = 0.0
    gpu0_mj: Optional[int] = None
    gpu1_mj: Optional[int] = None
    cpu0: Optional[Dict[str, int]] = None
    cpu1: Optional[Dict[str, int]] = None
    util0: Optional[tuple] = None
    util1: Optional[tuple] = None
    extra: dict = field(default_factory=dict)


class EnergyMeter:
    # usage: start(), run workload, stop(), result()

    def __init__(self, gpu_index: int, sample_period_s: float = 0.1,
                 cpu_model: Optional[dict] = None, cpu_energy: Optional[dict] = None):
        self.gpu = GpuCounter(gpu_index)
        self.cpu = RaplCounter()
        ce = cpu_energy or {}
        # use EnergiBridge only if powercap is not readable
        self.eb = None
        if not self.cpu.available and ce.get("method") == "energibridge":
            self.eb = EnergiBridgeCpu(ce.get("binary", "energibridge"), int(ce.get("interval_ms", 50)),
                                      float(ce.get("startup_s", 1.0)))
            if not self.eb.available:
                self.eb = None
        self.cpu_model = cpu_model or {}
        self.sample_period_s = sample_period_s
        self.m = Measurement()
        self.sampler: Optional[PowerSampler] = None

    def start(self, eb_csv: Optional[str] = None) -> None:
        self.m = Measurement()
        if self.eb is not None:
            self.eb.start(eb_csv or f"/tmp/energibridge_{os.getpid()}.csv")
        self.sampler = PowerSampler(self.gpu, self.sample_period_s)
        self.sampler.start()
        self.m.gpu0_mj = self.gpu.read_mj()
        self.m.cpu0 = self.cpu.read_uj()
        self.m.util0 = CpuUtilization.read()
        self.m.t0_epoch = time.time()
        self.m.t0_ns = time.perf_counter_ns()

    def stop(self) -> None:
        self.m.t1_ns = time.perf_counter_ns()
        self.m.t1_epoch = time.time()
        self.m.gpu1_mj = self.gpu.read_mj()
        self.m.cpu1 = self.cpu.read_uj()
        self.m.util1 = CpuUtilization.read()
        if self.sampler is not None:
            self.sampler.stop()
        if self.eb is not None:
            self.eb.stop()

    def result(self) -> dict:
        t_s = (self.m.t1_ns - self.m.t0_ns) / 1e9
        e_gpu = None if self.m.gpu0_mj is None else (self.m.gpu1_mj - self.m.gpu0_mj) / 1000.0
        e_cpu, cpu_src = None, None
        if self.m.cpu0 is not None:
            e_cpu, cpu_src = self.cpu.delta_j(self.m.cpu0, self.m.cpu1), "cpu_rapl"
        elif self.eb is not None:
            e_cpu, cpu_src = self.eb.delta_j(self.m.t0_epoch, self.m.t1_epoch), "cpu_energibridge"
        parts = [e for e in (e_gpu, e_cpu) if e is not None]
        e_total = sum(parts) if parts else None
        u = CpuUtilization.utilisation(getattr(self.m, "util0", None), getattr(self.m, "util1", None))
        e_cpu_est = None
        if u is not None and "tdp_w" in self.cpu_model:
            p_max = float(self.cpu_model["tdp_w"])
            p_idle = float(self.cpu_model.get("idle_w", 0.0))
            e_cpu_est = (p_idle + (p_max - p_idle) * u) * t_s
        return {
            "T_q_s": round(t_s, 4),
            "U_cpu": None if u is None else round(u, 4),
            "E_cpu_est_J": None if e_cpu_est is None else round(e_cpu_est, 4),
            "E_gpu_J": None if e_gpu is None else round(e_gpu, 4),
            "E_cpu_J": None if e_cpu is None else round(e_cpu, 4),
            "E_q_J": None if e_total is None else round(e_total, 4),
            "P_avg_W": None if (e_total is None or t_s <= 0) else round(e_total / t_s, 3),
            "cpu_energy_note": None if self.eb is None else self.eb.last_error,
            "energy_sources": "+".join(
                s for s, v in (("gpu", e_gpu), (cpu_src, e_cpu)) if v is not None) or "none",
        }
