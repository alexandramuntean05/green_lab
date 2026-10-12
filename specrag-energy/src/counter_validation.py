"""Checks the NVML energy counter: how often it updates, and whether it agrees
with the integrated power samples under a constant 30 s load.
(In the pilot the integrated samples were 50-60% lower for short queries.)

    CUDA_VISIBLE_DEVICES=0 python src/counter_validation.py
Stop the rag_server first.
"""
import statistics
import threading
import time

import pynvml
import torch


def update_period(h, seconds=3.0):
    last, t_changes = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        v = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        if v != last:
            t_changes.append(time.perf_counter())
            last = v
    gaps = [b - a for a, b in zip(t_changes, t_changes[1:])]
    return (statistics.median(gaps) * 1000 if gaps else None), len(t_changes)


def steady_load(h, seconds=30.0, period=0.1):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    samples, stop = [], threading.Event()

    def sampler():
        t0 = time.perf_counter()
        while not stop.is_set():
            samples.append((time.perf_counter() - t0, pynvml.nvmlDeviceGetPowerUsage(h) / 1000))
            stop.wait(period)

    for _ in range(50):                       # warm up
        a @ a
    torch.cuda.synchronize()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
    t1, e1 = time.perf_counter(), pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    stop.set()
    th.join()
    e_counter = (e1 - e0) / 1000
    e_int = sum((t2 - t1_) * (p1 + p2) / 2 for (t1_, p1), (t2, p2) in zip(samples, samples[1:]))
    # skip the first second, then scale to the whole interval
    steady = [p for t, p in samples if t > 1.0]
    return e_counter, e_int, statistics.mean(steady) * (t1 - t0), t1 - t0


def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("GPU:", pynvml.nvmlDeviceGetName(h))
    per, n = update_period(h)
    print(f"idle: energy counter changed {n} times in 3 s, median update period {per} ms")
    e_c, e_i, e_steady, dur = steady_load(h)
    print(f"steady load {dur:.1f} s: counter {e_c:.1f} J | integrated samples {e_i:.1f} J "
          f"({(e_i - e_c) / e_c:+.1%}) | mean steady power x time {e_steady:.1f} J ({(e_steady - e_c) / e_c:+.1%})")
    per, n = update_period(h)
    print(f"after load: counter changed {n} times in 3 s, median update period {per} ms")


if __name__ == "__main__":
    main()
