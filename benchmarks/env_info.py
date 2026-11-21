"""Environment metadata written next to every result file."""

from __future__ import annotations

import datetime
import json
import platform
import subprocess

import torch
import triton

# Dense fp16 tensor-core peak with fp32 accumulation, from NVIDIA's data sheets
# at the boost clock. Used only to express measured throughput as a fraction.
FP16_TENSOR_PEAK_TFLOPS = {
    "Tesla T4": 65.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-PCIE-40GB": 312.0,
    "NVIDIA A100 80GB PCIe": 312.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA L4": 121.0,
}


def _smi(query: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def peak_tflops(device_name: str) -> float | None:
    return FP16_TENSOR_PEAK_TFLOPS.get(device_name)


def collect(extra: dict | None = None) -> dict:
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    info = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu": name,
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(dev))),
        "driver": _smi("driver_version").splitlines()[0] if _smi("driver_version") else "",
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "fp16_tensor_peak_tflops": peak_tflops(name),
        "clocks_sm_mem_mhz": _smi("clocks.sm,clocks.mem"),
        "max_clocks_sm_mem_mhz": _smi("clocks.max.sm,clocks.max.mem"),
        "temperature_c": _smi("temperature.gpu"),
        "power_draw_w": _smi("power.draw"),
    }
    if extra:
        info.update(extra)
    return info


def write(path: str, extra: dict | None = None) -> dict:
    info = collect(extra)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info
