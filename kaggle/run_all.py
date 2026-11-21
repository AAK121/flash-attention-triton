"""Kaggle entry point: environment report, tests, benchmark sweep, training demo.

The package source is attached as the dataset alok121/flash-attn-triton-code.
Everything written to /kaggle/working is returned as the kernel output. Which
stages run is read from stages.txt at the dataset root (one comma-separated
line, e.g. "tests,bench"); without it every stage runs.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
ALL_STAGES = ["tests", "bench", "train"]


def sh(cmd: str, log: str | None = None) -> int:
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    p.wait()
    if log:
        with open(os.path.join(WORK, log), "w") as f:
            f.writelines(lines)
    return p.returncode


def find_source() -> str:
    hits = glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
    if not hits:
        raise SystemExit("package dataset not attached")
    return os.path.dirname(hits[0])


def stages(src: str) -> list[str]:
    path = os.path.join(src, "stages.txt")
    if not os.path.exists(path):
        return ALL_STAGES
    with open(path) as f:
        return [s.strip() for s in f.read().split(",") if s.strip()]


def main() -> int:
    os.makedirs(WORK, exist_ok=True)
    # /kaggle/input is read-only, and both the editable install and pytest's
    # cache write next to the sources, so work on a copy.
    src = "/tmp/flash-attn-triton"
    shutil.rmtree(src, ignore_errors=True)
    shutil.copytree(find_source(), src)
    py = sys.executable
    todo = stages(src)
    print(f"stages: {todo}", flush=True)

    sh("nvidia-smi", "nvidia_smi.txt")
    sh(f"{py} -m pip install --no-deps -q -e {src}")
    sh(f"{py} -m pip install -q pytest")
    sh(
        f"{py} -c \"import torch, triton; print('torch', torch.__version__, "
        "'triton', triton.__version__, 'cuda', torch.version.cuda); "
        "print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\"",
        "versions.txt",
    )

    failed = []
    if "tests" in todo:
        if sh(f"cd {src} && {py} -m pytest tests -v -rs --durations=10", "pytest_gpu.txt"):
            failed.append("tests")
    if "bench" in todo:
        results, plots = f"{WORK}/results", f"{WORK}/plots"
        if sh(f"cd {src} && {py} benchmarks/run_sharded.py --out-dir {results} --tag t4"):
            failed.append("bench")
        if os.path.exists(f"{results}/t4.csv"):
            sh(f"{py} {src}/benchmarks/plot_results.py {results}/t4.csv --out-dir {plots}")
    if "train" in todo:
        if sh(f"cd {src} && bash examples/run_train_demo.sh {WORK}/train"):
            failed.append("train")
    print(f"failed stages: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
