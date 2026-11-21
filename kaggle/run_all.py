"""Kaggle entry point: environment report, test suite, and (in later phases) benchmarks.

The package source is attached as the dataset alok121/flash-attn-triton-code.
Everything written to /kaggle/working is returned as the kernel output.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"


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


def main() -> int:
    os.makedirs(WORK, exist_ok=True)
    # /kaggle/input is read-only, and both the editable install and pytest's
    # cache write next to the sources, so work on a copy.
    src = "/tmp/flash-attn-triton"
    shutil.rmtree(src, ignore_errors=True)
    shutil.copytree(find_source(), src)
    sh("nvidia-smi", "nvidia_smi.txt")
    sh(f"{sys.executable} -m pip install --no-deps -q -e {src}")
    sh(f"{sys.executable} -m pip install -q pytest")
    sh(
        f"{sys.executable} -c \"import torch, triton; print('torch', torch.__version__, "
        "'triton', triton.__version__, 'cuda', torch.version.cuda); "
        "print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\"",
        "versions.txt",
    )
    rc = sh(f"cd {src} && {sys.executable} -m pytest tests -v -rs --durations=10", "pytest_gpu.txt")
    return rc


if __name__ == "__main__":
    sys.exit(main())
