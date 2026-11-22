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


TURING_TRITON = "3.2.0"


def pin_triton_for_turing(py: str) -> None:
    """Shadow the bundled Triton with 3.2.0 on pre-Ampere GPUs.

    Measured on the T4 (runs v14 and v17): Triton 3.4 to 3.6 lower tl.dot to
    scalar FMA on sm75 and 3.3 fails to compile fp16 dots, while 3.2 emits
    tensor-core mma. PYTHONPATH reaches every later subprocess, including the
    benchmark shards and torchrun workers. torch only imports Triton lazily.
    """
    out = subprocess.run(
        [py, "-c", "import torch; print(*torch.cuda.get_device_capability())"],
        capture_output=True, text=True,
    )
    major = int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else 0
    if major >= 8 or major == 0:
        return
    target = f"/tmp/triton-{TURING_TRITON}"
    if sh(f"{py} -m pip install -q --no-deps --target {target} triton=={TURING_TRITON}"):
        print("could not install the Turing Triton pin; keeping the bundled one", flush=True)
        return
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (target, os.environ.get("PYTHONPATH", "")) if p
    )
    sh(f"{py} -c \"import triton; print('triton in use', triton.__version__, triton.__file__)\"",
       "triton_pin.txt")


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

    pin_triton_for_turing(py)

    failed = []
    if "inspect" in todo:
        if sh(f"cd {src} && {py} benchmarks/inspect_kernels.py", "inspect_kernels.txt"):
            failed.append("inspect")
    if "triton_versions" in todo:
        # Each release goes into its own --target dir that shadows the bundled
        # Triton through PYTHONPATH; torch itself only imports Triton lazily.
        for ver in ("3.0.0", "3.1.0", "3.2.0", "3.3.1", "3.4.0", "3.5.1"):
            target = f"/tmp/triton-{ver}"
            if sh(f"{py} -m pip install -q --no-deps --target {target} triton=={ver}"):
                continue
            sh(f"cd {src} && PYTHONPATH={target}:{src}/src {py} benchmarks/inspect_kernels.py",
               f"inspect_triton_{ver}.txt")
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
