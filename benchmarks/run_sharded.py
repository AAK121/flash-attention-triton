"""Run the benchmark sweep with one process per GPU and merge the results.

Each shard sees exactly one device through CUDA_VISIBLE_DEVICES, so no two
kernels are ever timed on the same GPU at once. Clocks are locked to the
maximum application clock when the driver allows it; whether that worked is
recorded in the metadata, because unlocked clocks add run-to-run variance.

    python benchmarks/run_sharded.py --out-dir benchmarks/results --tag t4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.startswith("GPU ")])


def try_lock_clocks(n: int) -> list[str]:
    notes = []
    for i in range(n):
        q = subprocess.run(
            ["nvidia-smi", "-i", str(i), "--query-gpu=clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        mhz = q.stdout.strip()
        r = subprocess.run(
            ["nvidia-smi", "-i", str(i), "-lgc", f"{mhz},{mhz}"], capture_output=True, text=True
        )
        notes.append(f"gpu{i}: lock {mhz} MHz -> rc={r.returncode} {(r.stdout + r.stderr).strip()}")
    return notes


def reset_clocks(n: int) -> None:
    for i in range(n):
        subprocess.run(["nvidia-smi", "-i", str(i), "-rgc"], capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--num-gpus", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=16384)
    a = ap.parse_args()

    n = a.num_gpus or gpu_count()
    os.makedirs(a.out_dir, exist_ok=True)
    clock_notes = try_lock_clocks(n)
    for line in clock_notes:
        print(line, flush=True)

    procs = []
    try:
        for i in range(n):
            # TRITON_PRINT_AUTOTUNING logs the config each search picks, which
            # the performance write-up needs.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), TRITON_PRINT_AUTOTUNING="1")
            out = os.path.join(a.out_dir, f"{a.tag}_shard{i}.csv")
            log = open(os.path.join(a.out_dir, f"{a.tag}_shard{i}.log"), "w")
            procs.append((subprocess.Popen(
                [sys.executable, os.path.join(HERE, "bench_attention.py"), "--out", out,
                 "--shard", str(i), "--num-shards", str(n), "--max-seq", str(a.max_seq)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            ), log))
        codes = [p.wait() for p, _ in procs]
    finally:
        for _, log in procs:
            log.close()
        reset_clocks(n)
    for i, code in enumerate(codes):
        print(f"shard {i} exit code {code}", flush=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(a.out_dir, f"{a.tag}_shard*.csv"))):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    merged = os.path.join(a.out_dir, f"{a.tag}.csv")
    if rows:
        with open(merged, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    metas = []
    for path in sorted(glob.glob(os.path.join(a.out_dir, f"{a.tag}_shard*.meta.json"))):
        with open(path) as f:
            metas.append(json.load(f))
    with open(os.path.join(a.out_dir, f"{a.tag}.meta.json"), "w") as f:
        json.dump({"clock_lock": clock_notes, "exit_codes": codes, "shards": metas}, f, indent=2)
    print(f"merged {len(rows)} rows into {merged}", flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
