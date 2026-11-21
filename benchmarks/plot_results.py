"""Plots from a merged benchmark CSV.

One figure per (mode, head_dim) with causal and non-causal side by side:
achieved TFLOP/s against sequence length for every provider. Points where a
provider ran out of memory are drawn as crosses on the x axis and labelled, so
a missing line is never mistaken for a missing measurement. A second figure
family shows peak memory.

    python benchmarks/plot_results.py benchmarks/results/t4.csv --out-dir benchmarks/plots
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {
    "triton": ("#1f77b4", "o", "this kernel (Triton)"),
    "sdpa_mem": ("#2ca02c", "s", "SDPA mem-efficient"),
    "sdpa_flash": ("#9467bd", "D", "SDPA flash"),
    "sdpa_math": ("#ff7f0e", "^", "SDPA math"),
    "naive": ("#7f7f7f", "v", "naive PyTorch"),
}


def hardware_label(meta_path: str) -> str:
    if not os.path.exists(meta_path):
        return ""
    with open(meta_path) as f:
        meta = json.load(f)
    shard = meta["shards"][0] if "shards" in meta else meta
    return (
        f"{shard['gpu']} (sm{shard['compute_capability'].replace('.', '')}), "
        f"torch {shard['torch']}, triton {shard['triton']}, CUDA {shard['cuda_runtime']}, fp16"
    )


def plot_metric(df, metric, ylabel, title_metric, out_dir, label, log_y=False):
    paths = []
    for (mode, d), part in df.groupby(["mode", "head_dim"]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, causal in zip(axes, [False, True]):
            sub = part[part["causal"] == causal]
            for provider, (color, marker, name) in STYLE.items():
                rows = sub[sub["provider"] == provider].sort_values("seq_len")
                if rows.empty:
                    continue
                ok = rows[rows["status"] == "ok"]
                ax.plot(ok["seq_len"], ok[metric], marker=marker, color=color, label=name)
                failed = rows[rows["status"] != "ok"]
                if not failed.empty:
                    # Pinned to the bottom of the axes (axes coordinates in y), so the
                    # marker also works on a log-scaled memory axis.
                    bottom = ax.get_xaxis_transform()
                    ax.scatter(failed["seq_len"], [0] * len(failed), marker="x", color=color,
                               s=60, clip_on=False, zorder=5, transform=bottom)
                    ax.annotate("OOM", (failed["seq_len"].min(), 0), xycoords=bottom,
                                xytext=(0, 8), textcoords="offset points",
                                ha="center", fontsize=7, color=color)
            ax.set_xscale("log", base=2)
            seqs = sorted(part["seq_len"].unique())
            ax.set_xticks(seqs)
            ax.set_xticklabels([f"{s // 1024}k" if s >= 1024 else str(s) for s in seqs])
            if log_y:
                ax.set_yscale("log")
            ax.set_xlabel("sequence length")
            ax.set_title(f"{'causal' if causal else 'non-causal'}")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"{title_metric}, {mode}, head_dim {d}   (x on the axis = out of memory)\n{label}",
            fontsize=10,
        )
        fig.tight_layout()
        path = os.path.join(out_dir, f"{metric}_{mode}_d{d}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "plots"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv)
    df["causal"] = df["causal"].astype(str).str.lower().isin(["true", "1"])
    label = hardware_label(a.csv.replace(".csv", ".meta.json"))
    paths = plot_metric(df, "tflops", "TFLOP/s", "Throughput", a.out_dir, label)
    paths += plot_metric(df, "peak_mem_mb", "peak memory (MiB)", "Peak memory", a.out_dir,
                         label, log_y=True)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
