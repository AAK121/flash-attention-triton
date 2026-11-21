"""Loss curves and a summary table for the two training runs of run_train_demo.sh.

    python examples/plot_training.py OUT_DIR
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"triton": "this kernel (Triton)", "sdpa": "SDPA mem-efficient"}


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    out = sys.argv[1]
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 4))
    summaries = {}
    for attn, name in NAMES.items():
        log_path = os.path.join(out, attn, "log.jsonl")
        if not os.path.exists(log_path):
            continue
        recs = read_log(log_path)
        ax_train.plot([r["step"] for r in recs], [r["train_loss"] for r in recs],
                      label=name, alpha=0.8)
        val = [r for r in recs if "val_loss" in r]
        ax_val.plot([r["step"] for r in val], [r["val_loss"] for r in val], marker="o", label=name)
        with open(os.path.join(out, attn, "summary.json")) as f:
            summaries[attn] = json.load(f)
    for ax, title in ((ax_train, "train loss"), (ax_val, "validation loss")):
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy (nats/char)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    if summaries:
        s = next(iter(summaries.values()))
        fig.suptitle(
            f"{s['params_m']:.1f}M-param GPT, context {s['args']['ctx']}, "
            f"{s['world_size']}x {s['gpu']}, DDP, fp16 autocast, same seed", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss_curves.png"), dpi=130)

    rows = []
    for attn, s in summaries.items():
        recs = read_log(os.path.join(out, attn, "log.jsonl"))
        val = [r["val_loss"] for r in recs if "val_loss" in r]
        rows.append({
            "attention": s["attn_backend"],
            "tokens_per_sec": round(s["tokens_per_sec"]) if s["tokens_per_sec"] else None,
            "peak_mem_mb_per_gpu": s["peak_mem_mb_per_gpu"],
            "final_train_loss": round(recs[-1]["train_loss"], 4),
            "final_val_loss": round(val[-1], 4) if val else None,
            "steps": s["steps"],
            "resumed_from": s["resumed_from"],
        })
    if "triton" in summaries and "sdpa" in summaries:
        a = {r["step"]: r["train_loss"] for r in read_log(os.path.join(out, "triton", "log.jsonl"))}
        b = {r["step"]: r["train_loss"] for r in read_log(os.path.join(out, "sdpa", "log.jsonl"))}
        common = sorted(set(a) & set(b))
        diffs = [abs(a[k] - b[k]) for k in common]
        rows.append({"max_abs_train_loss_diff": max(diffs) if diffs else None,
                     "mean_abs_train_loss_diff": sum(diffs) / len(diffs) if diffs else None,
                     "compared_steps": len(common)})
    with open(os.path.join(out, "training_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
