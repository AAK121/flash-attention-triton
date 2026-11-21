"""Train a small character-level GPT with a switchable attention implementation.

    torchrun --standalone --nproc_per_node=2 examples/train_gpt.py --attn triton --out runs/x

The model, data order and initialization depend only on --seed, so two runs
that differ only in --attn see identical batches from identical weights. Their
loss curves should then overlap up to floating-point noise, which is an
end-to-end check of the forward and backward kernels inside real training.

Training uses DistributedDataParallel (one process per GPU), fp16 autocast and
a GradScaler. A checkpoint is written every --ckpt-every steps and picked up
automatically on restart, so a run survives a lost session.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import urllib.request

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from flash_attn_triton import flash_attention

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, attn: str):
        super().__init__()
        self.heads = heads
        self.attn = attn
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # (B, T, 3, H, D) -> three (B, H, T, D) views. Both backends accept the
        # resulting strides directly, so no .contiguous() copy is made.
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, C // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.attn == "triton":
            y = flash_attention(q, k, v, causal=True)
        else:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, attn: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, attn)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width, bias=False),
            nn.GELU(),
            nn.Linear(4 * width, width, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab: int, ctx: int, layers: int, width: int, heads: int, attn: str):
        super().__init__()
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(ctx, width)
        self.blocks = nn.ModuleList(Block(width, heads, attn) for _ in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        # GPT-2 scaling of residual projections keeps activations bounded with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def load_data(cache_dir: str, rank: int):
    path = os.path.join(cache_dir, "tinyshakespeare.txt")
    if rank == 0 and not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path + ".tmp")
        os.replace(path + ".tmp", path)
    if dist.is_initialized():
        dist.barrier()
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], len(chars)


def get_batch(data, step, rank, world, batch, ctx, seed, device):
    # Deterministic in (seed, step, rank): both attention backends, and a
    # resumed run, draw exactly the same batches.
    g = torch.Generator().manual_seed(seed * 1_000_003 + step * world + rank)
    ix = torch.randint(len(data) - ctx - 1, (batch,), generator=g)
    x = torch.stack([data[i : i + ctx] for i in ix])
    y = torch.stack([data[i + 1 : i + ctx + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, data, args, rank, world, device):
    model.eval()
    losses = []
    for i in range(args.eval_batches):
        x, y = get_batch(data, 10_000_000 + i, rank, world, args.batch, args.ctx, args.seed, device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1)))
    loss = torch.stack(losses).mean()
    if dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    model.train()
    return loss.item()


def save_checkpoint(path, model, opt, scaler, step):
    tmp = path + ".tmp"
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
         "step": step},
        tmp,
    )
    # Rename is atomic, so a session dying mid-write never leaves a torn checkpoint.
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["triton", "sdpa"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8, help="sequences per GPU per step")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--timing-skip", type=int, default=20,
                    help="steps excluded from throughput (compilation and autotuning)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume-dir", default=None,
                    help="directory holding a checkpoint.pt from an earlier session, "
                    "used when --out has none yet")
    args = ap.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    os.makedirs(args.out, exist_ok=True)

    train, val, vocab = load_data(args.data_dir, rank)
    torch.manual_seed(args.seed)
    model = GPT(vocab, args.ctx, args.layers, args.width, args.heads, args.attn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    earlier = args.resume_dir and os.path.join(args.resume_dir, "checkpoint.pt")
    if rank == 0 and earlier and os.path.exists(earlier) and not os.path.exists(ckpt_path):
        shutil.copy2(earlier, ckpt_path)
        log_src = os.path.join(args.resume_dir, "log.jsonl")
        if os.path.exists(log_src):
            shutil.copy2(log_src, os.path.join(args.out, "log.jsonl"))
    if distributed:
        dist.barrier()
    start = 0
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
        if rank == 0:
            print(f"resumed from step {start}", flush=True)

    ddp = DDP(model, device_ids=[local]) if distributed else model
    log_path = os.path.join(args.out, "log.jsonl")
    log = open(log_path, "a") if rank == 0 else None
    if rank == 0:
        print(f"attn={args.attn} params={n_params / 1e6:.2f}M world={world} "
              f"tokens/step={args.batch * args.ctx * world}", flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    timed_tokens, timed_seconds = 0, 0.0
    for step in range(start, args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x, y = get_batch(train, step, rank, world, args.batch, args.ctx, args.seed, device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ddp(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if step - start >= args.timing_skip:
            timed_tokens += args.batch * args.ctx * world
            timed_seconds += dt

        loss_avg = loss.detach()
        if distributed:
            dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
        record = {"step": step + 1, "train_loss": loss_avg.item(), "lr": opt.param_groups[0]["lr"],
                  "step_ms": dt * 1e3}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            record["val_loss"] = evaluate(model, val, args, rank, world, device)
        if rank == 0:
            if (step + 1) % args.log_every == 0 or "val_loss" in record:
                log.write(json.dumps(record) + "\n")
                log.flush()
            if "val_loss" in record or (step + 1) % 100 == 0:
                print(json.dumps(record), flush=True)
        if (step + 1) % args.ckpt_every == 0 and rank == 0:
            save_checkpoint(ckpt_path, model, opt, scaler, step + 1)

    peak = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**20], device=device)
    peaks = [torch.zeros_like(peak) for _ in range(world)]
    if distributed:
        dist.all_gather(peaks, peak)
    else:
        peaks = [peak]
    if rank == 0:
        summary = {
            "attn": args.attn,
            "attn_backend": "flash_attn_triton" if args.attn == "triton"
            else "torch SDPA EFFICIENT_ATTENTION",
            "params_m": n_params / 1e6,
            "world_size": world,
            "steps": args.steps,
            "resumed_from": start,
            "tokens_per_sec": timed_tokens / timed_seconds if timed_seconds else None,
            "timed_steps": max(0, args.steps - start - args.timing_skip),
            "peak_mem_mb_per_gpu": [round(p.item(), 1) for p in peaks],
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "args": vars(args),
        }
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary), flush=True)
        log.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
