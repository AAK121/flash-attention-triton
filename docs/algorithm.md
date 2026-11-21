# FlashAttention-2: derivation and design

This document derives what the kernels in `src/flash_attn_triton` compute and
why they are organized the way they are. Notation follows Dao (2023): one head,
queries $Q \in \mathbb{R}^{N_q \times d}$, keys and values
$K, V \in \mathbb{R}^{N_k \times d}$, scale $\tau = 1/\sqrt{d}$ unless given.

$$
S = \tau Q K^\top, \qquad P = \operatorname{softmax}_{\text{row}}(S), \qquad O = P V .
$$

Standard attention materializes $S$ and $P$, both $N_q \times N_k$. Everything
below exists to avoid that.

## 1. Online softmax

For one row $s \in \mathbb{R}^{N_k}$ the output row is

$$
o = \frac{\sum_j e^{s_j - m} v_j}{\sum_j e^{s_j - m}}, \qquad m = \max_j s_j ,
$$

where subtracting $m$ keeps every exponent $\le 0$ and so avoids overflow; it
cancels between numerator and denominator.

Split the keys into blocks $B_1, \dots, B_T$ and process them in order. After
block $t$ keep three running quantities:

$$
m^{(t)} = \max\bigl(m^{(t-1)}, \max_{j \in B_t} s_j\bigr), \qquad
\ell^{(t)} = e^{m^{(t-1)} - m^{(t)}}\,\ell^{(t-1)} + \sum_{j \in B_t} e^{s_j - m^{(t)}},
$$

$$
\tilde o^{(t)} = e^{m^{(t-1)} - m^{(t)}}\,\tilde o^{(t-1)} + \sum_{j \in B_t} e^{s_j - m^{(t)}} v_j ,
$$

with $m^{(0)} = -\infty$, $\ell^{(0)} = 0$, $\tilde o^{(0)} = 0$.

**Claim.** For every $t$,
$\ell^{(t)} = \sum_{j \in B_{\le t}} e^{s_j - m^{(t)}}$ and
$\tilde o^{(t)} = \sum_{j \in B_{\le t}} e^{s_j - m^{(t)}} v_j$.

*Proof by induction.* For $t = 1$ the correction factor multiplies zero.
Assume the claim for $t-1$. Then

$$
e^{m^{(t-1)} - m^{(t)}} \ell^{(t-1)}
= \sum_{j \in B_{<t}} e^{s_j - m^{(t-1)}} e^{m^{(t-1)} - m^{(t)}}
= \sum_{j \in B_{<t}} e^{s_j - m^{(t)}},
$$

and adding the block-$t$ terms gives the sum over $B_{\le t}$. The same algebra
applies to $\tilde o$. $\square$

So after the last block $o = \tilde o^{(T)} / \ell^{(T)}$ exactly: the
recurrence is a reassociation of the same sums, not an approximation (Milakov
and Gimelshein, 2018). Only the final division is deferred, which is the
FlashAttention-2 change over FlashAttention-1: FA1 rescaled $\tilde o$ by
$1/\ell$ in every iteration, FA2 keeps it unnormalized and divides once,
removing a non-matmul operation from the inner loop.

The kernel also saves the row log-sum-exp,
$\operatorname{lse} = m^{(T)} + \log \ell^{(T)}$, the only per-row state the
backward pass needs.

### 1.1 Base-2 exponentials

GPUs evaluate $2^x$ in one instruction (`ex2.approx`), while $e^x$ costs a
multiplication plus `ex2`. Since $e^{x} = 2^{x \log_2 e}$, fold the constant
into the scale once:

$$
\hat s_j = (\tau \log_2 e)\, q \cdot k_j, \qquad
e^{s_j - m} = 2^{\hat s_j - \hat m}, \qquad \hat m = m \log_2 e .
$$

All running statistics are kept in this base-2 domain, and the saved
log-sum-exp is $\widehat{\operatorname{lse}} = \hat m + \log_2 \ell$. The
backward pass recovers $P_{ij} = 2^{\hat s_{ij} - \widehat{\operatorname{lse}}_i}$
without converting back.

### 1.2 Fully masked rows

With causal or windowed masks a row may have no visible key in a block, or at
all. Masked scores are set to $-\infty$. If every score seen so far is
$-\infty$ then $m^{(t)} = -\infty$ and $m^{(t-1)} - m^{(t)}$ is
$-\infty - (-\infty) = \text{NaN}$. The kernel substitutes $0$ for $m^{(t)}$
in the exponent only when it is $-\infty$; then every exponential is
$2^{-\infty} = 0$ and the state stays $(-\infty, 0, 0)$. A row that never sees a
key ends with $\ell = 0$; it outputs $0$ and stores
$\operatorname{lse} = -\infty$, and the backward pass treats that row's $P$ as
zero.

## 2. Backward pass

Given $dO = \partial L / \partial O$:

$$
dV = P^\top dO, \qquad dP = dO\, V^\top .
$$

The row-wise softmax has Jacobian $\operatorname{diag}(p) - p p^\top$, so for
row $i$

$$
dS_{i} = p_i \odot dP_i - p_i\, (p_i^\top dP_i)
= p_i \odot \bigl(dP_i - \delta_i\bigr), \qquad \delta_i = p_i^\top dP_i .
$$

**Where $\delta$ comes from.** Computing $p_i^\top dP_i$ directly needs a whole
row of $P$ and $dP$. But

$$
\delta_i = \sum_j P_{ij}\, dP_{ij} = \sum_j P_{ij}\, (dO_i \cdot v_j)
= dO_i \cdot \Bigl(\sum_j P_{ij} v_j\Bigr) = dO_i \cdot O_i ,
$$

so $\delta = \operatorname{rowsum}(dO \odot O)$, an $O(N d)$ reduction over
tensors that already exist. That is the preprocess kernel.

Finally, since $S = \tau Q K^\top$,

$$
dQ = \tau\, dS\, K, \qquad dK = \tau\, dS^\top Q .
$$

$P$ is never stored. Each backward kernel recomputes a tile of it from $Q$, $K$
and the saved $\widehat{\operatorname{lse}}$: one extra matmul per tile instead
of reading an $N_q \times N_k$ matrix from memory.

### 2.1 Work partitioning in the backward

$dK_j, dV_j$ are sums over query rows and $dQ_i$ is a sum over key columns, so
no single loop order produces all three without cross-program accumulation.

- **This implementation:** two kernels. The dK/dV kernel gives each program a
  key block and loops over query blocks; the dQ kernel gives each program a
  query block and loops over key blocks. Every output tile has exactly one
  owner, so there are no atomics and results are bitwise reproducible. The cost
  is recomputing $Q K^\top$ and $dO V^\top$ in both kernels: 7 tile matmuls per
  (query block, key block) pair instead of 5.
- **Alternative:** one pass over key blocks that adds partial $dQ$ tiles into
  an fp32 buffer with atomics (the approach of the original FlashAttention-2 CUDA
  kernel). Fewer matmuls, but every key block does a read-modify-write of $dQ$,
  and floating-point addition order varies between runs.

### 2.2 Grouped-query attention

With $H$ query heads and $H_k$ key/value heads, query head $h$ uses key/value
head $\lfloor h / G \rfloor$, $G = H / H_k$. Forward and dQ simply index the
shared head. For a shared head,

$$
dK^{(h_k)} = \sum_{g=0}^{G-1} dK^{(h_k G + g)}_{\text{per query head}},
$$

and likewise for $dV$. The dK/dV program that owns a key block of head $h_k$
loops over the $G$ query heads of its group and accumulates in registers, so
the reduction needs neither atomics nor a $G$-times larger temporary buffer.

## 3. Memory and IO complexity

Let $M$ be the on-chip memory (shared memory plus registers) per streaming
multiprocessor, with $d \le M \le N d$.

**Standard attention** writes $S$ and $P$ to HBM and reads them back:
$\Theta(N d + N^2)$ HBM accesses and $\Theta(N^2)$ extra memory.

**FlashAttention.** Each query block of $B_r$ rows reads all of $K$ and $V$
once. With $B_r = \Theta(M/d)$ there are $N / B_r = \Theta(N d / M)$ query
blocks, each reading $\Theta(N d)$ elements, so

$$
\text{HBM accesses} = \Theta\!\left(\frac{N^2 d^2}{M}\right).
$$

For typical $d$ (64 to 128) and $M$ (around 100 KB, about 50k fp16 elements),
$d^2 / M$ is well below 1,
so this is many times fewer accesses than $N^2$ (Dao et al., 2022, Theorem 2,
which also shows no exact attention algorithm can do asymptotically better for
all $M$ in this range). Extra memory is $\Theta(N)$: one log-sum-exp per row.
FLOPs are unchanged, $4 N_q N_k d$ for the forward, because attention is still
computed exactly.

On a T4, $M$ is small (64 KB of shared memory per SM), which forces smaller
tiles at $d = 128$ and raises the $d^2/M$ factor; see `performance.md`.

## 4. Why FA2's loop order and partitioning help

FlashAttention-1 put the loop over key blocks outermost and parallelized over
batch and heads only. Each program held a key block and swept the queries,
reading and writing partial $O$, $\ell$ and $m$ for every query block from HBM.

FlashAttention-2 makes two changes:

1. **Query blocks outermost, one program per (batch, head, query block).** The
   output tile, $\ell$ and $m$ stay in registers for the whole key loop and are
   written once. Adding the query-block dimension to the grid gives
   $B \cdot H \cdot \lceil N_q / B_r \rceil$ programs instead of $B \cdot H$,
   which matters for long sequences with small batch: with $B H = 16$ and 40
   SMs (a T4), FA1 would leave most of the GPU idle.
2. **Work split inside a program.** Warps split the query rows of the tile
   rather than the key columns, so row maxima and sums need no cross-warp
   communication through shared memory. The autotuner's pruning enforces this:
   it rejects configurations with more than one warp per 16 accumulator rows.

## 5. Masks and block skipping

Query $i$ sees key $j$ when

$$
i + \text{off} - L \;\le\; j \;\le\; i + \text{off} + R, \qquad \text{off} = N_k - N_q ,
$$

with either bound possibly absent. $R = 0$ without $L$ is causal attention
(bottom-right aligned, as needed when decoding against a longer KV cache); a
finite $L$ is a sliding window.

For a program that owns query rows $[r_0, r_1]$, keys outside
$[r_0 + \text{off} - L,\; r_1 + \text{off} + R]$ are invisible to every row,
and the key tiles covering them are never loaded. Keys in
$[r_1 + \text{off} - L,\; r_0 + \text{off} + R]$ are visible to every row:
those tiles run without any mask. Only the tiles in between, along the two
edges of the band, pay for the elementwise predicate. The left edge is
tightest for the last row and the right edge for the first, which is why the
fully visible range uses $r_1$ on the left and $r_0$ on the right.

The dK/dV kernel owns keys $[c_0, c_1]$ and scans queries. Solving the same
inequality for $i$,

$$
j - \text{off} - R \;\le\; i \;\le\; j - \text{off} + L ,
$$

which is the first form with $\text{off} \to -\text{off}$ and $L \leftrightarrow R$.
One helper (`_masking.scan_range`) therefore serves all three kernels.

### 5.1 Effect on FLOPs

With block skipping, the number of tile pairs processed under a causal mask
with $N_q = N_k = N$ and square tiles of size $b$ is

$$
\sum_{i=1}^{N/b} i = \frac{(N/b)(N/b + 1)}{2} \approx \frac{1}{2}\left(\frac{N}{b}\right)^2 ,
$$

about half the non-causal count, so the work and the conventional FLOP count
both halve; the diagonal tiles contribute the $(N/b)/2$ excess. For a window of
width $W = L + R + 1$ the processed pairs are $O\!\left(\frac{N}{b}\left(\frac{W}{b} + 2\right)\right)$:
linear in $N$ rather than quadratic.

Under a causal mask the last query blocks have the most key tiles. The forward
and dQ kernels launch query blocks in reverse order so the longest programs
start first and short ones fill the tail of the launch (longest processing
time first). In the dK/dV kernel the first key blocks are the heaviest, so the
natural order already has this property.

## References

- T. Dao, D. Fu, S. Ermon, A. Rudra, C. Re. FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
- T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning. 2023.
- M. Milakov, N. Gimelshein. Online normalizer calculation for softmax. 2018.
