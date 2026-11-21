#!/usr/bin/env bash
# Train the same GPT twice on all visible GPUs, once per attention backend,
# then plot both loss curves and write a summary table.
#
#   bash examples/run_train_demo.sh OUT_DIR [RESUME_ROOT]
#
# RESUME_ROOT, if given, is an earlier OUT_DIR (for example the attached output
# of a Kaggle session that died); each backend resumes from its checkpoint there.
set -euo pipefail
OUT=${1:?output directory}
RESUME=${2:-}
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
STEPS=${STEPS:-2000}
mkdir -p "$OUT"

for ATTN in triton sdpa; do
  EXTRA=()
  if [[ -n "$RESUME" && -d "$RESUME/$ATTN" ]]; then EXTRA=(--resume-dir "$RESUME/$ATTN"); fi
  torchrun --standalone --nproc_per_node="$NGPU" examples/train_gpt.py \
    --attn "$ATTN" --out "$OUT/$ATTN" --data-dir "$OUT/data" --steps "$STEPS" "${EXTRA[@]}" \
    2>&1 | tee "$OUT/$ATTN.log"
done

python examples/plot_training.py "$OUT"
