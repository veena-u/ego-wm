#!/usr/bin/env bash
# Short real multi-GPU training run (default 40 steps) to check memory, speed and NCCL before
# the long run. Uses whatever latents exist; no text embeddings needed. Output: runs/ddp_smoke.
set -euo pipefail
cd /workspace/ego-wm
DATA=${DATA:-/workspace/data}
NGPU=$(nvidia-smi -L | wc -l)
rm -rf runs/ddp_smoke
OMP_NUM_THREADS=4 .venv/bin/torchrun --standalone --nproc_per_node "$NGPU" scripts/train.py \
  --data "$DATA/latents_full" --val-data "$DATA/latents" --splits train val_seen \
  --init runs/v2/model_bf16_step8000.pt --batch-size 2 --grad-ckpt --lr-new 3e-5 \
  --steps ${STEPS:-40} --warmup 10 --val-every 20 --val-batches 4 --save-every 20 --log-every 5 \
  --num-workers 6 --out runs/ddp_smoke
