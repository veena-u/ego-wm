#!/usr/bin/env bash
# Evaluate one checkpoint on every GPU in parallel (clips split across GPUs), then merge.
#   bash scripts/eval_parallel.sh CKPT OUT [extra evaluate.py args, e.g. --cfg 3 --clips 32]
# Results match a single-process run exactly (same clips, seeds and swap partners).
set -euo pipefail
CKPT=$1; OUT=$2; shift 2
DATA=${DATA:-/workspace/data}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PY=.venv/bin/python
ARGS=(--ckpt "$CKPT" --data "$DATA/latents" --text-emb "$DATA/text_emb.pt" --handpose runs/handpose/handpose.pt
      --out "$OUT" "$@")
mkdir -p "$OUT"
pids=()
for i in $(seq 1 "$NGPU"); do
  CUDA_VISIBLE_DEVICES=$((i - 1)) $PY scripts/evaluate.py "${ARGS[@]}" --shard "$i/$NGPU" > "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY scripts/evaluate.py "${ARGS[@]}" --merge "$NGPU"
