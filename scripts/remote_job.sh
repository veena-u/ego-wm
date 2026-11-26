#!/usr/bin/env bash
# The whole remote job, safe to start any number of times (container restarts call it from
# /workspace/autostart.sh): finish the data pipeline, then train (or resume training).
#   setsid nohup bash scripts/remote_job.sh >> /workspace/logs/job.log 2>&1 < /dev/null & disown
set -uo pipefail
cd /workspace/ego-wm
DATA=${DATA:-/workspace/data}
log() { echo "[$(date -u +%d-%H:%M)] $*"; }
ready() { [ -f "$DATA/pipeline_ready" ] || grep -q "^\[.*\] ready:" /workspace/logs/pipeline.log 2>/dev/null; }

# 1. data: from R2 when the encoding ran on other machines (DATA_FROM=r2), else here: wait for a
#    pipeline that is already running, or run it (finished parts are skipped)
if [ "${DATA_FROM:-$(cat /workspace/data_from 2>/dev/null)}" = r2 ]; then
  ready || bash scripts/r2_fetch.sh >> /workspace/logs/fetch.log 2>&1 || { log "fetch failed"; exit 1; }
else
  while pgrep -f "[b]ash scripts/full_pipeline.sh" >/dev/null; do sleep 60; done
  if ! ready; then
    log "running pipeline"
    bash scripts/full_pipeline.sh >> /workspace/logs/pipeline.log 2>&1 || { log "pipeline failed"; exit 1; }
  fi
fi
ready || { log "pipeline did not report ready"; exit 1; }
touch "$DATA/pipeline_ready"

# 2. training: --resume continues from runs/v3/last.pt if present, else starts from the v2 weights.
# Constant learning rate after warmup with open-ended --steps; to finish, stop it and resume with
# --steps <current + N> --cooldown N (see train.py --cooldown).
pgrep -f "[s]cripts/train.py" >/dev/null && { log "training already running"; exit 0; }
NGPU=$(nvidia-smi -L | wc -l)
STEPS=${STEPS:-$(cat "$DATA/train_steps" 2>/dev/null || echo 200000)}
COOLDOWN=${COOLDOWN:-$(cat "$DATA/train_cooldown" 2>/dev/null || echo 0)}
log "training on $NGPU GPUs, steps $STEPS, cooldown $COOLDOWN"
OMP_NUM_THREADS=4 .venv/bin/torchrun --standalone --nproc_per_node "$NGPU" scripts/train.py \
  --data "$DATA/latents_full" --val-data "$DATA/latents" --splits train val_seen \
  --text-emb "$DATA/text_emb.pt" --init runs/v2/model_bf16_step8000.pt \
  --batch-size 2 --grad-ckpt --lr-new 3e-5 --steps "$STEPS" --cooldown "$COOLDOWN" \
  --val-every 500 --val-batches 16 --save-every 1000 --num-workers 6 \
  --out runs/v3 --wandb-project ego-wm --resume >> /workspace/logs/train.log 2>&1
log "training exited with $?"
