#!/usr/bin/env bash
# Training machine: wait until every part is encoded and in R2 (done/<part> markers written by
# full_pipeline.sh R2=1 on the encoder machines), download it all, build text embeddings, and mark
# the data ready for remote_job.sh. Restartable (rclone copy skips what's already here).
set -uo pipefail
cd /workspace/ego-wm
DATA=${DATA:-/workspace/data}
PARTS=(${PARTS:-test part1 part2 part3 part4 part5 extra})
log() { echo "[$(date -u +%d-%H:%M)] $*"; }
. scripts/r2.sh
r2_ok || { log "no R2 credentials"; exit 1; }
while true; do
  missing=()
  for p in "${PARTS[@]}"; do r2 lsf "r2:$R2_BUCKET/done/$p" >/dev/null 2>&1 && [ -n "$(r2 lsf "r2:$R2_BUCKET/done/$p" 2>/dev/null)" ] || missing+=("$p"); done
  [ ${#missing[@]} = 0 ] && break
  log "waiting for parts: ${missing[*]}"
  sleep 600
done
log "all parts in R2; downloading"
mkdir -p "$DATA/latents" "$DATA/latents_full" "$DATA/index"
r2 copy "r2:$R2_BUCKET/latents" "$DATA/latents" -q --transfers 32 || exit 1
r2 copy "r2:$R2_BUCKET/latents_full" "$DATA/latents_full" -q --transfers 32 || exit 1
r2 copy "r2:$R2_BUCKET/index" "$DATA/index" -q || exit 1
log "downloaded: $(ls $DATA/latents_full | grep -c '\.pt$') training episodes, $(ls $DATA/latents | grep -c '\.pt$') validation"
.venv/bin/python scripts/encode_text.py --index "$DATA"/index/*/index.csv --out "$DATA/text_emb.pt" || exit 1
touch "$DATA/pipeline_ready"
log "ready"
