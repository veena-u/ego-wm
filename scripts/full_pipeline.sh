#!/usr/bin/env bash
# All of EgoDex -> training-ready latents, on a multi-GPU machine. Run from the repo root:
#   setsid nohup bash scripts/full_pipeline.sh > /workspace/logs/pipeline.log 2>&1 < /dev/null & disown
#
# Per part (test, part1..part5, extra = the 97 h collected after the paper's split): download -> unzip -> index (normalization stats untouched)
# -> encode on every GPU in parallel -> delete the raw video. Raw and encoded data never have
# to fit on disk together (raw 1.64 TB + latents ~1.7 TB), only one raw part at a time (~650 GB peak).
# Restartable: finished parts are skipped, and preprocess.py skips episodes already encoded.
#
#   test        -> $DATA/latents       validation (same clips as earlier runs, so numbers compare)
#   part1..5, extra -> $DATA/latents_full  training
# Env: DATA (default /workspace/data), PARTS, PROCS_PER_GPU (default 2), KEEP_RAW=1 to keep video,
#      MAX_CLIPS per training episode (default 2, 0 = all), LIMIT=n episodes per encoder process (testing only).
#      R2=1: upload to Cloudflare R2 while encoding (scripts/r2.sh): finished .pt files are MOVED to
#        r2:$R2_BUCKET/<latents dir>/ (local disk stays small; the .json sidecars stay as done-markers
#        and are copied too), the part's index goes to index/<part>/, and done/<part> marks it complete.
#        Several machines can each run different PARTS. SKIP_FINAL=1 skips text embeddings / counts
#        (the training machine does them after downloading everything).
set -euo pipefail
DATA=${DATA:-/workspace/data}
PARTS=(${PARTS:-test part1 part2 part3 part4 part5 extra})
NGPU=$(nvidia-smi -L | wc -l)
PPG=${PROCS_PER_GPU:-2}
NPROC=$((NGPU * PPG))
# CPUs this container may actually use (cgroup quota), not the host's core count
NCPU=$(awk '$1 != "max" {print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); NCPU=${NCPU:-$(nproc)}
[ "$NCPU" -le "$(nproc)" ] || NCPU=$(nproc)
THREADS=$(( NCPU / NPROC )); [ "$THREADS" -ge 1 ] || THREADS=1
PY=.venv/bin/python
mkdir -p "$DATA/egodex" "$DATA/index" /workspace/logs
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
if [ "${R2:-0}" = 1 ]; then
  . scripts/r2.sh
  r2_ok || { echo "R2=1 but no credentials (see scripts/r2.sh)"; exit 1; }
fi
r2_sync() {  # $1 local latents dir, $2 min age of files to move (e.g. 2m; 0s at the end)
  [ "${R2:-0}" = 1 ] || return 0
  local dst="r2:$R2_BUCKET/$(basename "$1")"
  r2 copy "$1" "$dst" --include "*.json" --exclude manifest.json -q || return 1
  r2 move "$1" "$dst" --include "*.pt" --min-age "$2" -q
}

# The model was trained with these normalization stats; fine-tuning must keep them.
[ -f data/action_stats.npz ] || { echo "data/action_stats.npz missing (it is in the repo)"; exit 1; }
STATS_SHA=$(sha256sum data/action_stats.npz | cut -c1-64)

log "CPUs $NCPU, GPUs $NGPU, encoder processes $NPROC ($PPG per GPU, $THREADS CPU threads each), parts: ${PARTS[*]}"
command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq aria2 unzip >/dev/null; }

for p in "${PARTS[@]}"; do
  if [ -f "$DATA/done_$p" ]; then log "$p: already done"; continue; fi
  [ "$p" = test ] && OUT="$DATA/latents" || OUT="$DATA/latents_full"
  mkdir -p "$OUT"
  # Validation (test) keeps every clip, as in earlier runs; training parts keep MAX_CLIPS per episode.
  [ "$p" = test ] && MAXC=0 || MAXC=${MAX_CLIPS:-2}

  if [ ! -d "$DATA/egodex/$p" ]; then
    FREE=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
    [ "$FREE" -ge 700 ] || [ "$p" = test ] || { log "$p: only ${FREE} GB free, need ~700 GB for zip + unzip"; exit 1; }
    log "$p: downloading"
    aria2c -q -x 16 -s 16 -c -d "$DATA/egodex" -o "$p.zip" "https://ml-site.cdn-apple.com/datasets/egodex/$p.zip"
    log "$p: unzipping"
    unzip -q "$DATA/egodex/$p.zip" -d "$DATA/egodex" && rm "$DATA/egodex/$p.zip"
  fi
  [ -d "$DATA/egodex/$p" ] || { log "expected $DATA/egodex/$p after unzip; got:"; ls "$DATA/egodex"; exit 1; }

  log "$p: indexing"
  mkdir -p "$DATA/index/$p"
  $PY scripts/build_index.py "$DATA/egodex/$p" --out "$DATA/index/$p" --workers "$NCPU" --no-stats
  if [ "$p" = test ] && [ ! -f "$DATA/index/stats_verified" ]; then
    # data/action_stats.npz came from the test split (seeded sample); recompute and compare, so a
    # wrong copy can't silently mis-normalize every training clip.
    log "test: verifying data/action_stats.npz against a recomputation"
    $PY scripts/build_index.py "$DATA/egodex/test" --out "$DATA/index/test_recheck" --workers "$NCPU"
    $PY - data/action_stats.npz "$DATA/index/test_recheck/action_stats.npz" <<'EOF'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = max(float(np.abs(a[k] - b[k]).max()) for k in ("mean", "std"))
print(f"stats max abs diff {d:.2e} (n {int(a['n'])} vs {int(b['n'])})")
sys.exit(0 if d < 1e-4 else 1)
EOF
    touch "$DATA/index/stats_verified"
  fi

  log "$p: encoding on $NGPU GPUs"
  pids=()
  for i in $(seq 1 $NPROC); do
    CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) OMP_NUM_THREADS=$THREADS \
      $PY scripts/preprocess.py --vae wan2.1 --index "$DATA/index/$p/index.csv" --out "$OUT" \
      --shard "$i/$NPROC" --max-clips "$MAXC" ${LIMIT:+--limit $LIMIT} > "/workspace/logs/prep_${p}_$i.log" 2>&1 &
    pids+=($!)
  done
  if [ "${R2:-0}" = 1 ]; then  # upload finished episodes every 2 min while the encoders run
    ( while sleep 120; do r2_sync "$OUT" 2m || echo "r2 sync failed (will retry)"; done ) &
    UPL=$!
  fi
  fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [ "${R2:-0}" = 1 ]; then
    kill $UPL 2>/dev/null
    r2_sync "$OUT" 0s && r2_sync "$OUT" 0s || { log "$p: final upload failed"; exit 1; }
    left=$(find "$OUT" -name "*.pt" | wc -l)
    [ "$left" = 0 ] || { log "$p: $left .pt files not uploaded"; exit 1; }
    r2 copy "$DATA/index/$p" "r2:$R2_BUCKET/index/$p" -q
  fi
  [ "$fail" = 0 ] || { log "$p: an encoder process crashed; see /workspace/logs/prep_${p}_*.log"; exit 1; }
  nfail=$(cat /workspace/logs/prep_${p}_*.log | grep -c "^FAILED" || true)
  log "$p: encoded. $(grep -h '^done:' /workspace/logs/prep_${p}_*.log | awk '{e+=$2; c+=$4} END {print e" episodes, "c" clips"}'), $nfail failed episodes"

  [ "${KEEP_RAW:-0}" = 1 ] || { log "$p: deleting raw video"; rm -rf "$DATA/egodex/$p"; }
  touch "$DATA/done_$p"
  [ "${R2:-0}" = 1 ] && { echo "$(date -u) $(hostname)" | r2 rcat "r2:$R2_BUCKET/done/$p"; log "$p: uploaded to R2"; }
  log "$p: done. Disk: $(df -h --output=used,avail "$DATA" | tail -1)"
done

[ "$(sha256sum data/action_stats.npz | cut -c1-64)" = "$STATS_SHA" ] || { echo "action_stats.npz changed!"; exit 1; }
[ "${SKIP_FINAL:-0}" = 1 ] && { log "parts done: ${PARTS[*]} (SKIP_FINAL)"; exit 0; }

log "text embeddings for every description"
$PY scripts/encode_text.py --index "$DATA"/index/*/index.csv --out "$DATA/text_emb.pt"

CLIPS=$($PY - "$DATA/latents_full" <<'EOF'
import json, sys
from pathlib import Path
print(sum(json.loads(p.read_text())["n"] for p in Path(sys.argv[1]).glob("*.json")
          if p.name != "manifest.json" and json.loads(p.read_text())["split"] == "train"))
EOF
)
STEPS=$(( CLIPS / (NGPU * 2) ))
touch "$DATA/pipeline_ready"
log "ready: $CLIPS training clips. One pass at $NGPU GPUs x batch 2 = $STEPS steps. Train with:"
cat <<EOF
torchrun --standalone --nproc_per_node $NGPU scripts/train.py --data $DATA/latents_full --val-data $DATA/latents \\
  --text-emb $DATA/text_emb.pt --init runs/v2/model_bf16_step8000.pt --batch-size 2 --grad-ckpt --lr-new 3e-5 \\
  --steps $STEPS --val-every 500 --val-batches 16 --save-every 1000 --num-workers 6 --out runs/v3 --wandb-project ego-wm
EOF
