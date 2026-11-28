#!/usr/bin/env bash
# Training machine set-up; idempotent, runs on every container start (from scripts/trainer_onstart.sh).
#  1. credentials from the instance environment -> /root/.config/ego-wm/r2.env, ~/.netrc (W&B)
#  2. venv, start weights (runs/v2/model_bf16_step8000.pt) and handpose.pt from r2:<bucket>/bootstrap/
#  3. remote_job.sh (waits for every done/<part>, downloads the data, trains with W&B logging, resumes)
#  4. sidecar: every 30 min copy runs/v3/weights/*.pt and logs to r2:<bucket>/runs/v3/ and write
#     trainer/<id>/status; after MAX_TRAIN_HOURS of training, wait for the next weights save, copy it,
#     and STOP the machine (GPU billing stops; disk kept so training can be resumed or cooled down).
#     The limit applies once: starting the machine again continues without it.
# TEST=1: only steps 1-2 checks, into a scratch HOME, no job (for trying the bootstrap on another machine).
set -uo pipefail
cd /workspace/ego-wm
log() { echo "[$(date -u +%d-%H:%M)] bootstrap: $*"; }
for v in R2_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET WANDB_API_KEY MAX_TRAIN_HOURS CONTAINER_ID CONTAINER_API_KEY; do
  [ -n "${!v:-}" ] || export "$v=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$v=//p")"
done
CFG=/root/.config/ego-wm; NETRC=/root/.netrc
[ "${TEST:-0}" = 1 ] && { CFG=/tmp/bt_home/.config/ego-wm; NETRC=/tmp/bt_home/.netrc; }
mkdir -p "$CFG" && chmod 700 "$CFG"
( umask 077
  printf 'R2_ENDPOINT=%s\nR2_ACCESS_KEY_ID=%s\nR2_SECRET_ACCESS_KEY=%s\nR2_BUCKET=%s\n' \
    "$R2_ENDPOINT" "$R2_ACCESS_KEY_ID" "$R2_SECRET_ACCESS_KEY" "$R2_BUCKET" > "$CFG/r2.env"
  for h in api.wandb.ai api.forge.coreweave.com; do  # W&B (now CoreWeave); train.py also gets WANDB_API_KEY
    grep -q "$h" "$NETRC" 2>/dev/null || printf 'machine %s\n  login user\n  password %s\n' "$h" "$WANDB_API_KEY" >> "$NETRC"
  done )
. scripts/r2.sh
r2_ok || { log "R2 credentials do not work"; exit 1; }
ID=${CONTAINER_ID:-unknown}
status() { echo "$(date -u) $*" | r2 rcat "r2:$R2_BUCKET/trainer/$ID/status"; }

if [ "${TEST:-0}" != 1 ]; then
  if [ ! -x .venv/bin/python ]; then
    log "building venv"
    (command -v uv >/dev/null || pip install -q uv) && uv venv --python 3.11 .venv \
      && uv pip install --python .venv/bin/python -r requirements.txt > /workspace/logs/venv.log 2>&1 \
      || { log "venv failed"; status "venv failed"; exit 1; }
  fi
  .venv/bin/python -c "import torch; assert torch.cuda.is_available()" || { log "torch/CUDA broken"; status "torch/CUDA broken"; exit 1; }
fi
mkdir -p runs/v2 runs/handpose
for f in "runs/v2/model_bf16_step8000.pt" "runs/handpose/handpose.pt"; do
  [ -f "$f" ] || r2 copy "r2:$R2_BUCKET/bootstrap/$(basename "$f")" "$(dirname "$f")/" || { log "fetch $f failed"; exit 1; }
done
want=$(r2 cat "r2:$R2_BUCKET/bootstrap/model_bf16_step8000.pt.sha256" 2>/dev/null | cut -c1-64)
have=$(sha256sum runs/v2/model_bf16_step8000.pt | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$have" ] || { log "start weights checksum mismatch"; status "weights checksum mismatch"; exit 1; }
log "weights ok"
[ "${TEST:-0}" = 1 ] && { log "TEST: set-up checks passed"; exit 0; }

echo r2 > /workspace/data_from
pgrep -f "[b]ash scripts/remote_job.sh" >/dev/null || \
  (setsid nohup bash scripts/remote_job.sh >> /workspace/logs/job.log 2>&1 < /dev/null &)
r2 lsf "r2:$R2_BUCKET/trainer/$ID/started" 2>/dev/null | grep -q . || \
  { echo "$(date -u) $(nvidia-smi -L | head -1)" | r2 rcat "r2:$R2_BUCKET/trainer/$ID/started"; }
log "remote_job running"; status "bootstrapped; remote_job running"

pgrep -f "[t]rainer_sidecar" >/dev/null && exit 0
exec -a trainer_sidecar bash -c '
  . scripts/r2.sh; ID='"$ID"'; MAX=${MAX_TRAIN_HOURS:-72}
  status() { echo "$(date -u) $*" | r2 rcat "r2:$R2_BUCKET/trainer/$ID/status"; }
  limit_s=$(awk -v h="$MAX" "BEGIN {print int(h * 3600)}")
  n=0
  idle=0
  while sleep 60; do
    pgrep -f "[s]cripts/train.py" >/dev/null && echo $(( $(cat /workspace/train_seconds 2>/dev/null || echo 0) + 60 )) > /workspace/train_seconds
    n=$((n + 1))
    # crash guard: neither training nor the data job running for 30 min -> logs to R2, stop the machine
    if pgrep -f "[s]cripts/train.py" >/dev/null || pgrep -f "[b]ash scripts/remote_job.sh" >/dev/null; then idle=0; else idle=$((idle + 1)); fi
    if [ "$idle" -ge 30 ]; then
      r2 copy /workspace/logs "r2:$R2_BUCKET/trainer/$ID/logs" --include "*.log" -q
      status "nothing running for 30 min (crash? see trainer/$ID/logs); stopping the machine"
      /opt/instance-tools/bin/vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" < /dev/null
      idle=0
    fi
    if [ $((n % 30)) = 0 ]; then
      [ -d runs/v3/weights ] && r2 copy runs/v3/weights "r2:$R2_BUCKET/runs/v3/weights" -q
      r2 copy /workspace/logs "r2:$R2_BUCKET/trainer/$ID/logs" --include "*.log" -q
      step=$(grep -o "\"step\": [0-9]*" /workspace/logs/train.log 2>/dev/null | tail -1 | grep -o "[0-9]*$")
      status "training: step ${step:-?}, $(( $(cat /workspace/train_seconds 2>/dev/null || echo 0) / 3600 )) h trained"
    fi
    if [ ! -f /workspace/limit_done ] && [ "$(cat /workspace/train_seconds 2>/dev/null || echo 0)" -ge "$limit_s" ]; then
      status "time limit ${MAX} h reached; waiting for the next weights save"
      before=$(ls runs/v3/weights 2>/dev/null | wc -l)
      for i in $(seq 1 240); do [ "$(ls runs/v3/weights 2>/dev/null | wc -l)" -gt "$before" ] && break; sleep 60; done
      sleep 120
      r2 copy runs/v3/weights "r2:$R2_BUCKET/runs/v3/weights" -q
      r2 copy /workspace/logs "r2:$R2_BUCKET/trainer/$ID/logs" --include "*.log" -q
      touch /workspace/limit_done
      status "time limit reached: weights in R2; stopping the machine (start it again to continue)"
      /opt/instance-tools/bin/vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" < /dev/null
    fi
  done' >> /workspace/logs/sidecar.log 2>&1
