#!/bin/bash
# onstart of the auto-launched training machine (scripts/launch_trainer.py sets it; vast runs it on
# every container start). R2_* / WANDB_API_KEY / MAX_TRAIN_HOURS come from the instance environment.
# First start: fetch the code bundle from R2. Every start: run the idempotent bootstrap.
mkdir -p /workspace/logs
{
  echo "[$(date -u +%d-%H:%M)] container start"
  if [ ! -f /workspace/ego-wm/scripts/trainer_bootstrap.sh ]; then
    export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT" \
      RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
      RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
    v=v1.68.2; d=$(mktemp -d)
    (cd "$d" && curl -fsSLO "https://downloads.rclone.org/$v/rclone-$v-linux-amd64.zip" \
      && python3 -m zipfile -e "rclone-$v-linux-amd64.zip" . \
      && install -m 755 "rclone-$v-linux-amd64/rclone" /usr/local/bin/rclone-ego)
    rclone-ego --config /dev/null copy "r2:$R2_BUCKET/bootstrap/ego-wm.tar.gz" /workspace/ \
      && mkdir -p /workspace/ego-wm && tar xzf /workspace/ego-wm.tar.gz -C /workspace/ego-wm \
      || echo "code bundle download FAILED"
  fi
  (cd /workspace/ego-wm && setsid nohup bash scripts/trainer_bootstrap.sh >> /workspace/logs/bootstrap.log 2>&1 < /dev/null &)
} >> /workspace/logs/bootstrap.log 2>&1
entrypoint.sh
