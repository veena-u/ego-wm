#!/usr/bin/env bash
# vast.ai start hook for ENCODER machines (training machines use scripts/autostart.sh). Install:
#   cp scripts/encoder_autostart.sh /workspace/autostart.sh
#   sed -i '/autostart.sh/d; /entrypoint.sh/i bash /workspace/autostart.sh' /root/onstart.sh
# (the line must come BEFORE entrypoint.sh, which never returns). Starts the work-queue worker
# (scripts/encode_worker.sh) unless it is already running.
mkdir -p /workspace/logs
echo "[$(date -u +%d-%H:%M)] container start" >> /workspace/logs/restarts.log
pgrep -f "[e]ncode_worker.sh" >/dev/null || (cd /workspace/ego-wm && setsid nohup bash scripts/encode_worker.sh > /dev/null 2>&1 < /dev/null &)
