#!/usr/bin/env bash
# Run on every container start: resume the job where it left off. Install on a new machine:
#   cp scripts/autostart.sh /workspace/autostart.sh && echo "bash /workspace/autostart.sh" >> /root/onstart.sh
cd /workspace/ego-wm
echo "[$(date -u +%d-%H:%M)] container start" >> /workspace/logs/restarts.log
pgrep -f "[b]ash scripts/remote_job.sh" >/dev/null || \
  setsid nohup bash scripts/remote_job.sh >> /workspace/logs/job.log 2>&1 < /dev/null &
