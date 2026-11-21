#!/usr/bin/env bash
# Run on the Mac: copy the code (not data, envs or runs) to the vast.ai machine.
#   bash scripts/push_to_vast.sh "ssh -p 12345 root@1.2.3.4"
# Pull results back:
#   rsync -avz -e "ssh -p 12345" root@1.2.3.4:/workspace/ego-wm/runs/ runs/
set -euo pipefail
SSH="${1:?usage: push_to_vast.sh \"ssh -p PORT root@HOST\"}"
HOST="${SSH##* }"
PORT_ARGS="${SSH% *}"; PORT_ARGS="${PORT_ARGS#ssh}"
rsync -avz --delete -e "ssh $PORT_ARGS" \
  --exclude .venv --exclude runs --exclude '__pycache__' \
  --exclude 'data/latents_*' --exclude 'data/vae_compare*' --exclude 'data/text_emb_smoke.pt' \
  --exclude data/index.csv --exclude data/summary.json --exclude data/action_stats.npz \
  ./ "$HOST:/workspace/ego-wm/"
