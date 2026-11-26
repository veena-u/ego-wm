#!/usr/bin/env bash
# Run on the Mac: copy the code (not data, envs or runs) to the vast.ai machine.
#   bash scripts/push_to_vast.sh "ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080"
# The SSH command can be pasted exactly as vast.ai shows it (extra flags like -L are ignored).
# Pull results back (resumable, checksum-verified): scripts/pull_from_vast.sh
set -euo pipefail
SSH="${1:?usage: push_to_vast.sh \"ssh -p PORT root@HOST [...]\"}"
read -r PORT HOST < <(python3 - "$SSH" <<'EOF'
import sys, shlex
a = shlex.split(sys.argv[1]); port, host = "22", None
for i, t in enumerate(a):
    if t == "-p": port = a[i + 1]
    elif "@" in t and not t.startswith("-"): host = t
if host is None: sys.exit("no user@host in the ssh command")
print(port, host)
EOF
)
# macOS ships openrsync: stick to options it supports (-a -z --delete --exclude -e).
rsync -az --delete -e "ssh -p $PORT" \
  --exclude .venv --exclude runs --exclude '__pycache__' --exclude .git --exclude .DS_Store \
  --exclude 'data/latents_*' --exclude 'data/vae_compare*' --exclude 'data/text_emb_smoke.pt' --exclude 'data/handpose_smoke' \
  --exclude data/index.csv --exclude data/summary.json --exclude data/action_stats.npz \
  ./ "$HOST:/workspace/ego-wm/"
echo "pushed to $HOST:$PORT"
