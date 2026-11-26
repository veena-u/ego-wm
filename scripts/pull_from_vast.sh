#!/usr/bin/env bash
# Run on the Mac: copy one (large) file back from the vast.ai machine, resuming after dropped
# connections and verifying the SHA-256 at the end.
#   bash scripts/pull_from_vast.sh "ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080" \
#        /workspace/ego-wm/runs/v2/model_bf16.pt runs/v2/model_bf16.pt
# Why not rsync: macOS ships openrsync, which lacks --append-verify, and a plain re-run of rsync
# restarts big files from scratch on these flaky links.
set -euo pipefail
SSH="${1:?usage: pull_from_vast.sh \"ssh ...\" REMOTE_PATH LOCAL_PATH}"; REMOTE="${2:?}"; LOCAL="${3:?}"
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
R() { ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$HOST" "$@" 2>/dev/null; }
TOTAL=$(R "stat -c %s '$REMOTE'") || { echo "cannot reach $HOST or no such file"; exit 1; }
mkdir -p "$(dirname "$LOCAL")"; touch "$LOCAL"
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
for i in $(seq 1 50); do
  S=$(size "$LOCAL")
  [ "$S" -ge "$TOTAL" ] && break
  R "tail -c +$((S + 1)) '$REMOTE'" >> "$LOCAL" || true
  echo "  $(size "$LOCAL") / $TOTAL bytes"
done
[ "$(size "$LOCAL")" -eq "$TOTAL" ] || { echo "incomplete after retries (or local file larger than remote)"; exit 1; }
REM=$(R "sha256sum '$REMOTE'" | cut -c1-64)
LOC=$(shasum -a 256 "$LOCAL" | cut -c1-64)
[ "$REM" = "$LOC" ] && echo "OK  $LOCAL  sha256 $LOC" || { echo "CHECKSUM MISMATCH remote $REM local $LOC"; exit 1; }
