#!/usr/bin/env bash
# Cloudflare R2 access for the GPU machines. Source it:  . scripts/r2.sh ; then  r2 ls r2:$R2_BUCKET
# Credentials: /root/.config/ego-wm/r2.env (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
# R2_BUCKET), mode 600, copied from the Mac's ~/.config/ego-wm/r2.env. Never in the repo.
R2_ENV=${R2_ENV:-/root/.config/ego-wm/r2.env}
if [ -f "$R2_ENV" ]; then
  set -a; . "$R2_ENV"; set +a
  export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT" \
    RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
fi
# Our own pinned rclone: images may ship an old one (v1.60 fails R2 uploads with 501 NotImplemented).
RCLONE=/usr/local/bin/rclone-ego
if [ ! -x "$RCLONE" ]; then
  v=v1.68.2; d=$(mktemp -d)
  (cd "$d" && curl -fsSLO "https://downloads.rclone.org/$v/rclone-$v-linux-amd64.zip" && unzip -q "rclone-$v-linux-amd64.zip" \
    && install -m 755 "rclone-$v-linux-amd64/rclone" "$RCLONE") && rm -rf "$d"
fi
r2() { "$RCLONE" --config /dev/null --transfers 16 --checkers 32 --retries 10 --low-level-retries 20 "$@"; }
r2_ok() { [ -n "${R2_BUCKET:-}" ]; }
