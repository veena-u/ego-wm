#!/usr/bin/env bash
# Encoder machine main loop: a pull-based work queue in R2. Every encoder machine runs this:
#   cd /workspace/ego-wm && setsid nohup bash scripts/encode_worker.sh [--claim UNIT...] > /dev/null 2>&1 < /dev/null &
# (scripts/encoder_autostart.sh restarts it after a container restart.)
#
# Work comes in units: a whole part ("part1") or a shard of one ("part5.3of8", every 8th episode),
# listed in PLAN. A machine claims a unit by writing claim/<unit> = "<machine> <unix time>" in R2 and
# refreshes it every 10 min while working; a claim not refreshed for STALE seconds (machine gone) is
# taken over. Next unit, in order of preference:
#   1. one this machine already owns (resuming after a restart),
#   2. a free shard of a part whose raw video is already on this disk (or half-downloaded),
#   3. a free unit of the part with the most free units (machines spread out over parts, then help
#      each other finish the last shards).
# When every unit is done or owned by another live machine, this machine DESTROYS ITSELF (billing,
# disk included, stops; everything it made is already in R2 and checked). A unit that fails 3 times
# is released for the other machines and this machine STOPS itself (GPU billing stops, disk kept for
# debugging; starting it again retries).
# When all shards of a part are done, done/<part> is written, which is what scripts/r2_fetch.sh waits for.
set -uo pipefail
cd /workspace/ego-wm
DATA=${DATA:-/workspace/data}
PLAN=${PLAN:-"test part1 part3 extra:4 part2:8 part4:8 part5:8"}
STALE=${STALE:-7200}
DESTROY_WHEN_DONE=${DESTROY_WHEN_DONE:-1}
LOG=/workspace/logs/pipeline.log
mkdir -p /workspace/logs
log() { echo "[$(date -u +%d-%H:%M)] worker: $*" >> "$LOG"; }
. scripts/r2.sh
r2_ok || { log "no R2 credentials"; exit 1; }
[ -s /workspace/machine_id ] || head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n' > /workspace/machine_id
ME="$(hostname)-$(cat /workspace/machine_id)"
# vast.ai sets these in the container's environment; a process not started from a login shell may lack them
for v in CONTAINER_ID CONTAINER_API_KEY; do
  [ -n "${!v:-}" ] || export "$v=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$v=//p")"
done
VAST=/opt/instance-tools/bin/vastai; [ -x "$VAST" ] || VAST=$(command -v vastai || echo vastai)

units() {
  local e p K k
  for e in $PLAN; do
    p=${e%%:*}; K=1; [ "$e" = "$p" ] || K=${e#*:}
    if [ "$K" = 1 ]; then echo "$p"; else for k in $(seq 1 "$K"); do echo "$p.${k}of$K"; done; fi
  done
}
part_of() { echo "${1%%.*}"; }
claim_of() { r2 cat "r2:$R2_BUCKET/claim/$1" 2>/dev/null | head -1; }
write_claim() { echo "$ME $(date +%s)" | r2 rcat "r2:$R2_BUCKET/claim/$1"; }

scan() {  # MINE: units I own, not done; FREE: not done and unclaimed or stale; BUSY: owned by live others
  MINE=(); FREE=(); BUSY=()
  local done_list now u c o t
  done_list=$(r2 lsf "r2:$R2_BUCKET/done/" 2>/dev/null) || return 1
  now=$(date +%s)
  for u in $(units); do
    grep -qxF "$u" <<< "$done_list" && continue
    c=$(claim_of "$u"); o=${c%% *}; t=${c##* }
    [[ "$t" =~ ^[0-9]+$ ]] || t=0
    if [ -z "$c" ]; then FREE+=("$u")
    elif [ "$o" = "$ME" ]; then MINE+=("$u")
    elif [ $(( now - t )) -gt "$STALE" ]; then FREE+=("$u")
    else BUSY+=("$u"); fi
  done
}

pick() {
  local u p best
  [ ${#MINE[@]} -gt 0 ] && { echo "${MINE[0]}"; return 0; }
  [ ${#FREE[@]} -gt 0 ] || return 1
  for u in "${FREE[@]}"; do p=$(part_of "$u")
    [ -f "$DATA/egodex/.unzipped_$p" ] && [ -d "$DATA/egodex/$p" ] && { echo "$u"; return 0; }; done
  for u in "${FREE[@]}"; do p=$(part_of "$u")
    [ -e "$DATA/egodex/$p.zip" ] && { echo "$u"; return 0; }; done
  best=$(printf '%s\n' "${FREE[@]}" | awk -F. '{n[$1]++; if (!($1 in o)) o[$1]=NR} END {for (p in n) print n[p], -o[p], p}' \
         | sort -k1,1nr -k2,2nr | head -1 | awk '{print $3}')
  for u in "${FREE[@]}"; do [ "$(part_of "$u")" = "$best" ] && { echo "$u"; return 0; }; done
  return 1
}

claim() {  # last writer wins; everyone reads back after the writes have landed
  write_claim "$1" || return 1
  sleep 20
  [ "$(claim_of "$1" | awk '{print $1}')" = "$ME" ]
}

run_unit() {
  local u=$1 p s SH HB rc
  p=$(part_of "$u")
  if [ "$u" = "$p" ]; then SH=1/1; else s=${u#*.}; SH="${s%of*}/${s#*of}"; fi
  ( while sleep 600; do write_claim "$u" > /dev/null 2>&1; done ) & HB=$!
  PARTS="$p" SHARD="$SH" R2=1 SKIP_FINAL=1 KEEP_RAW=1 PROCS_PER_GPU=${PROCS_PER_GPU:-2} \
    bash scripts/full_pipeline.sh >> "$LOG" 2>&1; rc=$?
  kill "$HB" 2>/dev/null
  return $rc
}

after_success() {
  local u=$1 p K k all
  p=$(part_of "$u")
  if [ "$u" != "$p" ]; then  # a shard: if it was the last of its part, mark the part done
    K=${u##*of}; all=1
    for k in $(seq 1 "$K"); do [ -n "$(r2 lsf "r2:$R2_BUCKET/done/$p.${k}of$K" 2>/dev/null)" ] || { all=0; break; }; done
    [ "$all" = 1 ] && { echo "$(date -u) all $K shards" | r2 rcat "r2:$R2_BUCKET/done/$p"; log "$p: all $K shards done -> done/$p"; }
  fi
  r2 deletefile "r2:$R2_BUCKET/claim/$u" 2>/dev/null
  rm -f "/workspace/fails_$u"
  scan || return 0
  if ! printf '%s\n' "${FREE[@]}" "${MINE[@]}" | grep -q "^$p\(\.\|$\)"; then  # nothing more of this part for me
    log "$p: no free units left here; deleting raw video"
    rm -rf "$DATA/egodex/$p" "$DATA/egodex/$p.zip" "$DATA/egodex/$p.zip.aria2" "$DATA/egodex/.unzipped_$p"
  fi
}

launch_trainer() {  # the machine that finishes the last unit rents the training machine (once: lock in R2)
  local lock out
  [ -s /root/.config/ego-wm/vast.env ] || { log "no /root/.config/ego-wm/vast.env: not launching the trainer"; return 0; }
  lock=$(r2 cat "r2:$R2_BUCKET/trainer/lock" 2>/dev/null)
  if [ -n "$lock" ] && [ "${lock%% *}" != "$ME" ]; then log "trainer launch is handled by ${lock%% *}"; return 0; fi
  if [ -z "$lock" ]; then
    write_claim trainer/lock; sleep 20
    [ "$(r2 cat "r2:$R2_BUCKET/trainer/lock" | awk '{print $1}')" = "$ME" ] || { log "another machine launches the trainer"; return 0; }
  fi
  [ -n "$(r2 lsf "r2:$R2_BUCKET/trainer/launched" 2>/dev/null)" ] && { log "trainer already launched"; return 0; }
  for attempt in $(seq 1 48); do  # up to a day: offers within the caps come and go
    log "launching the training machine (attempt $attempt)"
    out=$(python3 scripts/launch_trainer.py 2>&1); echo "$out" | sed 's/^/  launch: /' >> "$LOG"
    if grep -q "^LAUNCHED " <<< "$out"; then
      grep "^LAUNCHED " <<< "$out" | tail -1 | r2 rcat "r2:$R2_BUCKET/trainer/launched"
      log "trainer launched: $(grep "^LAUNCHED " <<< "$out" | tail -1)"; return 0
    fi
    grep -q "bootstrap never reported" <<< "$out" && { log "trainer bootstrap failed; giving up (machine stopped for debugging)"; return 1; }
    sleep 1800
  done
  log "could not rent a training machine within the caps for a day; giving up"; return 1
}

finish() {
  local left
  left=$(find "$DATA/latents" "$DATA/latents_full" -name "*.pt" 2>/dev/null | wc -l)
  if [ "$left" != 0 ]; then log "no work left but $left .pt files not uploaded; NOT destroying"; return 1; fi
  echo "$(date -u) $ME finished; busy elsewhere: ${BUSY[*]:-none}" | r2 rcat "r2:$R2_BUCKET/finished/$ME"
  log "no work left (others still on: ${BUSY[*]:-none}); all my output is in R2"
  if [ ${#BUSY[@]} = 0 ] && [ "${AUTO_TRAIN:-1}" = 1 ]; then
    launch_trainer || echo "$(date -u) $ME: trainer launch failed; see its pipeline.log" | r2 rcat "r2:$R2_BUCKET/trainer/failed"
  fi
  [ "$DESTROY_WHEN_DONE" = 1 ] || { log "DESTROY_WHEN_DONE=0: leaving the machine running"; return 0; }
  [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] || { log "no CONTAINER_ID/API key; cannot destroy"; return 1; }
  log "destroying this instance ($CONTAINER_ID) so billing stops"
  "$VAST" destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" < /dev/null >> "$LOG" 2>&1
  sleep 300  # still alive: destroy didn't work, stop instead (GPU billing stops)
  log "destroy did not take effect; stopping instead"
  "$VAST" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" < /dev/null >> "$LOG" 2>&1
}

give_up() {
  local u=$1
  r2 deletefile "r2:$R2_BUCKET/claim/$u" 2>/dev/null
  echo "$(date -u) $ME gave up on $u" | r2 rcat "r2:$R2_BUCKET/failed/$u.$ME"
  log "$u failed 3 times; released it for other machines and stopping this machine (start it again to retry)"
  rm -f /workspace/fails_*
  [ -n "${CONTAINER_ID:-}" ] && "$VAST" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" < /dev/null >> "$LOG" 2>&1
  exit 1
}

log "start as $ME; plan: $PLAN"
if [ "${1:-}" = --claim ]; then shift; for u in "$@"; do write_claim "$u" && log "pre-claimed $u"; done; fi
while true; do
  scan || { log "R2 unreachable; retrying in 5 min"; sleep 300; continue; }
  if ! u=$(pick); then finish; exit $?; fi
  if ! printf '%s\n' "${MINE[@]}" | grep -qxF "$u"; then
    claim "$u" || { log "lost the claim race for $u"; continue; }
    log "claimed $u"
  else
    log "resuming $u"
  fi
  if run_unit "$u"; then
    log "$u done"
    after_success "$u"
  else
    n=$(( $(cat "/workspace/fails_$u" 2>/dev/null || echo 0) + 1 )); echo "$n" > "/workspace/fails_$u"
    [ "$n" -ge 3 ] && give_up "$u"
    log "$u failed (attempt $n of 3); retrying in 5 min"
    sleep 300
  fi
done
