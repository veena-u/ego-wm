#!/usr/bin/env bash
# Wait for the running full_pipeline.sh to end, then run it again with the current default parts
# (finished parts are skipped). Used to add `extra` without disturbing a pipeline mid-run.
while pgrep -f "[b]ash scripts/full_pipeline.sh" >/dev/null; do sleep 60; done
grep -q "^\[.*\] ready:" /workspace/logs/pipeline.log || { echo "first pipeline did not finish cleanly; not continuing"; exit 1; }
exec bash scripts/full_pipeline.sh
