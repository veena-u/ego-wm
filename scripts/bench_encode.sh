#!/usr/bin/env bash
# Encoding throughput on this machine: N encoder processes per GPU on the test split.
# Prints steady-state clips/s (from preprocess.py's own progress lines, so model loading is excluded).
#   bash scripts/bench_encode.sh "2 3"      # try 2 and then 3 processes per GPU
set -uo pipefail
cd /workspace/ego-wm
D=/workspace/data; PY=.venv/bin/python
NCPU=$(awk '$1 != "max" {print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); NCPU=${NCPU:-$(nproc)}
NGPU=$(nvidia-smi -L | wc -l)
[ -f $D/index/test/index.csv ] || $PY scripts/build_index.py $D/egodex/test --out $D/index/test --workers $NCPU --no-stats
for P in ${1:-2 3}; do
  N=$((P * NGPU)); rm -rf $D/bench_$P; mkdir -p $D/bench_$P
  for i in $(seq 1 $N); do
    CUDA_VISIBLE_DEVICES=$(( (i - 1) % NGPU )) OMP_NUM_THREADS=$((NCPU / N)) $PY scripts/preprocess.py --vae wan2.1 \
      --index $D/index/test/index.csv --out $D/bench_$P --shard $i/$N --limit ${LIMIT:-60} > /workspace/logs/bench_${P}_$i.log 2>&1 &
  done
  wait
  tot=$(for f in /workspace/logs/bench_${P}_*.log; do grep 'clips/s' "$f" | tail -1; done | awk '{s+=$(NF-1)} END {print s}')
  echo "procs/GPU $P: $tot clips/s total, $(awk "BEGIN {print $tot / $NGPU}") per GPU (CPUs $NCPU)"
  grep -h FAILED /workspace/logs/bench_${P}_*.log | head -2
done
