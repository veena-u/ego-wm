#!/usr/bin/env bash
# Real-model checks for the demo + distillation code, sized for an idle window (<= ~25 min).
cd /workspace/ego-wm
D=/workspace/data; L=/workspace/logs; PY=.venv/bin/python
T=${T:-22m}
( CUDA_VISIBLE_DEVICES=0 timeout $T $PY scripts/encode_text.py --index $D/index/test/index.csv --out $D/text_emb_test.pt > $L/chk_text.log 2>&1
  CUDA_VISIBLE_DEVICES=0 timeout $T $PY scripts/bench_live.py --ckpt runs/v2/model_bf16_step8000.pt --data $D/latents \
     --text-emb $D/text_emb_test.pt --steps-list 10 30 --n 30 --out runs/bench_live_v2 > $L/chk_bench.log 2>&1 ) &
( C="--teacher runs/v2/model_bf16_step8000.pt --data $D/latents_full --grad-ckpt --num-workers 2 --log-every 1 --save-every 1000"
  CUDA_VISIBLE_DEVICES=1,2 timeout $T .venv/bin/torchrun --standalone --nproc_per_node 2 scripts/distill.py --stage ode $C --iters 3 --out runs/chk_ode > $L/chk_ode.log 2>&1
  CUDA_VISIBLE_DEVICES=1,2 timeout $T .venv/bin/torchrun --standalone --nproc_per_node 2 scripts/distill.py --stage dmd $C --iters 6 --fake-ratio 2 --out runs/chk_dmd > $L/chk_dmd.log 2>&1 ) &
( while sleep 20; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $L/chk_mem.log; done ) &
MEM=$!
wait %1 %2
kill $MEM
echo "checks done"
