# GPU runbook (vast.ai)

## What to rent
| | Minimum | Why |
|---|---|---|
| GPU | **1× A100 80GB or H100 80GB** | Full fine-tune of Wan 1.3B ≈ 25–35 GB + activations. 24–32 GB cards do not fit |
| CPU | **≥ 16 vCPU (32 better)** | Video decoding for latents and frame extraction is CPU-bound |
| RAM | ≥ 64 GB | Data loader workers + VAE |
| Disk | **≥ 300 GB** (test split phase) | Raw 17 GB, frames ~6 GB, latents ~20 GB, each training checkpoint ~17 GB (weights + optimizer), model weights |
| Network | ≥ 1 Gbps | EgoDex comes straight from Apple's CDN |
| Other | On-demand (not interruptible), reliability ≥ 99%, recent PyTorch/CUDA 12 image | |

For the full dataset later (1.64 TB raw), you'll need ≥ 4 TB of disk and preferably several GPUs.

## Order of work
```bash
# Mac
bash scripts/push_to_vast.sh "ssh -p PORT root@HOST"
# GPU machine
cd /workspace/ego-wm && bash scripts/vast_setup.sh test
D=/workspace/data; PY=.venv/bin/python

# 1. VAE check: does Wan 2.1 keep the hands? (~15 min)
$PY scripts/vae_compare.py --clips 12 --device cuda

# 2. Hand pose estimator: the main evaluation instrument (~30-45 min)
$PY scripts/extract_frames.py --out $D/handpose_frames --every 6 --workers $(nproc)
$PY scripts/train_handpose.py --frames $D/handpose_frames --epochs 6 --out runs/handpose
#    Check final.json: recall high, erased_present low (<5%), and it should beat MediaPipe.

# 3. Text embeddings + latents (run the latents in parallel shards)
$PY scripts/encode_text.py --out $D/text_emb.pt
for i in 1 2 3 4; do $PY scripts/preprocess.py --vae wan2.1 --out $D/latents --shard $i/4 > prep_$i.log 2>&1 & done; wait

# 4. World model: short run first. action_gain should become > 0
$PY scripts/train.py --data $D/latents --text-emb $D/text_emb.pt --batch-size 2 --grad-ckpt \
    --steps 3000 --val-every 250 --out runs/v1

# 5. Evaluate
$PY scripts/evaluate.py --ckpt runs/v1/last.pt --data $D/latents --text-emb $D/text_emb.pt \
    --handpose runs/handpose/handpose.pt --clips 32 --out runs/v1/eval

# Mac: pull the results
rsync -avz -e "ssh -p PORT" root@HOST:/workspace/ego-wm/runs/ runs/
```
Steps 2 and 3 are independent: the frame extraction (CPU) can run while the VAE check uses the GPU.
