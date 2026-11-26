# GPU runbook (vast.ai)

## What to rent
| | Minimum | Why |
|---|---|---|
| GPU | **1× A100 80GB or H100 80GB** | Full fine-tune of Wan 1.3B ≈ 25–35 GB + activations. 24–32 GB cards do not fit |
| CPU | **≥ 16 vCPU (32 better)** | Video decoding for latents and frame extraction is CPU-bound |
| RAM | ≥ 64 GB | Data loader workers + VAE |
| Disk | **≥ 300 GB** (test split phase). Set "Disk Space to Allocate" BEFORE renting; it can't be enlarged later | Raw 17 GB, frames ~6 GB, latents ~20 GB, each training checkpoint ~17 GB (weights + optimizer), model weights |
| Network | ≥ 1 Gbps | EgoDex comes straight from Apple's CDN |
| Other | On-demand (not interruptible), reliability ≥ 99%, recent PyTorch/CUDA 12 image | |

For the full dataset later (1.64 TB raw), you'll need ≥ 4 TB of disk and preferably several GPUs.

## Order of work
```bash
# Mac
bash scripts/push_to_vast.sh "ssh -p PORT root@HOST -L 8080:localhost:8080"   # paste vast's command as-is
# GPU machine
cd /workspace/ego-wm && bash scripts/vast_setup.sh test
D=/workspace/data; PY=.venv/bin/python

# 1. VAE check: does Wan 2.1 keep the hands? (~15 min)
$PY scripts/vae_compare.py --clips 12 --device cuda

# 2. Hand pose estimator: the main evaluation instrument (~30-45 min)
$PY scripts/extract_frames.py --out $D/handpose_frames --every 6 --workers $(nproc)
$PY scripts/train_handpose.py --frames $D/handpose_frames --epochs 6 --out runs/handpose --wandb-project ego-wm
#    Check final.json: recall high, erased_present low (<5%), and it should beat MediaPipe.

# 3. Text embeddings + latents (parallel shards, ~1.5 h for the test split; give it the GPU alone)
$PY scripts/encode_text.py --out $D/text_emb.pt
for i in 1 2 3 4; do OMP_NUM_THREADS=6 $PY scripts/preprocess.py --vae wan2.1 --out $D/latents --shard $i/4 > prep_$i.log 2>&1 & done; wait

# 4. World model. --lr-new 3e-5, NOT 1e-4: at 1e-4 the hand pathway destabilised after warmup
#    (v1: loss with hands 0.127 vs 0.092 without at step 1000). --grad-ckpt is required: without it
#    batch 2 OOMs on 80 GB. ~8 s/step on an A100. Watch hand_gain (hand-region loss, shuffled - true).
$PY scripts/train.py --data $D/latents --text-emb $D/text_emb.pt --batch-size 2 --grad-ckpt --lr-new 3e-5 \
    --steps 8000 --val-every 250 --val-batches 16 --out runs/v2 --wandb-project ego-wm
#    Continue a finished run: same command with a larger --steps and --resume.

# 5. Evaluate (~5.5 min per clip on an A100, so ~3 h for 32)
$PY scripts/evaluate.py --ckpt runs/v2/last.pt --data $D/latents --text-emb $D/text_emb.pt \
    --handpose runs/handpose/handpose.pt --clips 32 --out runs/v2/eval --wandb-project ego-wm
```
Steps 2 and 3 are independent: the frame extraction (CPU) can run while the VAE check uses the GPU.

## All of EgoDex (multi-GPU)
Rent: **8x A100/H100 80GB, >= 3 TB disk, >= 64 CPU cores** (8 encoders decode 1080p video in parallel),
>= 1 Gbps. Then:
```bash
# Mac: code, plus the step-8000 weights the big run starts from (2.8 GB)
bash scripts/push_to_vast.sh "ssh -p PORT root@HOST"
scp -P PORT runs/v2/model_bf16_step8000.pt root@HOST:/workspace/ego-wm/runs/v2/
# GPU machine: env (no data), then the pipeline (~1-2 days: download, encode, delete, per part)
cd /workspace/ego-wm && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# everything (pipeline, then training; restart-safe) - and make container restarts resume it:
setsid nohup bash scripts/remote_job.sh >> /workspace/logs/job.log 2>&1 < /dev/null & disown
# /workspace/autostart.sh runs remote_job.sh if it isn't running; hook: `bash /workspace/autostart.sh`
# appended to /root/onstart.sh (vast runs it on every container start)
```
- Training clips: at most 2 per recording (`MAX_CLIPS`, picked at random across the recording);
  recordings of one take look alike, so more recordings beat more clips per recording.
- Training: constant learning rate after warmup, open-ended steps (`$DATA/train_steps`, default
  200000). To finish: stop it, then `--resume --steps <current + N> --cooldown N` (the rate decays
  to 0 over the last N steps; ~10% of the run). Every save also writes `runs/v3/weights/stepN.pt`
  (2.8 GB bf16) - pull those to the Mac; `last.pt` (17 GB, with optimizer) is overwritten.
- Before a long run: `bash scripts/ddp_smoke.sh` (40 real steps on all GPUs; on 4x A100: 9.5 s/step,
  34 GB per GPU).
- The test split is encoded to `latents/` and used only for validation, so numbers compare with
  earlier runs; parts 1-5 and `extra` (97 h recorded after the paper's split, `extra.zip`, 224 GB)
  go to `latents_full/` for training.
- `data/action_stats.npz` is never rewritten (`build_index.py --no-stats`): the model was trained
  with those normalization stats, and the pipeline checks they are unchanged at the end.
- Restartable: rerun the same command; finished parts and encoded episodes are skipped.

## Evaluation on all GPUs
```bash
bash scripts/eval_parallel.sh runs/v3/weights/stepN.pt runs/v3/eval_N --clips 32 --cfg 3 --wandb-project ego-wm
```
Clips are split across GPUs and merged; identical to a one-process run (same clips, seeds).

## Real-time: distillation
```bash
# 1. warm start: the student learns to jump from the teacher's intermediate states to its result
torchrun --standalone --nproc_per_node 4 scripts/distill.py --stage ode --teacher runs/v3/weights/stepN.pt \
  --data /workspace/data/latents_full --text-emb /workspace/data/text_emb.pt --grad-ckpt --iters 2000 --out runs/d1_ode
# 2. distribution matching on the student's own rollouts (Self Forcing / DMD)
torchrun ... --stage dmd --teacher runs/v3/weights/stepN.pt --init runs/d1_ode/student.pt --grad-ckpt --out runs/d1
```
`student.pt` loads everywhere a checkpoint does (evaluate.py, demo, bench_live.py) and switches
to the 4-step sampler with guidance built in.

## Live demo
```bash
# GPU machine (any checkpoint: trained model or distilled student)
.venv/bin/python scripts/demo_server.py --ckpt runs/d1/student.pt --data /workspace/data/latents \
  --text-emb /workspace/data/text_emb.pt
# Mac: tunnel, then the client (webcam; --list for scenes, --scene N, --view ego for a head camera)
ssh -p PORT root@HOST -L 8765:localhost:8765
.venv/bin/python scripts/demo_client.py --scene 3
```
- `--replay EPISODE` drives it with a recorded EgoDex episode instead of the webcam.
- Speed without a webcam: `scripts/bench_live.py` (ms per step, real-time factor, a video).
- Live tracking = MediaPipe + PnP into EgoDex's camera frame (`egodex/live.py`); measured on
  EgoDex video: 99% correct hand side, 34 px median joint error at 1080p (MediaPipe-vs-EgoDex
  labels alone differ by ~27 px), ~7 cm depth error.

## Running long jobs safely
- Launch detached, so a dropped SSH connection can't kill or hang the job:
  `setsid nohup <cmd> > /workspace/logs/x.log 2>&1 < /dev/null & disown`
- Put multi-step jobs in a script file and launch that, instead of chaining with `&&` and `&`
  inline (a trailing `&` puts the whole chain, including variable assignments, in a subshell).
- Kill with a pattern that can't match your own SSH command line: `pkill -f "[s]cripts/train\.py"`
  only if the same SSH command doesn't also contain `scripts/train.py` (e.g. a relaunch); otherwise
  kill and relaunch in separate SSH calls.
- Two GPU jobs at once roughly halves the world model's speed; run the hand-pose training or
  evaluation alone when possible.
- Training writes `last.pt` every 1000 steps (overwritten). Keep a weights-only bf16 copy of any
  checkpoint you care about and pull it to the Mac.

## Getting results back to the Mac
```bash
# small files (logs, metrics, videos)
scp -P PORT root@HOST:/workspace/ego-wm/runs/v2/eval/metrics.json runs/v2/eval/
# big files: resumable + SHA-256 verified (macOS's rsync can't resume big files)
bash scripts/pull_from_vast.sh "ssh -p PORT root@HOST" /workspace/ego-wm/runs/v2/model_bf16.pt runs/v2/model_bf16.pt
```

## Moving to a new machine
A stopped instance can only restart on its original GPU; if someone else rents that GPU, it can be
stuck for days. Its disk is still readable, so move instead of waiting:
1. Rent a new instance (same requirements as above).
2. In the vast console, on the OLD instance, use the copy button: `/workspace/` -> new instance `/workspace/`.
   That carries code, venv packages, latents, frames, text embeddings and checkpoints (~110 GB).
3. On the new machine, things outside `/workspace` are missing:
   ```bash
   # the venv's Python interpreter (lives in /.uv, not /workspace)
   UV_PYTHON_INSTALL_DIR=/.uv/python_install uv python install 3.11
   ln -sf /.uv/python_install/cpython-3.11-linux-x86_64-gnu/bin/python3.11 /workspace/ego-wm/.venv/bin/python
   # W&B login (~/.netrc) - run `wandb login`, or copy the entry from the Mac's ~/.netrc
   # Hugging Face weights (Wan 2.1) re-download automatically on first use
   ```
4. Check: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`, then delete the old instance.
