"""Fine-tune the action-conditioned world model.

GPU (real run):
  python scripts/train.py --data /data/latents --text-emb /data/text_emb.pt --model wan1.3b \
      --batch-size 4 --grad-ckpt --out runs/v1
Multi-GPU (one process per GPU, data split across them; effective batch = GPUs x --batch-size):
  torchrun --nproc_per_node 8 scripts/train.py --data /data/latents_full --val-data /data/latents ...
Mac smoke test (tiny random model, stand-in latents):
  python scripts/train.py --data data/latents_smoke --model tiny --max-latent-frames 6 \
      --crop 16 16 --steps 50 --batch-size 2 --splits train val_seen val_unseen --out runs/smoke

Logged every --val-every steps:
  val_loss          flow-matching loss on held-out clips (fixed noise, so comparable over time)
  val_loss_shuffled same clips, but each clip gets ANOTHER clip's hand conditioning. This input
                    is realistic (real poses, just wrong for this video), unlike all-zeros.
  action_gain       val_loss_shuffled - val_loss: > 0 means the model uses the specific hand
                    poses it is given, not just their presence.
  val_loss_uncond   conditioning dropped (zeros), which training also sees (--cond-drop) so
                    classifier-free guidance on the hands is possible at sampling time.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex.conditioning import ACTION_DIM, NUM_SPATIAL_CHANNELS  # noqa: E402
from egodex.dataset import LatentClips  # noqa: E402
from egodex.model import ActionWorldModel, flow_loss, hand_region, load_wan, tiny_wan  # noqa: E402
from egodex import wb  # noqa: E402


def build_model(a, latent_channels, fpl, text_dim):
    if a.model == "tiny":
        tr = tiny_wan(latent_channels, text_dim)
    else:
        tr = load_wan(a.wan_repo, dtype=torch.float32)
        assert tr.config.in_channels == latent_channels, "latents were made with a different VAE"
    return ActionWorldModel(tr, NUM_SPATIAL_CHANNELS, ACTION_DIM, fpl, causal=a.causal)


def to_dev(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, batches, a, device, amp):
    model.eval()
    out = {}
    for i, b in enumerate(batches):
        B = b["latents"].shape[0]
        n_ctx = torch.ones(B, dtype=torch.long, device=device)
        # Hand region from the TRUE poses, so every variant is scored on the same pixels.
        region = hand_region(b["spatial"], model.fpl)
        variants = {"val_loss": (b["spatial"], b["actions"]),
                    "val_loss_uncond": (torch.zeros_like(b["spatial"]), torch.zeros_like(b["actions"]))}
        if B > 1:
            variants["val_loss_shuffled"] = (b["spatial"].roll(1, 0), b["actions"].roll(1, 0))
        for k, (sp, ac) in variants.items():
            g = torch.Generator(device=device).manual_seed(1000 + i)  # identical noise for every variant
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                l, _, _, lh = flow_loss(model, b["latents"], sp, ac, b["text"], n_ctx, a.shift, g, region)
            out.setdefault(k, []).append(l.item())
            out.setdefault(k.replace("val_loss", "val_hand_loss"), []).append(lh.item())
        out.setdefault("hand_area", []).append(region.mean().item())
    model.train()
    r = {k: float(np.mean(v)) for k, v in out.items() if v}
    if "val_loss_shuffled" in r:
        r["action_gain"] = r["val_loss_shuffled"] - r["val_loss"]
        r["hand_gain"] = r["val_hand_loss_shuffled"] - r["val_hand_loss"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None, help="latents dir for validation (default: --data)")
    ap.add_argument("--init", default=None, help="start from these model weights (e.g. an earlier run's "
                    "checkpoint); optimizer and step start fresh. --resume takes precedence")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--val-splits", nargs="+", default=["val_seen", "val_unseen"])
    ap.add_argument("--action-stats", default="data/action_stats.npz")
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--model", default="wan1.3b", choices=["wan1.3b", "tiny"])
    ap.add_argument("--wan-repo", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5, help="pretrained weights")
    ap.add_argument("--lr-new", type=float, default=1e-4, help="new hand-conditioning layers")
    ap.add_argument("--cond-drop", type=float, default=0.1,
                    help="drop hand conditioning for this fraction of samples (enables guidance)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--cooldown", type=int, default=0,
                    help="linearly decay the learning rate to 0 over the last N of --steps (warmup-stable-"
                         "decay). Constant until then, so a run can be stopped anywhere: to finish one, "
                         "--resume with --steps = current + N and --cooldown N")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=3.0, help="noise schedule shift (Wan uses ~3 at 480p)")
    ap.add_argument("--n-ctx-max", type=int, default=2, help="clean context latent frames: 1..n")
    ap.add_argument("--max-latent-frames", type=int, default=None)
    ap.add_argument("--crop", type=int, nargs=2, default=None, help="latent crop h w (smoke tests only)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--overfit", type=int, default=None, help="train on the first N clips only")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-weights", action=argparse.BooleanOptionalAction, default=True,
                    help="at every save also write weights/stepN.pt: bf16 weights only (2.8 GB, loads in "
                         "evaluate.py), since last.pt (weights + optimizer, ~17 GB) is overwritten")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb-project", default=None, help="log to Weights & Biases (off if unset)")
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp = world > 1
    if ddp:
        from datetime import timedelta
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            a.device = f"cuda:{local}"
        # 1 h timeout: the main GPU validates and saves alone while the others wait at a barrier.
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=1))
    main_proc = rank == 0
    # Different data order / noise / dropout per GPU (DistributedSampler splits the data itself).
    torch.manual_seed(a.seed + rank)
    random.seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    device = torch.device(a.device)
    amp = device.type == "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def barrier():
        if ddp:
            dist.barrier()

    text_dim = 64 if a.model == "tiny" else 4096
    ds_kw = dict(action_stats=a.action_stats, text_emb=a.text_emb, text_dim=text_dim,
                 text_len=16 if a.model == "tiny" else 512, max_latent_frames=a.max_latent_frames,
                 crop_hw=a.crop)
    # The main process builds/updates manifest.json first; the others then just read it.
    if not main_proc:
        barrier()
    train_ds = LatentClips(a.data, a.splits, **ds_kw)
    if a.overfit:
        train_ds = Subset(train_ds, range(min(a.overfit, len(train_ds))))
    val_ds = LatentClips(a.val_data or a.data, a.val_splits, seed=123, **ds_kw)
    if main_proc:
        barrier()
        print(f"train clips {len(train_ds)}, val clips {len(val_ds)}, GPUs {world}, "
              f"effective batch {world * a.batch_size * a.grad_accum}", flush=True)
    sample = train_ds[0]
    C = sample["latents"].shape[0]
    fpl = sample["frames_per_latent"]

    model = build_model(a, C, fpl, text_dim)
    if a.init and not (a.resume and (out / "last.pt").exists()):
        sd = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        if main_proc:
            print(f"initialized from {a.init}", flush=True)
    model = model.to(device)
    if a.grad_ckpt:
        model.enable_gradient_checkpointing()
    new_params = model.new_parameters()
    new_ids = {id(p) for p in new_params}
    base_params = [p for p in model.parameters() if id(p) not in new_ids]
    opt = torch.optim.AdamW([{"params": base_params, "lr": a.lr}, {"params": new_params, "lr": a.lr_new}],
                            betas=(0.9, 0.99), weight_decay=0.01)
    base_lrs = [a.lr, a.lr_new]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.1f}M, latent C={C}, frames/latent={fpl}, causal={a.causal}", flush=True)

    raw = model  # unwrapped module: for evaluation, saving, attributes
    step = 0
    ckpt = out / "last.pt"
    if a.resume and ckpt.exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
        if main_proc:
            print(f"resumed from step {step}", flush=True)
        # A fresh data order after a restart; otherwise the shuffle replays from its start and the
        # first clips are seen again while later ones never are.
        torch.manual_seed(a.seed + rank + 7919 * step)
    if main_proc:
        (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    if ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=a.seed, drop_last=True) if ddp else None
    epoch = step  # distinct sampler permutation per (re)start; see the resume note above
    if sampler is not None:
        sampler.set_epoch(epoch)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler,
                        drop_last=True, num_workers=a.num_workers, pin_memory=amp,
                        persistent_workers=a.num_workers > 0)
    assert a.batch_size >= 2, "batch size >= 2 needed for the shuffled-conditioning check"
    val_loader = DataLoader(Subset(val_ds, np.random.default_rng(0).permutation(len(val_ds))[
        : a.val_batches * a.batch_size].tolist()), batch_size=a.batch_size, num_workers=0)
    val_batches = [to_dev(b, device) for b in val_loader] if main_proc else []

    log = open(out / "log.jsonl", "a") if main_proc else None
    run = wb.init(a.wandb_project, out, vars(a), "world_model") if main_proc else None
    model.train()
    t0, it = time.time(), iter(loader)
    while step < a.steps:
        lr_scale = min(1.0, (step + 1) / a.warmup)
        if a.cooldown and step >= a.steps - a.cooldown:
            lr_scale *= (a.steps - step) / a.cooldown
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * lr_scale
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                it = iter(loader)
                b = next(it)
            b = to_dev(b, device)
            B, _, Fl = b["latents"].shape[:3]
            n_ctx = torch.randint(1, min(a.n_ctx_max, Fl - 1) + 1, (B,), device=device)
            if a.cond_drop > 0:
                keep = (torch.rand(B, device=device) >= a.cond_drop).float()
                b["spatial"] = b["spatial"] * keep.view(B, 1, 1, 1, 1)
                b["actions"] = b["actions"] * keep.view(B, 1, 1)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, *_ = flow_loss(model, b["latents"], b["spatial"], b["actions"], b["text"], n_ctx, a.shift)
            (loss / a.grad_accum).backward()
            tot += loss.item() / a.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
        # Decide on the (all-reduced, identical on every GPU) gradient norm so all GPUs skip together.
        if not torch.isfinite(gn):
            print(f"non-finite loss/grad at step {step}; skipping update", flush=True)
            continue
        opt.step()
        step += 1

        if step % a.log_every == 0:
            if ddp:  # mean training loss over GPUs
                t = torch.tensor([tot], device=device)
                dist.all_reduce(t)
                tot = t.item() / world
        if step % a.log_every == 0 and main_proc:
            rec = {"step": step, "loss": tot, "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"],
                   "sec_per_step": (time.time() - t0) / a.log_every}
            if device.type == "cuda":
                rec["mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            wb.log(run, rec, step)
            t0 = time.time()
        if (step % a.val_every == 0 or step == a.steps) and main_proc:
            rec = {"step": step, **evaluate(raw, val_batches, a, device, amp)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            wb.log(run, rec, step)
        if (step % a.save_every == 0 or step == a.steps) and main_proc:
            tmp = out / "last.tmp"
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a),
                        "latent_channels": C, "frames_per_latent": fpl}, tmp)
            tmp.rename(ckpt)
            if a.keep_weights:
                (out / "weights").mkdir(exist_ok=True)
                w = out / "weights" / f"step{step}.pt"
                torch.save({"model": {k: v.bfloat16() if v.is_floating_point() else v
                                      for k, v in raw.state_dict().items()},
                            "step": step, "args": vars(a), "latent_channels": C, "frames_per_latent": fpl},
                           w.with_suffix(".tmp"))
                w.with_suffix(".tmp").rename(w)
        if step % a.val_every == 0 or step % a.save_every == 0 or step == a.steps:
            barrier()  # other GPUs wait while the main one validates / saves
    if run is not None:
        run.finish()
    if ddp:
        dist.destroy_process_group()
    if main_proc:
        print("done", flush=True)


if __name__ == "__main__":
    main()
