"""Distill the world model into a few-step real-time generator (Self Forcing / DMD recipe).

The trained model needs ~30 denoising steps x 2 (guidance) per latent frame. The student does
STEPS (default 4) with guidance baked in, and is trained on its OWN autoregressive rollouts, so
what it learns matches how it runs live (no train/test gap from teacher-forced context).

Two stages, both started from the trained model's weights:
  ode   (warm start) the teacher generates frames with its full sampler + guidance; the student
        learns to jump from the teacher's intermediate states at the student's noise levels
        straight to the teacher's final frame. Our teacher is already causal / frame-by-frame,
        which is what Causal Forcing (2026) found makes this stage work well.
  dmd   (main) the student rolls out a clip from a real first frame and real hand poses; a
        distribution-matching loss pulls its frames toward what the teacher would generate
        (teacher with guidance = "real" score) and away from a second model that tracks the
        student's own outputs ("fake" score, trained alongside with the usual flow loss).

  torchrun --nproc_per_node 4 scripts/distill.py --stage ode --teacher runs/v3/weights/stepN.pt \
      --data /workspace/data/latents_full --text-emb /workspace/data/text_emb.pt --out runs/d1_ode
  torchrun ... --stage dmd --teacher runs/v3/weights/stepN.pt --init runs/d1_ode/student.pt --out runs/d1
Mac smoke test: --model-tiny-ok with a tiny checkpoint and data/latents_smoke (see test at bottom).

Student sampling (egodex/live_session.py uses it when the checkpoint says "student"): per frame,
from noise at sigma_0: predict x0, re-noise to sigma_{i+1} with fresh noise, ..., output the
last x0; then cache the frame at sigma 0.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex.conditioning import ACTION_DIM, NUM_SPATIAL_CHANNELS  # noqa: E402
from egodex.dataset import LatentClips  # noqa: E402
from egodex.model import ActionWorldModel, flow_loss, load_wan, tiny_wan  # noqa: E402
from egodex.stream import StreamingWorldModel  # noqa: E402
from egodex import wb  # noqa: E402


def student_sigmas(steps: int, shift: float) -> list[float]:
    """Student noise levels: uniform in the unshifted variable, Wan-shifted; e.g. 4 steps, shift 3
    -> [1.0, 0.9, 0.75, 0.5]. Teacher's ODE grid (teacher_grid) contains these exactly."""
    u = [1 - k / steps for k in range(steps)]
    return [shift * x / (1 + (shift - 1) * x) for x in u]


def shifted(u, shift):
    return shift * u / (1 + (shift - 1) * u)


def load_model(path, device, dtype=torch.float32):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    tr = tiny_wan(ck["latent_channels"], 64) if a["model"] == "tiny" else load_wan(a["wan_repo"], dtype=torch.float32)
    m = ActionWorldModel(tr, NUM_SPATIAL_CHANNELS, ACTION_DIM, ck["frames_per_latent"], a["causal"])
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return m.to(device, dtype), ck


def v_full(model, x, sigma, sp, ac, text, cfg=1.0):
    """Full-sequence (block-causal) prediction with optional guidance on the hands."""
    if cfg == 1.0:
        return model(x, sigma, sp, ac, text)
    v = model(torch.cat([x, x]), torch.cat([sigma, sigma]), torch.cat([sp, torch.zeros_like(sp)]),
              torch.cat([ac, torch.zeros_like(ac)]), torch.cat([text, text]))
    vc, vu = v.chunk(2)
    return vu + cfg * (vc - vu)


def frame_x0(model, past, x, sig, sp, ac, text):
    """Student x0 for the LAST frame given clean past frames (detached) and its noisy input:
    a full-sequence forward (equal to the cached streaming forward; scripts/test_stream.py),
    used where gradients are needed because the KV-cache processors can't be re-run by
    gradient checkpointing."""
    B = x.shape[0]
    seq = torch.cat([past, x], 2)
    sigma = torch.zeros(B, seq.shape[2], device=x.device)
    sigma[:, -1] = sig
    v = model(seq, sigma, sp[:, :, :seq.shape[2]], ac[:, :seq.shape[2]], text)[:, :, -1:]
    return x - sig * v


@torch.no_grad()
def teacher_rollout(teacher, ctx, sp, ac, text, n_ctx, grid, record_u, cfg, gen):
    """Teacher's sampler (Euler on the shifted grid, with guidance), recording the state at the
    student's noise levels. Returns final clips (B,C,F,h,w) and, per generated frame, a list of
    (x at each recorded level)."""
    s = StreamingWorldModel(teacher, text, cfg=cfg)
    for k in range(n_ctx):
        s.push(ctx[:, :, k:k + 1], sp[:, :, k:k + 1], ac[:, k:k + 1])
    out, states = [ctx[:, :, :n_ctx]], []
    F_ = sp.shape[2]
    for k in range(n_ctx, F_):
        x = torch.randn(ctx[:, :, :1].shape, device=ctx.device, dtype=ctx.dtype, generator=gen)
        rec = {}
        for i in range(len(grid) - 1):
            if i in record_u:
                rec[i] = x.clone()
            v = s._v(x, grid[i], sp[:, :, k:k + 1], ac[:, k:k + 1], write=False)
            x = x + (grid[i + 1] - grid[i]) * v.to(x.dtype)
        s.push(x, sp[:, :, k:k + 1], ac[:, k:k + 1])
        out.append(x)
        states.append([rec[i] for i in sorted(record_u)])
    return torch.cat(out, 2), states


def student_rollout(student, ctx, sp, ac, text, n_ctx, sigs, gen, grad_exit=None):
    """Student generates frames n_ctx.. autoregressively with its few-step sampler. With
    grad_exit (one exit index per frame), the exit step runs with gradients and its x0 is that
    frame's output (Self Forcing); otherwise no gradients. Returns (clip, outputs w/ grad)."""
    B = ctx.shape[0]
    s = StreamingWorldModel(student, text, cfg=1.0)
    with torch.no_grad():
        for k in range(n_ctx):
            s.push(ctx[:, :, k:k + 1], sp[:, :, k:k + 1], ac[:, k:k + 1])
    frames = [ctx[:, :, :n_ctx]]
    F_ = sp.shape[2]
    for j, k in enumerate(range(n_ctx, F_)):
        x = torch.randn(ctx[:, :, :1].shape, device=ctx.device, dtype=ctx.dtype, generator=gen)
        last = len(sigs) - 1 if grad_exit is None else grad_exit[j]
        x0 = None
        for i in range(last + 1):
            if grad_exit is not None and i == last:
                past = torch.cat(frames, 2).detach()
                x0 = frame_x0(student, past, x, sigs[i], sp, ac, text)
            else:
                with torch.no_grad():
                    v = s._v(x, sigs[i], sp[:, :, k:k + 1], ac[:, k:k + 1], write=False)
                    x0 = x - sigs[i] * v.to(x.dtype)
            if i < last:
                nxt = sigs[i + 1]
                x = ((1 - nxt) * x0 + nxt * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)).detach()
        with torch.no_grad():
            s.push(x0.detach(), sp[:, :, k:k + 1], ac[:, k:k + 1])
        frames.append(x0)
    return torch.cat(frames, 2)


def allreduce_grads(params, world):
    """Average gradients across GPUs (models aren't DDP-wrapped: the student runs many forwards
    per update, which DDP's static bucketing doesn't allow)."""
    if world == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    for i in range(0, len(grads), 64):
        chunk = grads[i:i + 64]
        flat = torch.cat([g.flatten() for g in chunk])
        dist.all_reduce(flat)
        flat /= world
        o = 0
        for g in chunk:
            g.copy_(flat[o:o + g.numel()].view_as(g))
            o += g.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ode", "dmd"])
    ap.add_argument("--teacher", required=True, help="trained world model weights")
    ap.add_argument("--init", default=None, help="student start (dmd: the ode stage's student.pt)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val_seen"])
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--action-stats", default="data/action_stats.npz")
    ap.add_argument("--student-steps", type=int, default=4)
    ap.add_argument("--teacher-steps", type=int, default=32, help="must be a multiple of --student-steps")
    ap.add_argument("--cfg", type=float, default=3.0, help="guidance baked into the student")
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--n-ctx", type=int, default=1)
    ap.add_argument("--max-latent-frames", type=int, default=None)
    ap.add_argument("--crop", type=int, nargs=2, default=None)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-6, help="student")
    ap.add_argument("--lr-fake", type=float, default=4e-7, help="fake score (dmd)")
    ap.add_argument("--fake-ratio", type=int, default=5, help="dmd: fake-score updates per student update")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default=None)
    a = ap.parse_args()
    assert a.teacher_steps % a.student_steps == 0

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        from datetime import timedelta
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=1))
    main_proc = rank == 0
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(a.seed + rank)
    random.seed(a.seed + rank)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    teacher, tck = load_model(a.teacher, device, torch.bfloat16 if amp else torch.float32)
    teacher.eval().requires_grad_(False)
    tiny = tck["args"]["model"] == "tiny"
    student, _ = load_model(a.init or a.teacher, device)
    fake = copy.deepcopy(student) if a.stage == "dmd" else None
    if a.stage == "dmd" and a.init:  # the fake score starts as the teacher, not the warm-started student
        fake, _ = load_model(a.teacher, device)
    for m in (student, fake):
        if m is not None and a.grad_ckpt:
            m.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.0, 0.999), weight_decay=0.01)
    opt_f = torch.optim.AdamW(fake.parameters(), lr=a.lr_fake, betas=(0.0, 0.999), weight_decay=0.01) if fake else None

    it0 = 0
    ckpt = out / "last.pt"
    if a.resume and ckpt.exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(s["student"])
        opt.load_state_dict(s["opt"])
        if fake is not None:
            fake.load_state_dict(s["fake"])
            opt_f.load_state_dict(s["opt_f"])
        it0 = s["iter"]

    text_dim = 64 if tiny else 4096
    ds = LatentClips(a.data, a.splits, action_stats=a.action_stats, text_emb=a.text_emb, text_dim=text_dim,
                     text_len=16 if tiny else 512, max_latent_frames=a.max_latent_frames, crop_hw=a.crop)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=a.seed) if world > 1 else None
    if sampler is not None:
        sampler.set_epoch(it0)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True,
                        num_workers=a.num_workers, persistent_workers=a.num_workers > 0)
    it = iter(loader)

    sigs = student_sigmas(a.student_steps, a.shift)
    grid = [shifted(1 - k / a.teacher_steps, a.shift) for k in range(a.teacher_steps + 1)]
    stride = a.teacher_steps // a.student_steps
    record = {k * stride for k in range(a.student_steps)}  # grid indices == student sigmas
    assert all(abs(grid[k * stride] - sigs[k]) < 1e-6 for k in range(a.student_steps))
    gen = torch.Generator(device=device).manual_seed(a.seed * 1000 + rank)
    log = open(out / "log.jsonl", "a") if main_proc else None
    run = wb.init(a.wandb_project, out, vars(a), f"distill_{a.stage}") if main_proc else None
    if main_proc:
        (out / "args.json").write_text(json.dumps(vars(a), indent=2))
        print(f"stage {a.stage}, student sigmas {[round(x, 3) for x in sigs]}, clips {len(ds)}, GPUs {world}", flush=True)

    def batch():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        dt = torch.bfloat16 if amp else torch.float32
        return (b["latents"].to(device, dt), b["spatial"].to(device, dt), b["actions"].to(device, dt),
                b["text"].to(device, dt))

    t0 = time.time()
    for i in range(it0, a.iters):
        x, sp, ac, tx = batch()
        rec = {"iter": i + 1}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            if a.stage == "ode":
                clip, states = teacher_rollout(teacher, x, sp, ac, tx, a.n_ctx, grid, record, a.cfg, gen)
                # every generated frame, one random student level: jump to the teacher's result
                losses = []
                for j, k in enumerate(range(a.n_ctx, clip.shape[2])):
                    lvl = random.randrange(a.student_steps)
                    pred = frame_x0(student, clip[:, :, :k].detach(), states[j][lvl], sigs[lvl], sp, ac, tx)
                    losses.append(F.mse_loss(pred.float(), clip[:, :, k:k + 1].float()))
                loss = torch.stack(losses).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                allreduce_grads(list(student.parameters()), world)
                rec["gn"] = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
                opt.step()
                rec["ode_loss"] = loss.item()
            else:
                gen_turn = i % a.fake_ratio == 0
                n_gen = sp.shape[2] - a.n_ctx
                if gen_turn:
                    exits = [random.randrange(a.student_steps) for _ in range(n_gen)]
                    if world > 1:  # same exit steps on every GPU (same graph shape and cost)
                        e = torch.tensor(exits, device=device)
                        dist.broadcast(e, 0)
                        exits = e.tolist()
                    clip = student_rollout(student, x, sp, ac, tx, a.n_ctx, sigs, gen, exits)
                else:
                    with torch.no_grad():
                        clip = student_rollout(student, x, sp, ac, tx, a.n_ctx, sigs, gen)
                B, Fr = clip.shape[0], clip.shape[2]
                n_ctx = torch.full((B,), a.n_ctx, device=device)
                if gen_turn:
                    # DMD: noise the student's clip, compare the teacher's and fake model's x0
                    t = shifted(torch.rand(B, Fr, device=device, generator=gen) * 0.96 + 0.02, a.shift)
                    t[:, :a.n_ctx] = 0
                    tt = t[:, None, :, None, None].to(clip.dtype)
                    eps = torch.randn(clip.shape, device=device, dtype=clip.dtype, generator=gen)
                    xt = ((1 - tt) * clip + tt * eps).detach()
                    with torch.no_grad():
                        x0_real = xt - tt * v_full(teacher, xt, t, sp, ac, tx, a.cfg)
                        x0_fake = xt - tt * v_full(fake, xt, t, sp, ac, tx)
                        g = gen_frames = slice(a.n_ctx, None)
                        norm = (clip - x0_real).abs().mean(dim=(1, 3, 4), keepdim=True).clamp(min=1e-6)
                        grad = ((x0_fake - x0_real) / norm)[:, :, g]
                    target = (clip[:, :, gen_frames] - grad).detach()
                    loss = 0.5 * F.mse_loss(clip[:, :, gen_frames].float(), target.float())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    allreduce_grads(list(student.parameters()), world)
                    rec["gn"] = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
                    opt.step()
                    rec["dmd_loss"] = loss.item()
                # fake score: ordinary flow loss on the student's own clips
                lf, *_ = flow_loss(fake, clip.detach(), sp, ac, tx, n_ctx, a.shift, gen)
                opt_f.zero_grad(set_to_none=True)
                lf.backward()
                allreduce_grads(list(fake.parameters()), world)
                torch.nn.utils.clip_grad_norm_(fake.parameters(), 1.0)
                opt_f.step()
                rec["fake_loss"] = lf.item()
        if main_proc and ((i + 1) % a.log_every == 0 or i == it0):
            rec["sec_per_iter"] = (time.time() - t0) / max(1, (i + 1 - it0) if i == it0 else a.log_every)
            t0 = time.time()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            wb.log(run, rec, i + 1)
        if main_proc and ((i + 1) % a.save_every == 0 or i + 1 == a.iters):
            sd = {k: v.bfloat16() if v.is_floating_point() else v for k, v in student.state_dict().items()}
            meta = {"args": tck["args"], "latent_channels": tck["latent_channels"],
                    "frames_per_latent": tck["frames_per_latent"],
                    "student": {"steps": a.student_steps, "sigmas": sigs, "cfg": a.cfg, "stage": a.stage, "iter": i + 1}}
            torch.save({"model": sd, **meta}, out / "student.tmp")
            (out / "student.tmp").rename(out / "student.pt")
            full = {"student": student.state_dict(), "opt": opt.state_dict(), "iter": i + 1}
            if fake is not None:
                full.update(fake=fake.state_dict(), opt_f=opt_f.state_dict())
            torch.save(full, out / "last.tmp")
            (out / "last.tmp").rename(ckpt)
        if world > 1 and (i + 1) % a.save_every == 0:
            dist.barrier()
    if run is not None:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
