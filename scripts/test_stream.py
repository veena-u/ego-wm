"""Check that KV-cached streaming generation equals full-recompute generation, and time both.

    python scripts/test_stream.py                       # tiny random model on CPU (exactness)
    python scripts/test_stream.py --ckpt runs/v2/last.pt --data /workspace/data/latents \
        --text-emb /workspace/data/text_emb.pt          # real model on GPU (exactness + speed)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egodex.conditioning import ACTION_DIM, NUM_SPATIAL_CHANNELS  # noqa: E402
from egodex.model import ActionWorldModel, load_wan, tiny_wan  # noqa: E402
from egodex.stream import StreamingWorldModel  # noqa: E402

spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--frames", type=int, default=6, help="latent frames (tiny model)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="real model in fp32 (separates rounding from bugs)")
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.manual_seed(0)

    if a.ckpt is None:  # tiny random model; make the zero-initialized conditioning paths non-zero
        fpl, C, h, w, F = 4, 16, 8, 12, a.frames
        model = ActionWorldModel(tiny_wan(C, 64), NUM_SPATIAL_CHANNELS, ACTION_DIM, fpl, causal=True)
        with torch.no_grad():
            model.spatial_embed.weight.normal_(0, 0.02)
            model.action_mlp[-1].weight.normal_(0, 0.02)
        model = model.to(device).eval()
        ctx = torch.randn(1, C, F, h, w, device=device)
        spatial = torch.rand(1, NUM_SPATIAL_CHANNELS * fpl, F, h, w, device=device)
        actions = torch.randn(1, F, ACTION_DIM * fpl, device=device)
        text = torch.randn(1, 16, 64, device=device)
    else:
        from egodex.dataset import LatentClips
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m = ck["args"]
        model = ActionWorldModel(load_wan(m["wan_repo"], dtype=torch.float32), NUM_SPATIAL_CHANNELS,
                                 ACTION_DIM, ck["frames_per_latent"], m["causal"])
        model.load_state_dict(ck["model"])
        dt = torch.float32 if a.fp32 else torch.bfloat16
        model = model.to(device).eval().to(dt)
        ds = LatentClips(a.data, ["val_unseen"], action_stats=str(ROOT / "data/action_stats.npz"),
                         text_emb=a.text_emb, text_dim=4096, text_len=512, max_latent_frames=None)
        b = ds[0]
        ctx = b["latents"][None].to(device, dt)
        spatial, actions, text = (b[k][None].to(device, dt) for k in ("spatial", "actions", "text"))
        F = ctx.shape[2]
    print(f"latent frames {F}, steps {a.steps}, device {device}, dtype {ctx.dtype}")

    # 1) Teacher-forced: same REAL past frames, one prediction for frame k at sigma=0.5. No rollout,
    #    so nothing amplifies small differences: this isolates whether the cache computes the same thing.
    with torch.no_grad():
        s = StreamingWorldModel(model, text)
        s.push(ctx[:, :, :1], spatial[:, :, :1], actions[:, :1])
        tf = []
        for k in range(1, F):
            eps = torch.randn(ctx[:, :, k:k + 1].shape, device=device, dtype=ctx.dtype)
            noisy = 0.5 * ctx[:, :, k:k + 1] + 0.5 * eps
            sig = torch.zeros(1, k + 1, device=device)
            sig[:, k] = 0.5
            v_full = model(torch.cat([ctx[:, :, :k], noisy], 2), sig, spatial[:, :, :k + 1], actions[:, :k + 1], text)[:, :, k:]
            v_str = s._run(noisy, 0.5, spatial[:, :, k:k + 1], actions[:, k:k + 1], write=False)
            tf.append(((v_str.float() - v_full.float()).norm() / v_full.float().norm()).item())
            s.push(ctx[:, :, k:k + 1], spatial[:, :, k:k + 1], actions[:, k:k + 1])
    print("teacher-forced relative error per frame:", " ".join(f"{e:.1e}" for e in tf))

    g = torch.Generator(device=device).manual_seed(1)
    sync(device); t0 = time.time()
    full = ev.generate(model, ctx, spatial, actions, text, n_ctx=1, steps=a.steps, generator=g)
    sync(device); t_full = time.time() - t0

    g = torch.Generator(device=device).manual_seed(1)
    s = StreamingWorldModel(model, text)
    sync(device); t0 = time.time()
    s.push(ctx[:, :, :1], spatial[:, :, :1], actions[:, :1])
    outs = [ctx[:, :, :1]]
    for k in range(1, F):
        outs.append(s.next(spatial[:, :, k:k + 1], actions[:, k:k + 1], steps=a.steps, generator=g))
    sync(device); t_stream = time.time() - t0
    stream = torch.cat(outs, 2)

    err = ((stream.float() - full.float()).abs().max() / full.float().abs().max()).item()
    per = [((stream[:, :, k] - full[:, :, k]).float().abs().max() / full[:, :, k].float().abs().max()).item()
           for k in range(F)]
    # The rollout comparison is information only: autoregressive sampling is chaotic, so even
    # 1e-7 differences (fp32 kernel ordering) grow into a different-but-valid sample over many
    # frames x steps. Correctness is judged on the teacher-forced error, which must stay at the
    # dtype's rounding level and must NOT grow with the frame index.
    print("rollout difference per frame (info):", " ".join(f"{e:.1e}" for e in per))
    print(f"full recompute {t_full:.2f}s   streaming {t_stream:.2f}s   speedup {t_full / t_stream:.1f}x")
    tol = 1e-5 if ctx.dtype == torch.float32 else 2e-2
    worst = max(tf)
    print("PASS" if worst < tol else f"FAIL (teacher-forced error {worst:.2e} > {tol})")

    # Rolling window keeps memory flat: run past the window and check the cache size.
    s2 = StreamingWorldModel(model, text, window=3, sink=1)
    s2.push(ctx[:, :, :1], spatial[:, :, :1], actions[:, :1])
    for k in range(1, F):
        s2.next(spatial[:, :, k:k + 1], actions[:, k:k + 1], steps=1)
    print(f"rolling window: cached frames {len(s2.cache)} (window 3) after {F} frames, next position {s2.t}")


if __name__ == "__main__":
    main()
