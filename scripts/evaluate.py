"""Roll out the world model on held-out clips and measure whether it obeys the hands.

For each clip: start from the first real frame, generate the rest from the clip's actions,
decode, and score:
  hand_err_px     MediaPipe joints on generated frames vs the commanded (projected) joints,
                  px at 1080p, median over detected hands
  hand_recall     fraction of commanded visible hands that MediaPipe finds in generated frames
  psnr            vs the real frames (only meaningful for the true-action rollout)
  swapped_*       same start frame, actions taken from a DIFFERENT clip: does the hand follow?
Reference rows (no model needed):
  oracle          real frames through the VAE: the best achievable hand_err / recall
  context_copy    every frame = the first frame: what ignoring the actions scores

python scripts/evaluate.py --ckpt runs/v1/last.pt --data /data/latents --clips 32 --out runs/v1/eval
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex.conditioning import ACTION_DIM, NUM_SPATIAL_CHANNELS  # noqa: E402
from egodex.dataset import LatentClips, frames_for_latents  # noqa: E402
from egodex.handdet import HandDetector, match_to_hands  # noqa: E402
from egodex.model import ActionWorldModel, load_wan, tiny_wan  # noqa: E402
from egodex.vae import from_video_tensor, latent_norm, load_vae  # noqa: E402


def shifted_sigmas(steps, shift):
    s = torch.linspace(1, 0, steps + 1)
    return shift * s / (1 + (shift - 1) * s)


@torch.no_grad()
def generate(model, ctx, spatial, actions, text, n_ctx=1, steps=30, shift=3.0, generator=None):
    """ctx: (1,C,F,h,w) whose first n_ctx latent frames are real. Causal model: frame-by-frame,
    each frame denoised from noise given the clean past. Bidirectional: all future frames jointly."""
    x = ctx.clone()
    F = x.shape[2]
    sig = shifted_sigmas(steps, shift).to(x.device)
    chunks = [[k] for k in range(n_ctx, F)] if model.causal else [list(range(n_ctx, F))]
    for ks in chunks:
        end = ks[-1] + 1
        x[:, :, ks] = torch.randn(x[:, :, ks].shape, device=x.device, generator=generator, dtype=x.dtype)
        for i in range(steps):
            sigma = torch.zeros(1, end, device=x.device)
            sigma[:, ks] = sig[i]
            v = model(x[:, :, :end], sigma, spatial[:, :, :end], actions[:, :end], text)
            x[:, :, ks] = x[:, :, ks] + (sig[i + 1] - sig[i]) * v[:, :, ks].to(x.dtype)
    return x


def decode(vae, z, mean, std):
    if mean is not None:
        shape = (1, -1) + (1,) * (z.ndim - 2)
        z = z.float() * std.to(z.device).view(shape) + mean.to(z.device).view(shape)
    params = list(vae.parameters())
    return from_video_tensor(vae.decode(z.to(params[0].dtype) if params else z).sample)


def hand_metrics(det, frames, uv, valid, size_wh):
    """frames: (T,H,W,3) generated at size_wh; uv/valid: commanded joints (T,2,21,2)/(T,2,21) at 1080p."""
    sx, sy = size_wh[0] / 1920, size_wh[1] / 1080
    errs, found, total = [], 0, 0
    for t in range(1, len(frames)):  # frame 0 is the given context
        target = uv[t] * [sx, sy]
        dets = det(frames[t])
        m = match_to_hands(dets, target, valid[t], max_dist=120 * sx / 0.44)
        for h in range(2):
            if valid[t, h].sum() < 15:
                continue
            total += 1
            if h in m:
                found += 1
                errs.append(np.linalg.norm((m[h] - target[h])[valid[t, h]], axis=-1).mean() / sx)
    return errs, found, total


def summarize(errs, found, total):
    return {"hand_err_px": round(float(np.median(errs)), 1) if errs else None,
            "hand_recall": round(found / max(total, 1), 3), "n_hands": total}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="omit to compute only the reference rows")
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["val_unseen", "val_seen"])
    ap.add_argument("--action-stats", default="data/action_stats.npz")
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--clips", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-latent-frames", type=int, default=None)
    ap.add_argument("--videos", type=int, default=6, help="side-by-side mp4s to write")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, margs = None, {}
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        margs = ck["args"]
        tiny = margs["model"] == "tiny"
        tr = tiny_wan(ck["latent_channels"], 64) if tiny else load_wan(margs["wan_repo"], dtype=torch.float32)
        model = ActionWorldModel(tr, NUM_SPATIAL_CHANNELS, ACTION_DIM, ck["frames_per_latent"], margs["causal"])
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        if device.type == "cuda":
            model = model.to(torch.bfloat16)
    tiny = margs.get("model") == "tiny"
    ds = LatentClips(a.data, a.splits, action_stats=a.action_stats, text_emb=a.text_emb,
                     text_dim=64 if tiny else 4096, text_len=16 if tiny else 512,
                     max_latent_frames=a.max_latent_frames, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    pick = rng.permutation(len(ds))[: a.clips]
    meta = ds._load(ds.items[pick[0]][0])["meta"]
    size_wh = tuple(meta["size_wh"])
    vae = load_vae(meta["vae"], device, dtype=torch.float32, tiling=device.type != "cuda")
    mean, std = latent_norm(vae)
    det = HandDetector()

    rows = {k: {"errs": [], "found": 0, "total": 0, "psnr": []} for k in
            ("oracle", "context_copy", "model", "swapped")}

    def add(key, frames, uv, valid, real=None):
        e, f, t = hand_metrics(det, frames, uv, valid, size_wh)
        rows[key]["errs"] += e
        rows[key]["found"] += f
        rows[key]["total"] += t
        if real is not None:
            rows[key]["psnr"].append(np.mean([psnr(x, y) for x, y in zip(frames[1:], real[1:])]))

    for n, idx in enumerate(pick):
        b = ds[int(idx)]
        fi, ci = ds.items[int(idx)]
        raw = ds._load(fi)
        Fl = b["latents"].shape[1]
        groups = frames_for_latents(raw["uv"].shape[1], raw["latents"].shape[2])[:Fl]
        vid_idx = np.arange(groups.max() + 1)
        uv, valid = raw["uv"][ci].numpy()[vid_idx], raw["valid"][ci].numpy()[vid_idx]
        lat = b["latents"][None].to(device)
        real = decode(vae, lat, mean, std)[: len(vid_idx)]
        add("oracle", real, uv, valid)
        add("context_copy", np.repeat(real[:1], len(real), 0), uv, valid, real)
        if model is None:
            continue
        dt = next(model.parameters()).dtype
        cond = lambda bb: (bb["spatial"][None].to(device, dt), bb["actions"][None].to(device, dt),  # noqa: E731
                           bb["text"][None].to(device, dt))
        g = torch.Generator(device=device).manual_seed(a.seed + n)
        gen = generate(model, lat.to(dt), *cond(b), steps=a.steps, shift=margs.get("shift", 3.0), generator=g)
        fake = decode(vae, gen, mean, std)[: len(vid_idx)]
        add("model", fake, uv, valid, real)

        # Swapped actions: another clip's hand motion from this clip's first frame. The target is
        # the other clip's joints; they won't match the scene, but the hands should follow them.
        other = int(pick[(n + 1) % len(pick)])
        b2 = ds[other]
        fi2, ci2 = ds.items[other]
        raw2 = ds._load(fi2)
        uv2, valid2 = raw2["uv"][ci2].numpy()[vid_idx], raw2["valid"][ci2].numpy()[vid_idx]
        gen2 = generate(model, lat.to(dt), *cond(b2), steps=a.steps, shift=margs.get("shift", 3.0), generator=g)
        fake2 = decode(vae, gen2, mean, std)[: len(vid_idx)]
        add("swapped", fake2, uv2, valid2)

        if n < a.videos:
            W, H = size_wh
            vw = cv2.VideoWriter(str(out / f"clip{n:02d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (3 * W, H))
            for t in range(len(real)):
                row = [real[t], fake[t], fake2[t]]
                tiles = []
                for im, lab in zip(row, ["real", "generated", "swapped actions"]):
                    im = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR)
                    cv2.putText(im, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    tiles.append(im)
                vw.write(np.concatenate(tiles, 1))
            vw.release()
        print(f"clip {n + 1}/{len(pick)}", flush=True)

    det.close()
    res = {}
    for k, r in rows.items():
        if r["total"] == 0 and not r["psnr"]:
            continue
        res[k] = summarize(r["errs"], r["found"], r["total"])
        if r["psnr"]:
            res[k]["psnr"] = round(float(np.mean(r["psnr"])), 2)
    res["_info"] = {"ckpt": a.ckpt, "clips": len(pick), "splits": a.splits, "steps": a.steps}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
