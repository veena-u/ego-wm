"""Roll out the world model on held-out clips and measure whether it obeys the hands.

For each clip: start from the first real frame, generate the rest from the clip's actions,
decode, and score. Hand metrics use the EgoDex-trained estimator (--handpose; primary) and/or
MediaPipe (--mediapipe; optional, weaker on egocentric hands):
  <inst>_err_px   estimated joints on generated frames vs the commanded joints, px at 1080p,
                  median over found hands of the mean joint error
  <inst>_recall   fraction of commanded visible hands (>=15 valid joints) that are found
  psnr            vs the real frames (only meaningful for the true-action rollout)
  bg_flow_px      median optical flow outside the hands between consecutive frames (px @ output
                  res): how much the camera/background moves
Rows:
  oracle          real frames through the VAE: the best achievable scores (the floor)
  context_copy    every frame = the first frame: what ignoring the actions scores
  model           rollout with the clip's own actions
  swapped         same start frame, ANOTHER clip's actions: do the hands follow the new motion?
  static_head     clip's hand actions but head motion = none: is the camera steady (bg_flow_px
                  well below the real clip's) while the hands still follow (err/recall)?

python scripts/evaluate.py --ckpt runs/v1/last.pt --data /data/latents --handpose runs/handpose/handpose.pt \
    --clips 32 --out runs/v1/eval
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


def mediapipe_metrics(det, frames, uv, valid, size_wh):
    """frames: (T,H,W,3) at size_wh; uv/valid: commanded joints (T,2,21,2)/(T,2,21) at 1080p."""
    sx, sy = size_wh[0] / 1920, size_wh[1] / 1080
    errs, found, total = [], 0, 0
    for t in range(1, len(frames)):  # frame 0 is the given context
        target = uv[t] * [sx, sy]
        m = match_to_hands(det(frames[t]), target, valid[t], max_dist=120 * sx / 0.44)
        for h in range(2):
            if valid[t, h].sum() < 15:
                continue
            total += 1
            if h in m:
                found += 1
                errs.append(np.linalg.norm((m[h] - target[h])[valid[t, h]], axis=-1).mean() / sx)
    return errs, found, total


def handpose_metrics(est, frames, uv, valid):
    """Same rules with the EgoDex-trained estimator (it knows left from right, so no matching)."""
    puv, _, prob = est(frames[1:])
    errs, found, total = [], 0, 0
    for t in range(len(puv)):
        for h in range(2):
            v = valid[t + 1, h]
            if v.sum() < 15:
                continue
            total += 1
            if prob[t, h] > 0.5:
                found += 1
                errs.append(np.linalg.norm(puv[t, h] - uv[t + 1, h], axis=-1)[v].mean())
    return errs, found, total


def background_flow(frames, uv, valid, size_wh, margin=40):
    """Camera motion: median displacement of textured background points (Shi-Tomasi corners outside
    dilated hand boxes, tracked with Lucas-Kanade) between consecutive frames. Dense flow is NOT used:
    walls and tables are textureless, their flow is ~0, and a median over all pixels always read 0."""
    sx, sy = size_wh[0] / 1920, size_wh[1] / 1080
    gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in frames]
    mags = []
    for t in range(len(frames) - 1):
        mask = np.ones(gray[t].shape, bool)
        for tt in (t, t + 1):
            for h in range(2):
                if valid[tt, h].sum() < 3:
                    continue
                p = uv[tt, h][valid[tt, h]] * [sx, sy]
                x0, y0 = np.maximum(p.min(0) - margin, 0).astype(int)
                x1, y1 = (p.max(0) + margin).astype(int)
                mask[y0:y1, x0:x1] = False
        if mask.mean() < 0.2:
            continue
        pts = cv2.goodFeaturesToTrack(gray[t], 300, 0.01, 8, mask=mask.astype(np.uint8) * 255)
        if pts is None or len(pts) < 10:
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t + 1], pts, None, winSize=(21, 21), maxLevel=3)
        ok = st[:, 0] == 1
        if ok.sum() >= 10:
            mags.append(float(np.median(np.linalg.norm((nxt - pts)[ok, 0], axis=-1))))
    return float(np.median(mags)) if mags else None


def summarize(prefix, errs, found, total):
    return {f"{prefix}_err_px": round(float(np.median(errs)), 1) if errs else None,
            f"{prefix}_recall": round(found / max(total, 1), 3), "n_hands": total}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


@torch.no_grad()  # without this the VAE decode keeps its autograd graph (~75 GB for one clip)
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
    ap.add_argument("--handpose", default=None, help="EgoDex hand pose estimator checkpoint (primary metric)")
    ap.add_argument("--mediapipe", action="store_true", help="also score with MediaPipe")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance on the hand conditioning")
    ap.add_argument("--full-recompute", action="store_true", help="old non-cached sampler (for comparison)")
    ap.add_argument("--shard", default="1/1", help="i/n: score every n-th clip (one process per GPU); "
                    "then run once with --merge n to combine. scripts/eval_parallel.sh does both")
    ap.add_argument("--merge", type=int, default=None, help="combine n shard files in --out into metrics.json")
    a = ap.parse_args()
    device = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    si, ns = map(int, a.shard.split("/"))
    if a.merge:
        shards = [json.loads((out / f"shard{i}of{a.merge}.json").read_text()) for i in range(1, a.merge + 1)]
        rows, instruments = shards[0]["rows"], shards[0]["instruments"]
        for sh in shards[1:]:
            for k, r in sh["rows"].items():
                rows[k]["psnr"] += r["psnr"]
                rows[k]["bg"] += r["bg"]
                for name in instruments:
                    for f in ("errs", "found", "total"):
                        rows[k][name][f] += r[name][f]
        a.ckpt, a.cfg, a.steps = shards[0]["ckpt"], shards[0]["cfg"], shards[0]["steps"]
        n_clips = sum(sh["n"] for sh in shards)
        return finish(a, out, rows, instruments, n_clips)

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
    # bf16 on GPU: same reconstruction quality as fp32 (measured for preprocessing) at half the
    # memory; a full fp32 untiled decode of one clip needs ~75 GB.
    vae = load_vae(meta["vae"], device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                   tiling=device.type != "cuda")
    mean, std = latent_norm(vae)
    instruments = {}
    if a.handpose:
        from egodex.handpose import Estimator
        est = Estimator(a.handpose, device)
        instruments["hp"] = lambda fr, uv, va: handpose_metrics(est, fr, uv, va)
    if a.mediapipe or not a.handpose:
        det = HandDetector()
        instruments["mp"] = lambda fr, uv, va: mediapipe_metrics(det, fr, uv, va, size_wh)
    # Normalized head action for "no head motion": identity rotation (6D = [1,0,0,0,1,0]) and zero
    # translation, mapped through the same normalization as training - NOT all-zeros.
    a_mean, a_std = ds.a_mean[:9], ds.a_std[:9]
    still = (np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], np.float32) - a_mean) / a_std

    student = ck.get("student") if a.ckpt else None
    if student:  # distilled few-step student (scripts/distill.py): guidance is baked in
        from egodex.stream import stream_generate
        print(f"student checkpoint: {student['steps']} steps, guidance {student['cfg']} baked in", flush=True)

        def sample(*x, **k):
            return stream_generate(*x, **k, student_sigmas=student["sigmas"])
    elif model is not None and model.causal and not a.full_recompute:
        # KV-cached streaming: exact up to float rounding (scripts/test_stream.py), ~6x faster
        from egodex.stream import stream_generate

        def sample(*x, **k):
            return stream_generate(*x, **k, cfg=a.cfg)
    else:
        assert a.cfg == 1.0, "--cfg needs the streaming sampler (causal model)"
        sample = generate

    keys = ("oracle", "context_copy", "model", "swapped", "static_head")
    rows = {k: {"psnr": [], "bg": [], **{i: {"errs": [], "found": 0, "total": 0} for i in instruments}}
            for k in keys}

    def add(key, frames, uv, valid, real=None):
        for name, fn in instruments.items():
            e, f, t = fn(frames, uv, valid)
            r = rows[key][name]
            r["errs"] += e
            r["found"] += f
            r["total"] += t
        bg = background_flow(frames, uv, valid, size_wh)
        if bg is not None:
            rows[key]["bg"].append(bg)
        if real is not None:
            rows[key]["psnr"].append(np.mean([psnr(x, y) for x, y in zip(frames[1:], real[1:])]))

    n_done = 0
    for n, idx in enumerate(pick):
        if n % ns != si - 1:  # global n: same seeds, swap partners and video names as a 1-process run
            continue
        n_done += 1
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
        gen = sample(model, lat.to(dt), *cond(b), steps=a.steps, shift=margs.get("shift", 3.0), generator=g)
        fake = decode(vae, gen, mean, std)[: len(vid_idx)]
        add("model", fake, uv, valid, real)

        # Swapped actions: another clip's hand motion from this clip's first frame. The target is
        # the other clip's joints; they won't match the scene, but the hands should follow them.
        other = int(pick[(n + 1) % len(pick)])
        b2 = ds[other]
        fi2, ci2 = ds.items[other]
        raw2 = ds._load(fi2)
        uv2, valid2 = raw2["uv"][ci2].numpy()[vid_idx], raw2["valid"][ci2].numpy()[vid_idx]
        gen2 = sample(model, lat.to(dt), *cond(b2), steps=a.steps, shift=margs.get("shift", 3.0), generator=g)
        fake2 = decode(vae, gen2, mean, std)[: len(vid_idx)]
        add("swapped", fake2, uv2, valid2)

        # Static head: same hands, no head motion. Head dims are per frame, repeated per VAE group.
        sp, ac, tx = cond(b)
        fpl = groups.shape[1]
        ac = ac.view(1, Fl, fpl, -1).clone()
        ac[..., :9] = torch.from_numpy(still).to(ac)
        gen3 = sample(model, lat.to(dt), sp, ac.view(1, Fl, -1), tx, steps=a.steps,
                        shift=margs.get("shift", 3.0), generator=g)
        fake3 = decode(vae, gen3, mean, std)[: len(vid_idx)]
        add("static_head", fake3, uv, valid)

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

    if ns > 1:
        clean = lambda x: [float(v) for v in x]  # noqa: E731
        dump = {k: {"psnr": clean(r["psnr"]), "bg": clean(r["bg"]),
                    **{i: {"errs": clean(r[i]["errs"]), "found": int(r[i]["found"]), "total": int(r[i]["total"])}
                       for i in instruments}} for k, r in rows.items()}
        (out / f"shard{si}of{ns}.json").write_text(json.dumps(
            {"rows": dump, "instruments": list(instruments), "n": n_done, "ckpt": a.ckpt, "cfg": a.cfg,
             "steps": a.steps}))
        print(f"shard {si}/{ns}: {n_done} clips", flush=True)
        return
    return finish(a, out, rows, list(instruments), len(pick))


def finish(a, out, rows, instruments, n_clips):
    res = {}
    for k, r in rows.items():
        if not r["bg"] and not r["psnr"]:
            continue
        res[k] = {}
        for name in instruments:
            res[k].update(summarize(name, r[name]["errs"], r[name]["found"], r[name]["total"]))
        if r["bg"]:
            res[k]["bg_flow_px"] = round(float(np.median(r["bg"])), 2)
        if r["psnr"]:
            res[k]["psnr"] = round(float(np.mean(r["psnr"])), 2)
    res["_info"] = {"ckpt": a.ckpt, "clips": n_clips, "splits": a.splits, "steps": a.steps, "cfg": a.cfg}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    if a.wandb_project:
        from egodex import wb
        import wandb
        run = wb.init(a.wandb_project, out, vars(a), "eval")
        run.summary.update({f"{row}/{k}": v for row, m in res.items() if not row.startswith("_") for k, v in m.items()})
        for vid in sorted(out.glob("clip*.mp4")):
            run.log({f"videos/{vid.stem}": wandb.Video(str(vid), format="mp4")})
        run.finish()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
