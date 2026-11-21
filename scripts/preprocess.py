"""Encode EgoDex into training-ready files: one .pt per episode.

Each episode is cut into non-overlapping clips that avoid tracking glitches. Every clip is
encoded by the VAE on its own, so its first latent is a single-frame latent - exactly what
the model sees at inference when it starts from one real image.

Per episode file (torch.save dict):
  latents   (N, C, 1+(F-1)/4, H/8, W/8) float16, normalized with the VAE's latent mean/std
  starts    (N,) clip start frame (30 fps index)
  uv        (N, F, 2, 21, 2) float32  joint pixels in the 1080p image
  z         (N, F, 2, 21)    float32  joint depth (m)
  valid     (N, F, 2, 21)    bool
  elbow_uv  (N, F, 2, 2)     float32  elbow pixels;  elbow_ok (N, F, 2) bool (in front of camera)
  actions   (N, F, 143)      float32  raw (unnormalized) vector actions
  text, task, split, path
Heatmaps are NOT stored (~20x larger than latents); the dataset rebuilds them from uv/z/valid.

python scripts/preprocess.py --vae wan2.1 --out /data/egodex_latents --shard 1/4
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import Episode, make_clip  # noqa: E402
from egodex.index import load_index, valid_clip_starts  # noqa: E402
from egodex.vae import latent_norm, load_vae, to_video_tensor  # noqa: E402


def clip_starts(n_frames, spikes, clip_frames, stride):
    """Greedy non-overlapping clip starts among glitch-free starts."""
    span = (clip_frames - 1) * stride + 1
    starts, nxt = [], 0
    for s in valid_clip_starts(n_frames, spikes, clip_frames, stride):
        if s >= nxt:
            starts.append(int(s))
            nxt = s + span
    return starts


@torch.no_grad()
def encode(vae, frames, device, dtype, mean, std):
    z = vae.encode(to_video_tensor(frames, device, dtype)).latent_dist.mode().float()
    if mean is not None:
        shape = (1, -1) + (1,) * (z.ndim - 2)
        z = (z - mean.to(z.device).view(shape)) / std.to(z.device).view(shape)
    return z[0].half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default="wan2.1")
    ap.add_argument("--clip-frames", type=int, default=33, help="1+4k; 33 = 9 latents = 2.2 s at 15 fps")
    ap.add_argument("--stride", type=int, default=2, help="30 fps / stride")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--shard", default="1/1", help="i/n: process every n-th episode starting at i")
    ap.add_argument("--limit", type=int, default=None, help="max episodes (for testing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    a = ap.parse_args()

    i, n = map(int, a.shard.split("/"))
    rows = load_index(a.index)[i - 1::n][: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if a.dtype == "bf16" and a.device == "cuda" else torch.float32
    vae = load_vae(a.vae, a.device, dtype=dtype, tiling=False)
    mean, std = latent_norm(vae)

    t0, done, clips_total = time.time(), 0, 0
    for r in rows:
        ep_path = Path(r["path"])  # <part>/<task>/<n>: the part is needed, ids repeat across parts
        dst = out / f"{ep_path.parent.parent.name}__{ep_path.parent.name}__{ep_path.name}.pt"
        if dst.exists():
            continue
        try:
            starts = clip_starts(r["n_frames"], r["spike_frames"], a.clip_frames, a.stride)
            if not starts:
                continue
            ep = Episode(r["path"])
            keys = ("uv", "z", "valid", "elbow_uv", "elbow_ok", "actions")
            rec = {k: [] for k in ("latents",) + keys}
            for s in starts:
                c = make_clip(ep, s, a.clip_frames, a.stride, (a.width, a.height))
                rec["latents"].append(encode(vae, c["frames"], a.device, dtype, mean, std))
                for k in keys:
                    rec[k].append(c[k])
            item = {
                "latents": torch.stack(rec["latents"]),
                "starts": torch.tensor(starts),
                **{k: torch.from_numpy(np.stack(rec[k])) for k in keys},
                "text": c["text"], "task": r["task"], "split": r["split"], "path": r["path"],
                "meta": {"vae": a.vae, "clip_frames": a.clip_frames, "stride": a.stride,
                         "size_wh": (a.width, a.height)},
            }
            tmp = dst.with_suffix(".tmp")
            torch.save(item, tmp)
            tmp.rename(dst)  # atomic: a crash never leaves a half-written file that looks done
            done += 1
            clips_total += len(starts)
        except Exception:  # noqa: BLE001
            print(f"FAILED {r['path']}\n{traceback.format_exc()}", flush=True)
        if done and done % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(rows)} episodes, {clips_total} clips, {clips_total / el:.2f} clips/s", flush=True)
    print(f"done: {done} episodes, {clips_total} clips in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
