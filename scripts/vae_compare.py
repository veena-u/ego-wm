"""Compare 8x video VAEs on EgoDex, focusing on the hands.

A world model can't generate finger detail its VAE can't reconstruct, so this measures the
ceiling each base model imposes. For each clip (17 frames, 15 fps, 848x480):
  psnr_full    PSNR over the whole frame
  psnr_hand    PSNR inside boxes around the projected hands
  lpips_hand   LPIPS on hand crops (lower = better)
  kp_err_px    MediaPipe landmarks on reconstruction vs on original (px @480p)
  kp_recall    fraction of hands MediaPipe finds in the original that it still finds after the VAE

python scripts/vae_compare.py --clips 12 --device mps
"""
import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import Episode, make_clip  # noqa: E402
from egodex.handdet import HandDetector  # noqa: E402
from egodex.index import load_index, valid_clip_starts  # noqa: E402
from egodex.vae import VAES, from_video_tensor, load_vae, to_video_tensor  # noqa: E402

W, H, NF = 848, 480, 17


@torch.no_grad()
def roundtrip(vae, frames, device):
    z = vae.encode(to_video_tensor(frames, device)).latent_dist.mode()
    y = from_video_tensor(vae.decode(z).sample)
    return y[: len(frames)], tuple(z.shape)


def hand_boxes(uv, valid, margin=20):
    boxes = []
    for h in range(2):
        if valid[h].sum() < 10:
            continue
        p = uv[h][valid[h]] * [W / 1920, H / 1080]
        x0, y0 = np.floor(p.min(0) - margin).astype(int)
        x1, y1 = np.ceil(p.max(0) + margin).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 > 16 and y1 - y0 > 16:
            boxes.append((x0, y0, x1, y1))
    return boxes


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def kp_compare(det, orig, recon):
    errs, found, total = [], 0, 0
    for o, r in zip(orig, recon):
        do, dr = det(o), det(r)
        for d in do:
            total += 1
            if not dr:
                continue
            c = d["uv"].mean(0)
            best = min(dr, key=lambda e: np.linalg.norm(e["uv"].mean(0) - c))
            if np.linalg.norm(best["uv"].mean(0) - c) < 60:
                found += 1
                errs.append(np.linalg.norm(best["uv"] - d["uv"], axis=-1).mean())
    return errs, found, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--vaes", nargs="+", default=["bicubic8x"] + list(VAES),
                    help=f"any of bicubic8x {' '.join(VAES)}")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="data/vae_compare")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False)
    det = HandDetector()

    # Fixed clip set from validation episodes, preferring clips where both hands are visible.
    rows = [r for r in load_index(a.index) if r["split"] != "train"]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    clips = []
    for r in rows:
        starts = valid_clip_starts(r["n_frames"], r["spike_frames"], NF)
        if not len(starts):
            continue
        ep = Episode(r["path"])
        c = make_clip(ep, int(rng.choice(starts)), NF, size_wh=(W, H))
        if c["valid"].mean() > 0.6:
            c["task"] = r["task"]
            clips.append(c)
        if len(clips) == a.clips:
            break
    print(f"{len(clips)} clips: {[c['task'] for c in clips]}")

    results, crops_vis = {}, {}
    for name in a.vaes:
        t0 = time.time()
        try:
            vae = load_vae(name, a.device, tiling=a.device != "cuda")  # tiles add seams
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] load failed: {e}")
            continue
        m = {"psnr_full": [], "psnr_hand": [], "lpips_hand": [], "kp_err": [], "found": 0, "total": 0, "sec": []}
        for ci, c in enumerate(clips):
            ts = time.time()
            try:
                rec, zshape = roundtrip(vae, c["frames"], a.device)
            except Exception as e:  # noqa: BLE001
                print(f"[{name}] clip {ci} failed on {a.device}: {e}")
                break
            m["sec"].append(time.time() - ts)
            orig = c["frames"]
            m["psnr_full"].append(np.mean([psnr(o, r) for o, r in zip(orig, rec)]))
            for t in range(NF):
                for (x0, y0, x1, y1) in hand_boxes(c["uv"][t], c["valid"][t]):
                    o, r = orig[t, y0:y1, x0:x1], rec[t, y0:y1, x0:x1]
                    m["psnr_hand"].append(psnr(o, r))
                    to = lambda im: torch.from_numpy(cv2.resize(im, (128, 128))).permute(2, 0, 1)[None].float() / 127.5 - 1  # noqa: E731
                    m["lpips_hand"].append(float(lp(to(o), to(r))))
                    if ci == 0 and t == NF // 2 and name not in crops_vis:
                        crops_vis.setdefault("original", o)
                        crops_vis[name] = r
            errs, f, tot = kp_compare(det, orig[::4], rec[::4])
            m["kp_err"] += errs
            m["found"] += f
            m["total"] += tot
        if not m["sec"]:
            continue
        results[name] = {
            "latent_shape": list(zshape),
            "psnr_full": round(float(np.mean(m["psnr_full"])), 2),
            "psnr_hand": round(float(np.mean(m["psnr_hand"])), 2),
            "lpips_hand": round(float(np.mean(m["lpips_hand"])), 4),
            "kp_err_px": round(float(np.median(m["kp_err"])), 2) if m["kp_err"] else None,
            "kp_recall": round(m["found"] / max(m["total"], 1), 3),
            "sec_per_clip": round(float(np.mean(m["sec"])), 1),
            "load_plus_total_sec": round(time.time() - t0, 1),
        }
        print(name, json.dumps(results[name]))
        del vae
        gc.collect()
        if a.device == "mps":
            torch.mps.empty_cache()
    det.close()
    (out / "results.json").write_text(json.dumps(results, indent=2))

    if crops_vis:
        hh = 256
        tiles = []
        for k, im in crops_vis.items():
            im = cv2.resize(im, (int(im.shape[1] * hh / im.shape[0]), hh), interpolation=cv2.INTER_NEAREST)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.putText(im, k, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(im)
        cv2.imwrite(str(out / "hand_crops.jpg"), np.concatenate(tiles, 1))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
