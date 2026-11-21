"""Decode EgoDex frames once into a JPEG cache + labels, for training the hand pose estimator.

Labels are the projected 21 joints per hand (1080p pixels) after the pose-lag and calibration
corrections, and a validity mask: in the image AND confidence > 0 (when confidences exist).

python scripts/extract_frames.py --out /workspace/data/handpose_frames --every 6   # 5 fps
Writes <out>/<part>__<task>__<n>/<t>.jpg and <out>/labels.npz.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import Episode  # noqa: E402
from egodex.conditioning import joint_validity  # noqa: E402
from egodex.handpose import IN_WH  # noqa: E402
from egodex.index import load_index  # noqa: E402

def work(job):
    r, OUT, EVERY = job  # passed explicitly: macOS/spawn workers don't inherit globals
    p = Path(r["path"])
    key = f"{p.parent.parent.name}__{p.parent.name}__{p.name}"
    d = OUT / key
    ep = Episode(p)
    idx = [t for t in range(0, r["n_frames"], EVERY) if all(abs(t - s) > 1 for s in r["spike_frames"])]
    if not idx:
        return None
    uv, _ = ep.project(idx)
    valid = joint_validity(ep, idx, margin_px=0)
    d.mkdir(parents=True, exist_ok=True)
    frames = ep.read_frames(idx, IN_WH)
    files = []
    for t, f in zip(idx, frames):
        fn = d / f"{t}.jpg"
        if not fn.exists():
            cv2.imwrite(str(fn), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        files.append(f"{key}/{t}.jpg")
    n = len(idx)
    return files, uv.astype(np.float32), valid, [r["split"]] * n, [r["task"]] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=6, help="keep every n-th 30 fps frame")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_index(a.index)[: a.limit]
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(work, [(r, OUT, a.every) for r in rows], chunksize=4) if r is not None]
    files, uv, valid, split, task = (sum((list(x[i]) for x in res), []) if i in (0, 3, 4)
                                     else np.concatenate([x[i] for x in res]) for i in range(5))
    np.savez(OUT / "labels.npz", files=np.array(files), uv=uv, valid=valid, split=np.array(split),
             task=np.array(task))
    s = np.array(split)
    print(f"{len(files)} frames: " + ", ".join(f"{k} {(s == k).sum()}" for k in np.unique(s)))


if __name__ == "__main__":
    main()
