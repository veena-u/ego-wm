"""Long live-session test on the GPU: speed per step and a video, without a webcam.

Starts from a scene and drives it with another clip's recorded hand motion (played forward and
backward so it is continuous for as long as needed), through the same LiveSession the demo uses,
so re-anchoring over long sessions is exercised.
  python scripts/bench_live.py --ckpt runs/v2/model_bf16_step8000.pt --data /workspace/data/latents \
      --text-emb /workspace/data/text_emb.pt --steps-list 10 30 --n 40 --out runs/bench_live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_server import Demo  # noqa: E402
from egodex.live import pose_dict  # noqa: E402


def clip_poses(demo, k):
    """All 33 video-frame poses of scene k's clip, as pose dicts."""
    idx = demo.scenes[k]
    fi, ci = demo.ds.items[idx]
    d = demo.ds._load(fi)
    act, valid = d["actions"][ci].numpy(), d["valid"][ci].numpy()
    return [pose_dict(a[9:135].reshape(2, 21, 3).astype(np.float64), v.any(-1), demo.K) for a, v in zip(act, valid)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--motion", type=int, default=7, help="scene whose hand motion drives it")
    ap.add_argument("--n", type=int, default=40, help="model steps (each = fpl frames)")
    ap.add_argument("--steps-list", type=int, nargs="+", default=[10])
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    demo = Demo(a.ckpt, a.data, a.text_emb)
    frame0, p0, text = demo.scene(a.scene, n=len(demo.scenes))
    motion = clip_poses(demo, a.motion)
    motion = motion + motion[::-1][1:-1]  # ping-pong: continuous forever
    fpl = demo.model.fpl
    res = {}
    for steps in ([None] if demo.student else a.steps_list):
        sess = demo.session(text, a.cfg, steps or 10)
        sess.start([frame0], [p0])
        H, W = frame0.shape[:2]
        tag = "student" if steps is None else f"steps{steps}"
        vw = cv2.VideoWriter(str(out / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
        vw.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
        times = []
        for i in range(a.n):
            poses = [motion[(i * fpl + j) % len(motion)] for j in range(fpl)]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            frames = sess.step(poses)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        warm = times[2:] or times
        ms = 1000 * float(np.median(warm))
        res[tag] = {"ms_per_step": round(ms, 1), "fps": round(fpl / (ms / 1000), 2),
                    "realtime_x": round(fpl / (ms / 1000) / 15, 3), "reanchors": sess.reanchors,
                    "seconds_of_video": round(a.n * fpl / 15, 1)}
        print(tag, json.dumps(res[tag]), flush=True)
    res["_info"] = {"ckpt": a.ckpt, "scene": text, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    (out / "bench.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
