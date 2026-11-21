"""Estimate the time offset between hand poses and video.

For each frame t, compare the pose-predicted 2D velocity of the hand joints, (uv[t+1+L]-uv[t+L]),
with the optical-flow velocity sampled at those joints in frames t->t+1. The lag L with the
best agreement is the pose/video offset. L>0 means the pose describes motion that appears in
the video L frames earlier, i.e. the poses are delayed relative to video.

python scripts/sync_check.py ~/Downloads/test --episodes 40
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import Episode, list_episodes  # noqa: E402

FLOW_W, FLOW_H = 480, 270


def flow_series(ep, n):
    frames = ep.read_frames(range(n), (FLOW_W, FLOW_H))
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    return [cv2.calcOpticalFlowFarneback(gray[t], gray[t + 1], None, 0.5, 4, 21, 3, 7, 1.5, 0)
            for t in range(n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-lag", type=int, default=4)
    ap.add_argument("--min-speed", type=float, default=2.0, help="px/frame at flow res")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-lag", type=int, default=0, help="loader correction to apply (0 = raw data)")
    a = ap.parse_args()

    eps = list_episodes(a.root)
    random.Random(a.seed).shuffle(eps)
    lags = list(range(-a.max_lag, a.max_lag + 1))
    err = {L: [] for L in lags}
    cos = {L: [] for L in lags}
    per_ep = []
    s = np.array([FLOW_W / 1920, FLOW_H / 1080])
    for path in eps[: a.episodes]:
        ep = Episode(path, pose_lag=a.pose_lag)
        n = min(ep.T, ep.num_video_frames(), 450)
        if n < 2 * a.max_lag + 10:
            continue
        flows = flow_series(ep, n)
        uv, _ = ep.project(range(n))
        uv = uv * s
        valid = ep.in_view(range(n))
        ep_err = {L: [] for L in lags}
        for t in range(a.max_lag, n - 1 - a.max_lag):
            f = flows[t]
            for h in range(2):
                m = valid[t, h]
                if m.sum() < 10:
                    continue
                pts = uv[t, h][m]
                xi = np.clip(pts[:, 0].round().astype(int), 0, FLOW_W - 1)
                yi = np.clip(pts[:, 1].round().astype(int), 0, FLOW_H - 1)
                fv = np.median(f[yi, xi], 0)
                if np.linalg.norm(fv) < a.min_speed:
                    continue
                for L in lags:
                    pv = np.median((uv[t + 1 + L, h] - uv[t + L, h])[m], 0)
                    e = np.linalg.norm(pv - fv)
                    err[L].append(e)
                    ep_err[L].append(e)
                    cos[L].append(pv @ fv / (np.linalg.norm(pv) * np.linalg.norm(fv) + 1e-6))
        if ep_err[0]:
            per_ep.append(min(lags, key=lambda L: np.median(ep_err[L])))
    print(f"samples per lag: {len(err[0])} from {len(per_ep)} episodes")
    print(" lag  median|v_pose - v_flow| (px@480w)  mean cos")
    for L in lags:
        print(f"{L:4d}  {np.median(err[L]):8.3f}                     {np.mean(cos[L]):.3f}")
    best = min(lags, key=lambda L: np.median(err[L]))
    print(f"best lag overall: {best}")
    vals, counts = np.unique(per_ep, return_counts=True)
    print("per-episode best lag histogram:", dict(zip(vals.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
