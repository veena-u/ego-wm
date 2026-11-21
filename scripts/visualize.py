"""Render a side-by-side check video: skeleton overlay | latent-grid heatmaps over the frame.

python scripts/visualize.py ~/Downloads/test/basic_pick_place/0 out.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import BONES, Episode  # noqa: E402
from egodex.conditioning import joint_heatmaps, joint_validity  # noqa: E402

COLORS = {0: (255, 120, 0), 1: (0, 80, 255)}  # BGR: left blue-ish, right red-ish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    ep = Episode(a.episode)
    n = min(ep.T, ep.num_video_frames())
    idx = list(range(0, n, a.stride))
    W, H = a.width, a.height
    frames = ep.read_frames(idx, (W, H))
    uv, z = ep.project(idx)
    valid = joint_validity(ep, idx)
    euv, ez = ep.project_points(ep.elbow_cam(idx))
    eok = ez > 0.05
    sx, sy = W / 1920, H / 1080
    grid = (H // 8, W // 8)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30 / a.stride, (2 * W, H))
    for i in range(len(idx)):
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        left = img.copy()
        for h in range(2):
            pts = uv[i, h] * [sx, sy]
            for a_, b_ in BONES:
                if valid[i, h, a_] and valid[i, h, b_]:
                    cv2.line(left, tuple(pts[a_].astype(int)), tuple(pts[b_].astype(int)), COLORS[h], 2)
        hm = joint_heatmaps(uv[i], z[i], valid[i], grid, elbow_uv=euv[i], elbow_ok=eok[i])
        heat = np.clip(hm[:42].sum(0) + 0.5 * hm[44:46].sum(0), 0, 1)
        heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)
        heat_c = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        right = cv2.addWeighted(img, 0.5, heat_c, 0.5, 0)
        cv2.putText(right, f"latent grid {grid[1]}x{grid[0]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(np.concatenate([left, right], 1))
    vw.release()
    print(f"wrote {a.out}: {len(idx)} frames, task={ep.attrs.get('task')}, '{ep.attrs.get('description')}'")


if __name__ == "__main__":
    main()
