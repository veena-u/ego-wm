"""Measure and fit the offset between projected EgoDex joints and real hand pixels.

Runs MediaPipe on sampled frames, pairs its 21 joints with our projected 21 joints, then
compares correction models by held-out residual:
  none            as recorded
  shift2d         constant pixel shift
  affine2d        2x3 affine in pixel space
  cam_offset3d    the camera that recorded the video sits at a fixed 3D offset (+ small
                  rotation) from the pose 'camera' transform - physically the most plausible
python scripts/calibrate.py ~/Downloads/test --episodes 80 --frames 6
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import Episode  # noqa: E402
from egodex.handdet import HandDetector, match_to_hands  # noqa: E402
from egodex.index import load_index  # noqa: E402


def collect(rows, frames_per_ep, seed):
    det = HandDetector()
    rng = random.Random(seed)
    P_cam, UV_proj, UV_det, EP = [], [], [], []
    for k, r in enumerate(rows):
        ep = Episode(r["path"], calib=None)  # measure against raw poses
        n = r["n_frames"]
        ts = sorted(rng.sample(range(n), min(frames_per_ep, n)))
        imgs = ep.read_frames(ts)
        uv, _ = ep.project(ts)
        pc = ep.joints_cam(ts)
        valid = ep.in_view(ts)
        for i in range(len(ts)):
            m = match_to_hands(det(imgs[i]), uv[i], valid[i], max_dist=250)
            for h, duv in m.items():
                if valid[i, h].all():
                    P_cam.append(pc[i, h]); UV_proj.append(uv[i, h]); UV_det.append(duv); EP.append(k)
    det.close()
    return np.array(P_cam), np.array(UV_proj), np.array(UV_det), np.array(EP), ep.K


# Wrist (0) and thumb base (1) are defined differently by EgoDex (Apple) and MediaPipe
# (~77 px apart even after correction), so they are excluded from fitting and scoring.
FIT_JOINTS = list(range(2, 21))


def fit_cam_offset(P, D, K, iters=300):
    P = torch.tensor(P, dtype=torch.float64).reshape(-1, 3)
    D = torch.tensor(D, dtype=torch.float64).reshape(-1, 2)
    t = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)  # axis-angle
    opt = torch.optim.LBFGS([t, w], max_iter=iters, line_search_fn="strong_wolfe")

    def rot(w):
        th = w.norm() + 1e-12
        k = w / th
        Kx = torch.zeros(3, 3, dtype=torch.float64)
        Kx[0, 1], Kx[0, 2], Kx[1, 2] = -k[2], k[1], -k[0]
        Kx = Kx - Kx.T
        return torch.eye(3, dtype=torch.float64) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx

    def proj(P):
        Q = P @ rot(w).T + t
        return torch.stack([K[0, 0] * Q[:, 0] / Q[:, 2] + K[0, 2], K[1, 1] * Q[:, 1] / Q[:, 2] + K[1, 2]], -1)

    def closure():
        opt.zero_grad()
        r = (proj(P) - D).norm(dim=-1)
        loss = torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=10.0)
        loss.backward()
        return loss

    opt.step(closure)
    R = rot(w).detach().numpy()
    tt = t.detach().numpy()
    return R, tt


def apply_cam_offset(P, R, t, K):
    Q = P @ R.T + t
    return np.stack([K[0, 0] * Q[..., 0] / Q[..., 2] + K[0, 2], K[1, 1] * Q[..., 1] / Q[..., 2] + K[1, 2]], -1)


def err(a, b):
    e = np.linalg.norm(a - b, axis=-1)
    return {"median_px": round(float(np.median(e)), 2), "mean_px": round(float(e.mean()), 2),
            "p90_px": round(float(np.percentile(e, 90)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="unused if --index given; kept for symmetry")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/calibration.json")
    a = ap.parse_args()

    rows = load_index(a.index)
    random.Random(a.seed).shuffle(rows)
    P, U, D, E, K = collect(rows[: a.episodes], a.frames, a.seed)
    print(f"matched hands: {len(P)} from {len(set(E.tolist()))} episodes")
    per_joint_P, per_joint_U, per_joint_D = P, U, D  # all 21 joints, for the per-joint report
    P, U, D = P[:, FIT_JOINTS], U[:, FIT_JOINTS], D[:, FIT_JOINTS]

    # Hold out 25% of EPISODES for evaluation.
    eps = sorted(set(E.tolist()))
    test_eps = set(eps[::4])
    te = np.array([e in test_eps for e in E])
    tr = ~te
    res = {"n_hands_train": int(tr.sum()), "n_hands_test": int(te.sum())}
    res["none"] = err(U[te], D[te])

    shift = np.median((D[tr] - U[tr]).reshape(-1, 2), 0)
    res["shift2d"] = {**err(U[te] + shift, D[te]), "shift_px": shift.round(2).tolist()}

    X = np.concatenate([U[tr].reshape(-1, 2), np.ones((U[tr].size // 2, 1))], 1)
    A, *_ = np.linalg.lstsq(X, D[tr].reshape(-1, 2), rcond=None)
    Xte = np.concatenate([U[te].reshape(-1, 2), np.ones((U[te].size // 2, 1))], 1)
    res["affine2d"] = {**err((Xte @ A).reshape(U[te].shape), D[te]), "A": A.round(4).tolist()}

    R, t = fit_cam_offset(P[tr], D[tr], K)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    res["cam_offset3d"] = {**err(apply_cam_offset(P[te], R, t, K), D[te]),
                           "t_cm": (t * 100).round(2).tolist(), "rot_deg": round(ang, 3), "R": R.round(6).tolist()}
    # per-joint error (none vs best) to see if fingertips behave differently from the wrist
    res["fit_joints"] = FIT_JOINTS
    res["per_joint_median_px_none"] = np.median(
        np.linalg.norm(per_joint_U[te] - per_joint_D[te], axis=-1), 0).round(1).tolist()
    res["per_joint_median_px_cam_offset3d"] = np.median(
        np.linalg.norm(apply_cam_offset(per_joint_P[te], R, t, K) - per_joint_D[te], axis=-1), 0).round(1).tolist()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_joint")}, indent=2))
    print("per-joint median px (none):        ", res["per_joint_median_px_none"])
    print("per-joint median px (cam_offset3d):", res["per_joint_median_px_cam_offset3d"])


if __name__ == "__main__":
    main()
