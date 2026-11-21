"""Action/conditioning encodings for the world model.

Two signals per target frame:
  * spatial: per-joint Gaussian heatmaps + per-hand depth, rasterized directly at the
    latent grid resolution (no VAE). Channel-concatenated with the noisy latent.
  * vector: head pose relative to a reference (context) frame + exact 3D joints in the
    camera frame + per-hand visibility. Fed through an MLP into AdaLN.

Hands are ABSOLUTE poses in the target frame's own camera frame: at demo time a hand tracker
provides exactly that, and errors never accumulate. Head motion is FRAME-TO-FRAME (target frame
relative to the previous sampled frame): it has no reference to a clip start, so the same action
statistics hold in an unbounded real-time session.
"""
from __future__ import annotations

import numpy as np

from .data import IMAGE_HW, NUM_JOINTS, Episode

DEPTH_MEAN, DEPTH_STD = 0.35, 0.10  # meters; measured p10/p50/p90 = 0.24/0.34/0.45
NUM_SPATIAL_CHANNELS = 2 * NUM_JOINTS + 2 + 2  # 42 joint heatmaps + 2 depth maps + 2 forearms


def _segment_dist2(px, py, a, b):
    """Squared distance from grid points (px,py) to segment a->b (each (2,) in grid units)."""
    d = b - a
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0.0, 1.0)
    return (px - (a[0] + t * d[0])) ** 2 + (py - (a[1] + t * d[1])) ** 2


def joint_heatmaps(uv: np.ndarray, z: np.ndarray, valid: np.ndarray,
                   grid_hw: tuple[int, int], sigma: float = 1.0, elbow_uv: np.ndarray | None = None,
                   elbow_ok: np.ndarray | None = None) -> np.ndarray:
    """Rasterize one frame's joints onto a latent grid.

    uv: (2,21,2) pixel coords in the 1080p image; z: (2,21) depth (m); valid: (2,21) bool.
    grid_hw: latent grid (H, W). sigma is in grid cells, so sub-cell position survives
    in the soft values. Returns float32 (46, H, W):
      [0:42]  heatmap per joint (left 0-20, right 21-41), peak 1
      [42:44] per-hand normalized depth, heatmap-weighted, 0 where the hand is absent
      [44:46] per-arm forearm: Gaussian around the wrist->elbow segment (the image clips it)
    elbow_uv: (2,2) pixels; elbow_ok: (2,) bool. Omitted -> forearm channels are zero.
    """
    H, W = grid_hw
    sy, sx = H / IMAGE_HW[0], W / IMAGE_HW[1]
    gx = uv[..., 0] * sx - 0.5  # cell-center coordinates
    gy = uv[..., 1] * sy - 0.5
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    d2 = (xs - gx[..., None, None]) ** 2 + (ys - gy[..., None, None]) ** 2  # (2,21,H,W)
    g = np.exp(-d2 / (2 * sigma ** 2)) * valid[..., None, None]
    zn = (z - DEPTH_MEAN) / DEPTH_STD
    wsum = g.sum(1)  # (2,H,W)
    depth = (g * zn[..., None, None]).sum(1) / np.maximum(wsum, 1e-6)
    depth = np.where(wsum > 1e-3, depth, 0.0)
    arms = np.zeros((2, H, W), dtype=np.float32)
    if elbow_uv is not None:
        for h in range(2):
            if elbow_ok[h] and valid[h, 0]:
                a = np.array([gx[h, 0], gy[h, 0]])
                b = np.array([elbow_uv[h, 0] * sx - 0.5, elbow_uv[h, 1] * sy - 0.5])
                arms[h] = np.exp(-_segment_dist2(xs, ys, a, b) / (2 * sigma ** 2))
    out = np.concatenate([g.reshape(2 * NUM_JOINTS, H, W), depth, arms], 0)
    return out.astype(np.float32)


def rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of a rotation matrix, flattened (continuous rotation repr)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def action_vector(ep: Episode, idx: int, prev: int, valid: np.ndarray, elbow_ok: np.ndarray) -> np.ndarray:
    """Vector action for target frame `idx`; `prev` is the previous sampled frame.

    Layout (float32, 137 dims):
      [0:6]    head rotation (6D) of frame idx relative to frame prev
      [6:9]    head translation (m) relative to frame prev
      [9:135]  joints in camera frame of idx, (2,21,3), zeroed for invalid joints
      [135:137] per-hand visibility (fraction of valid joints)
      [137:143] elbows in camera frame of idx, (2,3), zeroed when behind the camera
    """
    rel = ep.head_relative(idx, prev)
    joints = ep.joints_cam(idx) * valid[..., None]
    vis = valid.mean(-1)
    elbows = ep.elbow_cam(idx) * elbow_ok[:, None]
    return np.concatenate([rot6d(rel[:3, :3]), rel[:3, 3], joints.ravel(), vis, elbows.ravel()]).astype(np.float32)


ACTION_DIM = 6 + 3 + 2 * NUM_JOINTS * 3 + 2 + 2 * 3


def joint_validity(ep: Episode, idx, conf_thresh: float | None = None, margin_px: float = 64.0):
    """Joints that are in front of the camera and roughly in view (small off-image margin so
    heatmaps fade in at the edges). When confidences exist, confidence 0 means 'fully occluded or
    not detected' (EgoDex docs) and is always invalid; conf_thresh raises the bar further."""
    v = ep.in_view(idx, margin=margin_px)
    if ep.confidence is not None:
        v &= ep.confidence[idx] > (0.0 if conf_thresh is None else conf_thresh)
    return v


def make_clip(ep: Episode, start: int, num_frames: int, stride: int = 2,
              size_wh: tuple[int, int] = (848, 480), vae_downsample: int = 8,
              conf_thresh: float | None = None, read_video: bool = True) -> dict:
    """One training clip: frames, spatial conditioning and vector actions.

    stride=2 gives 15 fps. Head motion is relative to the previous sampled frame (for the first
    frame of the clip, the frame `stride` earlier in the episode, if any). Temporal grouping to match the VAE (e.g. 4x) happens in
    the model, so everything here is per-frame.
    """
    idx = list(range(start, start + num_frames * stride, stride))
    if idx[-1] >= min(ep.T, ep.num_video_frames()):
        raise IndexError("clip runs past end of episode")
    grid_hw = (size_wh[1] // vae_downsample, size_wh[0] // vae_downsample)
    uv, z = ep.project(idx)
    valid = joint_validity(ep, idx, conf_thresh)
    euv, ez = ep.project_points(ep.elbow_cam(idx))
    eok = ez > 0.05
    spatial = np.stack([joint_heatmaps(uv[i], z[i], valid[i], grid_hw, elbow_uv=euv[i], elbow_ok=eok[i])
                        for i in range(len(idx))])
    prev = [max(idx[0] - stride, 0)] + idx[:-1]
    actions = np.stack([action_vector(ep, t, p, valid[i], eok[i]) for i, (t, p) in enumerate(zip(idx, prev))])
    clip = {"frame_idx": np.array(idx), "spatial": spatial, "actions": actions,
            "uv": uv.astype(np.float32), "z": z.astype(np.float32), "valid": valid,
            "elbow_uv": euv.astype(np.float32), "elbow_ok": eok, "text": ep.description}
    if read_video:
        clip["frames"] = ep.read_frames(idx, size_wh)
    return clip
