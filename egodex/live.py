"""Live hand input for the demo: webcam -> MediaPipe -> EgoDex-format conditioning.

The model was trained on EgoDex poses: 21 joints per hand (MediaPipe order) in the camera frame
of a head-mounted camera with EgoDex's intrinsics (1920x1080). A webcam gives MediaPipe's
2D landmarks (pixels) plus metric 3D "world" landmarks (meters, hand-centered). We convert:

  pose      the world landmarks give the hand's metric SHAPE (2.5 cm mean joint error vs EgoDex
            after alignment) but not its orientation (off by a median 53 deg from the camera
            axes, measured on EgoDex frames). So the full 3D pose comes from PnP: fit that rigid
            shape to the 2D landmarks with the webcam's intrinsics (from its field of view),
            which also gives the hand's distance from the camera.
  elbow     not visible to a hand tracker: placed 26.5 cm behind the wrist along the hand's
            long axis (EgoDex's measured forearm length), enough for the forearm channel.

Two camera setups:
  ego     the webcam looks out from roughly your eyes (head-mounted, or held at the brow
          pointing at your hands): like EgoDex, no change of viewpoint.
  selfie  a laptop webcam facing you. The image is mirrored (so it works like a mirror), and
          the hands are re-expressed as seen from your
          own eyes: x stays (after the mirror), z flips, and moving a hand toward the screen
          moves it away from you.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .conditioning import ACTION_DIM, DEPTH_MEAN, joint_heatmaps
from .data import IMAGE_HW, NUM_JOINTS

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
FOREARM_M = 0.265
MIDDLE_MCP = 9
MP_SCALE = 0.95  # EgoDex / MediaPipe hand size (median over 99 hands): MediaPipe's hands are ~5% large


def egodex_intrinsics() -> np.ndarray:
    """EgoDex camera intrinsics: identical in every file (checked), so fixed here."""
    return np.array([[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]])


class LiveHands:
    def __init__(self, view: str = "selfie", webcam_hfov_deg: float = 65.0, K=None,
                 smooth: float = 0.5, model_path=DEFAULT_MODEL, min_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        assert view in ("ego", "selfie")
        self.view, self.hfov, self.smooth = view, webcam_hfov_deg, smooth
        self.K = egodex_intrinsics() if K is None else np.asarray(K, np.float64)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._t_ms = 0
        self._ref_z = None  # webcam depth that maps to DEPTH_MEAN (set from the first detections)
        self._ref_samples: list[float] = []
        self._prev = None

    def mirror(self, rgb):
        return np.ascontiguousarray(rgb[:, ::-1]) if self.view == "selfie" else rgb

    def __call__(self, rgb: np.ndarray, t_ms: int | None = None) -> dict:
        """rgb: webcam frame (H,W,3) uint8, NOT mirrored (this mirrors it in selfie mode).
        Returns a pose dict in the EgoDex camera frame (see pose_dict)."""
        import mediapipe as mp

        img = self.mirror(rgb)
        H, W = img.shape[:2]
        self._t_ms = self._t_ms + 33 if t_ms is None else max(t_ms, self._t_ms + 1)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), self._t_ms)
        f = W / (2 * np.tan(np.radians(self.hfov) / 2))
        joints = np.zeros((2, NUM_JOINTS, 3))
        found = np.zeros(2, bool)
        # MediaPipe labels the hand shape it sees. Unmirrored egocentric frames: the label is the
        # true hand (measured on EgoDex: swapping gave 2% correct). A mirrored image flips a
        # hand's chirality, so in selfie mode the label is the opposite hand.
        for lms, wlms, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            label = hd[0].category_name  # "Left" / "Right"
            side = (1 if label == "Left" else 0) if self.view == "selfie" else (0 if label == "Left" else 1)
            if found[side]:
                continue
            uv = np.array([[p.x * W, p.y * H] for p in lms])
            wl = np.array([[p.x, p.y, p.z] for p in wlms])
            P = self._to_ego(uv, wl, f, W, H)
            if P is not None:
                joints[side], found[side] = P, True
        return self._finish(joints, found)

    def _to_ego(self, uv, wl, f, W, H):
        """One hand: webcam pixels + metric landmarks -> 21 joints in the EgoDex camera frame."""
        import cv2

        Kw = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        obj = (wl - wl[0]) * MP_SCALE
        ok, rvec, tvec = cv2.solvePnP(obj, uv, Kw, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        P = obj @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()  # joints in the webcam's camera frame
        if self.view == "ego":
            return P  # the webcam is (roughly) your eye
        # Selfie: the image was mirrored, so x already matches your own left/right. Distance from
        # the screen becomes distance from your eyes, reversed: your resting distance maps to
        # EgoDex's typical 0.35 m and pushing a hand toward the screen moves it away from you.
        if self._ref_z is None:
            self._ref_samples.append(float(P[0, 2]))
            if len(self._ref_samples) >= 15:  # ~0.5 s of hands at rest
                self._ref_z = float(np.median(self._ref_samples))
        ref = self._ref_z or float(np.median(self._ref_samples))
        P = P.copy()
        P[:, 2] = DEPTH_MEAN + ref - P[:, 2]
        shift = np.clip(P[0, 2], 0.15, 0.7) - P[0, 2]  # keep the wrist at a plausible depth
        P[:, 2] += shift
        return P

    def _finish(self, joints, found):
        if self._prev is not None and self.smooth > 0:  # light smoothing of hands seen in both frames
            pj, pf = self._prev
            both = found & pf
            joints[both] = self.smooth * pj[both] + (1 - self.smooth) * joints[both]
        self._prev = (joints.copy(), found.copy())
        return pose_dict(joints, found, self.K)


def pose_dict(joints: np.ndarray, found: np.ndarray, K: np.ndarray) -> dict:
    """joints (2,21,3) camera frame, found (2,) -> everything the model conditioning needs."""
    z = joints[..., 2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = np.stack([K[0, 0] * joints[..., 0] / zs + K[0, 2], K[1, 1] * joints[..., 1] / zs + K[1, 2]], -1)
    H, W, m = IMAGE_HW[0], IMAGE_HW[1], 64.0  # same validity rule as training (joint_validity)
    valid = found[:, None] & (z > 0.05) & (uv[..., 0] >= -m) & (uv[..., 0] < W + m) \
        & (uv[..., 1] >= -m) & (uv[..., 1] < H + m)
    d = joints[:, 0] - joints[:, MIDDLE_MCP]
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
    elbows = joints[:, 0] + FOREARM_M * d
    ez = elbows[:, 2]
    ezs = np.where(np.abs(ez) < 1e-6, 1e-6, ez)
    elbow_uv = np.stack([K[0, 0] * elbows[:, 0] / ezs + K[0, 2], K[1, 1] * elbows[:, 1] / ezs + K[1, 2]], -1)
    elbow_ok = found & (ez > 0.05)
    return {"joints": joints, "uv": uv, "z": z, "valid": valid, "found": found,
            "elbows": elbows, "elbow_uv": elbow_uv, "elbow_ok": elbow_ok}


class Conditioner:
    """Pose dicts (one per video frame, 15 fps) -> per-latent model conditioning, exactly as
    LatentClips builds it in training: heatmaps at the latent grid, normalized action vector with
    the head held still, invalid joints zeroed after normalization, fpl frames per latent."""

    def __init__(self, action_stats="data/action_stats.npz", grid_hw=(60, 106), fpl=4):
        st = np.load(action_stats)
        self.mean, self.std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
        self.grid_hw, self.fpl = grid_hw, fpl
        # head: identity rotation (6D) and zero translation = "head still"
        self.head = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], np.float32)

    def frame(self, p: dict):
        """One video frame -> (spatial (46,h,w), action (143,))."""
        sp = joint_heatmaps(p["uv"], p["z"], p["valid"], self.grid_hw, 1.0, p["elbow_uv"], p["elbow_ok"])
        raw = np.concatenate([self.head, (p["joints"] * p["valid"][..., None]).ravel(), p["valid"].mean(-1),
                              (p["elbows"] * p["elbow_ok"][:, None]).ravel()]).astype(np.float32)
        assert raw.shape == (ACTION_DIM,)
        a = (raw - self.mean) / self.std
        j = a[9:135].reshape(2, NUM_JOINTS, 3)
        j[~p["valid"]] = 0.0
        a[9:135] = j.ravel()
        return sp, a

    def latent(self, poses: list[dict]):
        """fpl pose dicts (or 1 for the first latent, which the VAE builds from one frame, repeated)
        -> spatial (46*fpl, h, w), actions (fpl*143,), as float32 numpy."""
        if len(poses) == 1:
            poses = poses * self.fpl
        assert len(poses) == self.fpl
        fr = [self.frame(p) for p in poses]
        return np.concatenate([s for s, _ in fr]), np.concatenate([a for _, a in fr])
