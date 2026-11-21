"""EgoDex episode loading.

Each episode is a pair <n>.hdf5 + <n>.mp4 (1920x1080, 30 fps). The HDF5 holds:
  camera/intrinsic      (3,3)      fixed across the dataset
  transforms/camera     (T,4,4)    camera-to-world, OpenCV convention (+z forward, +y down)
  transforms/<joint>    (T,4,4)    joint-to-world
  confidences/<joint>   (T,)       tracking confidence (missing in ~13% of episodes)

We keep 21 joints per hand in MediaPipe order (wrist, thumb x4, index/middle/ring/little x4),
dropping EgoDex's 4 metacarpals, so live hand trackers map onto the same skeleton.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

SIDES = ("left", "right")
_SEG = ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
FINGERS = ("Index", "Middle", "Ring", "Little")


def hand_joint_names(side: str) -> list[str]:
    names = [f"{side}Hand"]
    names += [f"{side}Thumb{s}" for s in _SEG]
    for fg in FINGERS:
        names += [f"{side}{fg}Finger{s}" for s in _SEG]
    return names


# Bones in MediaPipe 21-joint indexing.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4)]
for _base in (5, 9, 13, 17):
    BONES += [(0, _base), (_base, _base + 1), (_base + 1, _base + 2), (_base + 2, _base + 3)]
NUM_JOINTS = 21
IMAGE_HW = (1080, 1920)


def _py(v):
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode()
    return v


class Episode:
    """One EgoDex recording. All poses are float64; joints are (T, 2 hands, 21, 3)."""

    def __init__(self, path: str | Path, pose_lag: int = 1, calib="default"):
        """pose_lag: EgoDex poses trail the video by ~1 frame (scripts/sync_check.py measured
        best lag 1 in 24/40 episodes, 2 in 12, 0 in 3). Video frame t is paired with stored
        pose t+pose_lag; the last pose_lag video frames are dropped (self.T shrinks).

        calib: fixed extrinsic correction (R, t) from the pose 'camera' to the camera that
        actually recorded the video, fit by scripts/calibrate.py against MediaPipe (~3 cm
        backwards, ~1.8 deg). "default" loads data/calibration.json if present; None = raw.
        Applied in joints_cam(), so both heatmaps and vector actions use it."""
        p = Path(path)
        self.pose_lag = pose_lag
        self.calib = load_calibration() if calib == "default" else calib
        self.h5_path = p.with_suffix(".hdf5")
        self.mp4_path = p.with_suffix(".mp4")
        with h5py.File(self.h5_path, "r") as f:
            self.K = f["camera/intrinsic"][()].astype(np.float64)
            self.cam_to_world = f["transforms/camera"][()].astype(np.float64)
            joints, conf = [], []
            has_conf = "confidences" in f
            for side in SIDES:
                names = hand_joint_names(side)
                joints.append(np.stack([f[f"transforms/{n}"][:, :3, 3] for n in names], 1))
                if has_conf:
                    conf.append(np.stack([f[f"confidences/{n}"][()] for n in names], 1))
            self.joints_world = np.stack(joints, 1).astype(np.float64)
            # 'Forearm' is the elbow (~26.5 cm from the wrist); the elbow itself is almost never in
            # view, but the wrist->elbow segment is the forearm you see in nearly every frame.
            self.elbow_world = np.stack([f[f"transforms/{s}Forearm"][:, :3, 3] for s in SIDES], 1).astype(np.float64)
            self.confidence = np.stack(conf, 1) if has_conf else None  # (T,2,21) or None
            self.attrs = {k: _py(v) for k, v in f.attrs.items()}
        if pose_lag:
            self.cam_to_world = self.cam_to_world[pose_lag:]
            self.joints_world = self.joints_world[pose_lag:]
            self.elbow_world = self.elbow_world[pose_lag:]
            if self.confidence is not None:
                self.confidence = self.confidence[pose_lag:]
        self.T = len(self.cam_to_world)
        self.world_to_cam = np.linalg.inv(self.cam_to_world)

    # ---------- geometry ----------
    def joints_cam(self, idx=None) -> np.ndarray:
        """Joints in the camera frame of the same timestep. (T,2,21,3) or (2,21,3)."""
        idx = slice(None) if idx is None else idx
        W = self.world_to_cam[idx]
        P = self.joints_world[idx]
        pc = np.einsum("...ij,...hnj->...hni", W[..., :3, :3], P) + W[..., None, None, :3, 3]
        if self.calib is not None:
            R, t = self.calib
            pc = pc @ R.T + t
        return pc

    def elbow_cam(self, idx=None) -> np.ndarray:
        """Elbows in the camera frame: (T,2,3) or (2,3)."""
        idx = slice(None) if idx is None else idx
        W = self.world_to_cam[idx]
        pc = np.einsum("...ij,...hj->...hi", W[..., :3, :3], self.elbow_world[idx]) + W[..., None, :3, 3]
        if self.calib is not None:
            R, t = self.calib
            pc = pc @ R.T + t
        return pc

    def project_points(self, pc):
        z = pc[..., 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return np.stack([self.K[0, 0] * pc[..., 0] / zs + self.K[0, 2], self.K[1, 1] * pc[..., 1] / zs + self.K[1, 2]], -1), z

    @property
    def description(self) -> str:
        """Language annotation that applies to THIS episode: reversible tasks carry two
        descriptions and 'which_llm_description' picks one (27% of episodes use the second)."""
        a = self.attrs
        d = a.get("llm_description2") if str(a.get("which_llm_description", "1")) == "2" else a.get("llm_description")
        if not d or d == "None":
            d = a.get("llm_description") or a.get("description", "")
        return d

    def project(self, idx=None):
        """Returns uv (...,2,21,2) in 1080p pixels and depth z (...,2,21) in meters."""
        pc = self.joints_cam(idx)
        z = pc[..., 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.K[0, 0] * pc[..., 0] / zs + self.K[0, 2]
        v = self.K[1, 1] * pc[..., 1] / zs + self.K[1, 2]
        return np.stack([u, v], -1), z

    def in_view(self, idx=None, margin=0.0) -> np.ndarray:
        """Per-joint mask: in front of the camera and inside the image (+margin px)."""
        uv, z = self.project(idx)
        H, W = IMAGE_HW
        return (z > 0.05) & (uv[..., 0] >= -margin) & (uv[..., 0] < W + margin) \
            & (uv[..., 1] >= -margin) & (uv[..., 1] < H + margin)

    def head_relative(self, idx, ref) -> np.ndarray:
        """Pose of the (calibrated) video camera at `idx` expressed in its frame at `ref` (4x4).
        With correction T (pose-camera -> video-camera), the video camera's world pose is
        C @ inv(T), so the relative pose is T @ inv(C_ref) @ C_idx @ inv(T)."""
        rel = self.world_to_cam[ref] @ self.cam_to_world[idx]
        if self.calib is not None:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = self.calib
            rel = T @ rel @ np.linalg.inv(T)
        return rel

    # ---------- quality ----------
    def spike_frames(self, thresh_m=0.03) -> np.ndarray:
        """Frames whose wrist acceleration exceeds thresh (m/frame^2, ~2.7 g at 30 fps):
        tracking glitches. Returns sorted unique frame indices."""
        P = self.joints_world[:, :, 0]  # (T,2,3)
        if self.T < 3:
            return np.zeros(0, dtype=int)
        acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1)  # (T-2,2), centered at t+1
        return np.unique(np.nonzero(acc > thresh_m)[0]) + 1

    def wrist_spikes(self, thresh_m=0.03) -> int:
        return len(self.spike_frames(thresh_m))

    # ---------- video ----------
    def num_video_frames(self) -> int:
        cap = cv2.VideoCapture(str(self.mp4_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def read_frames(self, indices, size_wh: tuple[int, int] | None = None) -> np.ndarray:
        """Read frames (RGB uint8) at sorted-or-not indices by decoding sequentially."""
        indices = list(indices)
        wanted = set(indices)
        last = max(indices)
        cap = cv2.VideoCapture(str(self.mp4_path))
        start = min(indices)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        got = {}
        t = start
        while t <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if t in wanted:
                if size_wh is not None:
                    frame = cv2.resize(frame, size_wh, interpolation=cv2.INTER_AREA)
                got[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t += 1
        cap.release()
        missing = [i for i in indices if i not in got]
        if missing:
            raise IndexError(f"{self.mp4_path}: could not decode frames {missing[:5]}...")
        return np.stack([got[i] for i in indices])


CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "data" / "calibration.json"
_CALIB_CACHE = {}


def load_calibration(path: str | Path = CALIBRATION_PATH):
    """(R, t) of the 'cam_offset3d' model in calibration.json, or None if the file is absent."""
    path = Path(path)
    if path not in _CALIB_CACHE:
        if not path.exists():
            _CALIB_CACHE[path] = None
        else:
            import json
            c = json.loads(path.read_text())["cam_offset3d"]
            _CALIB_CACHE[path] = (np.array(c["R"], dtype=np.float64), np.array(c["t_cm"], dtype=np.float64) / 100)
    return _CALIB_CACHE[path]


def list_episodes(root: str | Path) -> list[Path]:
    """All episode stems (without extension) under an EgoDex split folder."""
    root = Path(root)
    return sorted(p.with_suffix("") for p in root.glob("*/*.hdf5") if p.with_suffix(".mp4").exists())
