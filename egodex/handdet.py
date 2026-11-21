"""MediaPipe hand landmark detection (21 joints, same order as our skeleton).

Used for (1) calibrating EgoDex projections against real hand pixels and (2) evaluating
generated frames: run the detector on the output and compare to the conditioning pose.
MediaPipe's left/right labels assume mirrored selfie images, so callers should match
detections to hands by position, not by the handedness label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


class HandDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, num_hands: int = 2,
                 min_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray):
        """rgb: (H,W,3) uint8. Returns list of dicts {uv: (21,2) pixels, score: float}."""
        import mediapipe as mp

        H, W = rgb.shape[:2]
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            uv = np.array([[p.x * W, p.y * H] for p in lms], dtype=np.float64)
            out.append({"uv": uv, "score": float(hd[0].score)})
        return out

    def close(self):
        self._lm.close()


def match_to_hands(dets, proj_uv: np.ndarray, proj_valid: np.ndarray, max_dist: float):
    """Assign detections to projected hands by centroid distance (greedy).

    proj_uv: (2,21,2); proj_valid: (2,21). Returns {hand_index: detection_uv (21,2)}.
    """
    pairs = []
    for h in range(2):
        if proj_valid[h].sum() < 10:
            continue
        c = proj_uv[h][proj_valid[h]].mean(0)
        for i, d in enumerate(dets):
            pairs.append((np.linalg.norm(d["uv"].mean(0) - c), h, i))
    out, used_h, used_d = {}, set(), set()
    for dist, h, i in sorted(pairs):
        if dist > max_dist or h in used_h or i in used_d:
            continue
        out[h] = dets[i]["uv"]
        used_h.add(h)
        used_d.add(i)
    return out
