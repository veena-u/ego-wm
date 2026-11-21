"""Reading data/index.csv and choosing clip start frames that avoid tracking glitches."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_index(path: str | Path = "data/index.csv", split: str | None = None, keep_only: bool = True):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n_frames"] = int(r["n_frames"])
            r["keep"] = r["keep"] == "True"
            r["spike_frames"] = [int(x) for x in r["spike_frames"].split(";") if x]
            if keep_only and not r["keep"]:
                continue
            if split is not None and r["split"] != split:
                continue
            rows.append(r)
    return rows


def valid_clip_starts(n_frames: int, spike_frames, clip_frames: int, stride: int = 2,
                      pad: int = 1) -> np.ndarray:
    """Start frames whose clip span [start, start+span) is at least `pad` frames clear of
    every glitch frame."""
    span = (clip_frames - 1) * stride + 1
    if n_frames < span:
        return np.zeros(0, dtype=int)
    starts = np.arange(n_frames - span + 1)
    ok = np.ones(len(starts), dtype=bool)
    for s in spike_frames:
        ok &= (starts > s + pad) | (starts + span - 1 < s - pad)
    return starts[ok]
