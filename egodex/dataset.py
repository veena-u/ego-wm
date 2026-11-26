"""Training dataset over the per-episode files written by scripts/preprocess.py."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditioning import NUM_SPATIAL_CHANNELS, joint_heatmaps


def frames_for_latents(num_video_frames: int, num_latent_frames: int) -> np.ndarray:
    """(F_lat, fpl) video-frame indices covered by each latent frame of a causal video VAE:
    latent 0 <- frame 0 (repeated), latent k <- frames (k-1)*fpl+1 .. k*fpl."""
    if num_latent_frames == num_video_frames:
        return np.arange(num_video_frames)[:, None]
    fpl = (num_video_frames - 1) // (num_latent_frames - 1)
    assert 1 + fpl * (num_latent_frames - 1) == num_video_frames, (num_video_frames, num_latent_frames)
    first = np.zeros((1, fpl), dtype=int)
    rest = 1 + np.arange((num_latent_frames - 1) * fpl).reshape(num_latent_frames - 1, fpl)
    return np.concatenate([first, rest])


class TextEmbeddings:
    """text -> (L, D) embedding, zero-padded to max_len (Wan pads umT5 outputs with zeros)."""

    def __init__(self, path: str | Path | None, dim: int, max_len: int = 512):
        self.dim, self.max_len = dim, max_len
        self.table = {}
        if path is not None:
            d = torch.load(path, weights_only=False)
            self.table = dict(zip(d["texts"], d["emb"]))
            self.dim = next(iter(self.table.values())).shape[-1]

    def __call__(self, text: str) -> torch.Tensor:
        out = torch.zeros(self.max_len, self.dim)
        e = self.table.get(text)
        if e is not None:
            out[: min(len(e), self.max_len)] = e[: self.max_len].float()
        return out


class LatentClips(Dataset):
    def __init__(self, root, splits=("train",), action_stats="data/action_stats.npz", text_emb=None,
                 text_dim=4096, text_len=512, max_latent_frames=None, crop_hw=None, heatmap_sigma=1.0,
                 cache_files=32, seed=0):
        self.root = Path(root)
        self.files, self.items = self._manifest(splits)
        st = np.load(action_stats)
        self.a_mean, self.a_std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
        self.text = TextEmbeddings(text_emb, text_dim, text_len)
        self.max_latent_frames, self.crop_hw, self.sigma = max_latent_frames, crop_hw, heatmap_sigma
        self._cache: OrderedDict = OrderedDict()
        self._cache_n = cache_files
        self._rng = np.random.default_rng(seed)

    def _manifest(self, splits):
        mpath = self.root / "manifest.json"
        files = sorted(p.name for p in self.root.glob("*.pt") if not p.name.startswith("."))  # skip macOS ._ files
        man = json.loads(mpath.read_text()) if mpath.exists() else {}
        changed = False
        missing = [f for f in files if f not in man]
        if missing:
            from concurrent.futures import ThreadPoolExecutor

            def info(f):
                side = (self.root / f).with_suffix(".json")  # written by preprocess.py
                if side.exists():
                    return f, json.loads(side.read_text())
                d = torch.load(self.root / f, weights_only=False, map_location="cpu")
                return f, {"n": int(d["latents"].shape[0]), "split": d["split"], "task": d["task"]}

            with ThreadPoolExecutor(32) as ex:
                man.update(dict(ex.map(info, missing)))
            changed = True
        if changed:
            mpath.write_text(json.dumps(man))
        keep = [f for f in files if man[f]["split"] in splits]
        items = [(fi, ci) for fi, f in enumerate(keep) for ci in range(man[f]["n"])]
        return keep, items

    def _load(self, fi):
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        d = torch.load(self.root / self.files[fi], weights_only=False, map_location="cpu")
        self._cache[fi] = d
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return d

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fi, ci = self.items[i]
        d = self._load(fi)
        lat = d["latents"][ci].float()  # (C,Fl,h,w)
        uv, z, valid, euv, eok, act = (d[k][ci].numpy() for k in ("uv", "z", "valid", "elbow_uv", "elbow_ok", "actions"))
        C, Fl, h, w = lat.shape
        groups = frames_for_latents(uv.shape[0], Fl)  # (Fl,fpl)
        if self.max_latent_frames is not None and Fl > self.max_latent_frames:
            Fl = self.max_latent_frames
            lat, groups = lat[:, :Fl], groups[:Fl]
        fpl = groups.shape[1]
        used = np.unique(groups)
        hm = {t: joint_heatmaps(uv[t], z[t], valid[t], (h, w), self.sigma, euv[t], eok[t]) for t in used}
        spatial = np.stack([np.concatenate([hm[t] for t in g]) for g in groups], 1)  # (S*fpl,Fl,h,w)
        a = (act - self.a_mean) / self.a_std
        joints = a[:, 9:135].reshape(len(a), 2, 21, 3).copy()  # see conditioning.action_vector layout
        joints[~valid] = 0.0
        a[:, 9:135] = joints.reshape(len(a), -1)
        actions = a[groups].reshape(Fl, fpl * a.shape[-1])
        spatial = torch.from_numpy(spatial)
        if self.crop_hw is not None:
            ch, cw = self.crop_hw
            y0 = int(self._rng.integers(0, h - ch + 1))
            x0 = int(self._rng.integers(0, w - cw + 1))
            lat, spatial = lat[..., y0:y0 + ch, x0:x0 + cw], spatial[..., y0:y0 + ch, x0:x0 + cw]
        return {"latents": lat, "spatial": spatial, "actions": torch.from_numpy(actions.astype(np.float32)),
                "text": self.text(d["text"]), "frames_per_latent": fpl}


SPATIAL_PER_FRAME = NUM_SPATIAL_CHANNELS
