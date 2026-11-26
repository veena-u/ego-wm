"""One interactive session: hand poses in, video frames out, for an unbounded time.

  s = LiveSession(model, vae, text, cfg=3.0, steps=10)
  s.start(frames0, poses0)        # scene: 1 frame (or 1+fpl) and its hand poses
  frames = s.step(poses)          # fpl new hand poses (15 fps) -> fpl new frames

Generation streams through the KV cache (egodex/stream.py) and decoding streams through the
causal VAE's own frame cache (VAEStream), so each step costs one latent frame of work.

Training clips are 9 latent frames long, so the model never saw frame positions past 8. Before
that, the session RE-ANCHORS: the last 1+fpl generated frames are re-encoded into 2 clean latents
at positions 0-1, exactly how every training clip starts (a 1-frame latent, then a fpl-frame one),
with those frames' poses as their conditioning. Positions and context stay in-distribution forever.
"""
from __future__ import annotations

import numpy as np
import torch

from .live import Conditioner
from .stream import StreamingWorldModel
from .vae import from_video_tensor, latent_norm, to_video_tensor


class VAEStream:
    """Decode a latent video one latent frame at a time, identical to decoding it all at once
    (Wan's causal decoder keeps per-layer frame caches between calls). Also encodes."""

    def __init__(self, vae):
        self.vae = vae
        self.mean, self.std = latent_norm(vae)
        self.dtype = next(vae.parameters()).dtype if list(vae.parameters()) else torch.float32
        self.streaming = hasattr(vae, "clear_cache")  # Wan-style causal VAE with frame caches
        self.reset()

    def reset(self):
        if self.streaming:
            self.vae.clear_cache()
        self.first = True

    def _denorm(self, z):
        if self.mean is None:
            return z.float()
        shape = (1, -1) + (1,) * (z.ndim - 2)
        return z.float() * self.std.to(z.device).view(shape) + self.mean.to(z.device).view(shape)

    @torch.no_grad()
    def decode_next(self, z):
        """z: (1,C,1,h,w) normalized latent -> its frames (T,H,W,3) uint8 (1 frame first, then fpl)."""
        z = self._denorm(z).to(self.dtype)
        if not self.streaming:
            return from_video_tensor(self.vae.decode(z).sample)
        v = self.vae
        x = v.post_quant_conv(z)
        v._conv_idx = [0]
        out = v.decoder(x, feat_cache=v._feat_map, feat_idx=v._conv_idx, first_chunk=self.first)
        self.first = False
        if v.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
            out = unpatchify(out, patch_size=v.config.patch_size)
        return from_video_tensor(out.clamp(-1, 1))

    @torch.no_grad()
    def encode(self, frames, device):
        """(T,H,W,3) uint8 -> normalized latents (1,C,Fl,h,w), as preprocess.py makes them.
        Wan's encode() clears ALL caches, the decoder's too; keep the decoder's stream intact."""
        saved = (self.vae._feat_map, self.vae._conv_idx) if self.streaming else None
        z = self.vae.encode(to_video_tensor(frames, device, self.dtype)).latent_dist.mode().float()
        if saved is not None:
            self.vae._feat_map, self.vae._conv_idx = saved
        if self.mean is not None:
            shape = (1, -1) + (1,) * (z.ndim - 2)
            z = (z - self.mean.to(z.device).view(shape)) / self.std.to(z.device).view(shape)
        return z


class LiveSession:
    def __init__(self, model, vae, text, cfg: float = 3.0, steps: int = 10, shift: float = 3.0,
                 max_frames: int = 9, action_stats="data/action_stats.npz", seed: int = 0,
                 student_sigmas=None):
        """text: (1,L,D) embedding. max_frames: latent positions the model was trained on.
        student_sigmas: a distilled student's few-step sampler (then cfg must be 1: it is baked in)."""
        assert student_sigmas is None or cfg == 1.0
        self.student_sigmas = student_sigmas
        self.model, self.text, self.cfg, self.steps, self.shift = model, text, cfg, steps, shift
        self.max_frames = max_frames
        self.p = next(model.parameters())
        self.vs = VAEStream(vae)
        self.fpl = model.fpl
        self.action_stats = action_stats
        self.cond = None
        self.g = torch.Generator(device=self.p.device).manual_seed(seed)
        self.recent_frames, self.recent_poses = [], []  # for re-anchoring
        self.reanchors = 0

    def _c(self, poses):
        sp, ac = self.cond.latent(poses)
        return (torch.from_numpy(sp)[None, :, None].to(self.p.device, self.p.dtype),
                torch.from_numpy(ac)[None, None].to(self.p.device, self.p.dtype))

    def _remember(self, frames, poses):
        self.recent_frames = (self.recent_frames + list(frames))[-(1 + self.fpl):]
        self.recent_poses = (self.recent_poses + list(poses))[-(1 + self.fpl):]

    def _anchor(self, frames, poses):
        """New stream with these frames as clean context: 1 frame -> 1 latent, 1+fpl -> 2."""
        z = self.vs.encode(np.stack(frames), self.p.device).to(self.p.dtype)
        self.s = StreamingWorldModel(self.model, self.text, cfg=self.cfg)
        groups = [poses[:1]] + [poses[1 + k * self.fpl:1 + (k + 1) * self.fpl] for k in range(z.shape[2] - 1)]
        for k, g in enumerate(groups):
            self.s.push(z[:, :, k:k + 1], *self._c(g))
        return z

    def start(self, frames0, poses0):
        """frames0: 1 (or 1+fpl) frames (H,W,3) uint8 at the training size; poses0: one pose dict
        per frame (egodex.live.pose_dict) - the scene's own hands, or the user's current pose."""
        frames0, poses0 = list(frames0), list(poses0)
        H, W = frames0[0].shape[:2]
        h, w = H // 8, W // 8
        self.cond = Conditioner(self.action_stats, grid_hw=(h, w), fpl=self.fpl)
        z = self._anchor(frames0, poses0)
        self.vs.reset()
        for k in range(z.shape[2]):  # warm the decoder's frame cache on the context
            self.vs.decode_next(z[:, :, k:k + 1])
        self._remember(frames0, poses0)

    @torch.no_grad()
    def step(self, poses):
        """fpl pose dicts (15 fps) -> fpl new frames (fpl,H,W,3) uint8."""
        assert len(poses) == self.fpl
        if self.s.t >= self.max_frames:
            self._anchor(self.recent_frames, self.recent_poses)
            self.reanchors += 1
        if self.student_sigmas is not None:
            z = self.s.next_few(*self._c(poses), self.student_sigmas, generator=self.g)
        else:
            z = self.s.next(*self._c(poses), steps=self.steps, shift=self.shift, generator=self.g)
        frames = self.vs.decode_next(z)
        self._remember(frames, poses)
        return frames
