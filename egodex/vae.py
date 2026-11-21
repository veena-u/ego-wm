"""Video VAE loading shared by preprocessing, the VAE comparison and evaluation.

All VAEs here compress 8x spatially; the video ones also 4x temporally with a causal first
frame (T frames -> 1 + (T-1)/4 latents). 'bicubic8x' is a learned-nothing stand-in that
runs fast on a Mac so the surrounding code can be tested without a GPU.
"""
from __future__ import annotations

import torch

VAES = {
    "wan2.1": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "AutoencoderKLWan"),
    "cogvideox": ("THUDM/CogVideoX-2b", "AutoencoderKLCogVideoX"),
    "hunyuan": ("hunyuanvideo-community/HunyuanVideo", "AutoencoderKLHunyuanVideo"),
}


class _Dist:
    def __init__(self, z):
        self.z = z

    def mode(self):
        return self.z


class Bicubic8x(torch.nn.Module):
    """Stand-in 'VAE': 8x bicubic down/up per frame, 3 channels, no temporal compression."""

    latent_channels = 3
    temporal_compression = 1

    def encode(self, x):
        B, C, T, H, W = x.shape
        z = torch.nn.functional.interpolate(x.transpose(1, 2).reshape(B * T, C, H, W), scale_factor=1 / 8,
                                            mode="bicubic", antialias=True)
        out = type("E", (), {})()
        out.latent_dist = _Dist(z.reshape(B, T, C, H // 8, W // 8).transpose(1, 2))
        return out

    def decode(self, z):
        B, C, T, h, w = z.shape
        y = torch.nn.functional.interpolate(z.transpose(1, 2).reshape(B * T, C, h, w), scale_factor=8, mode="bicubic")
        out = type("D", (), {})()
        out.sample = y.reshape(B, T, C, h * 8, w * 8).transpose(1, 2)
        return out


def load_vae(name: str, device, dtype=torch.float32, tiling: bool = True):
    if name == "bicubic8x":
        return Bicubic8x().to(device)
    import diffusers

    repo, cls = VAES[name]
    vae = getattr(diffusers, cls).from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
    if tiling:
        for m in ("enable_tiling", "enable_slicing"):
            if hasattr(vae, m):
                getattr(vae, m)()
    return vae.to(device).eval()


def latent_norm(vae):
    """Per-channel (mean, std) used to normalize latents to ~unit variance, as the matching
    diffusion pipeline does. Returns (None, None) when the VAE defines none."""
    cfg = getattr(vae, "config", None)
    mean = getattr(cfg, "latents_mean", None) if cfg is not None else None
    std = getattr(cfg, "latents_std", None) if cfg is not None else None
    if mean is not None and std is not None:
        return torch.tensor(mean), torch.tensor(std)
    sf = getattr(cfg, "scaling_factor", None) if cfg is not None else None
    if sf is not None:
        return torch.tensor(0.0), torch.tensor(1.0 / sf)
    return None, None


def to_video_tensor(frames_uint8, device, dtype=torch.float32):
    """(T,H,W,3) uint8 numpy -> (1,3,T,H,W) in [-1,1]."""
    x = torch.from_numpy(frames_uint8).to(device).permute(3, 0, 1, 2)[None]
    return x.to(dtype).div(127.5).sub(1)


def from_video_tensor(y):
    """(1,3,T,H,W) in [-1,1] -> (T,H,W,3) uint8 numpy."""
    return ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()
