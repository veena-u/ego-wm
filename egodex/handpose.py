"""EgoDex-trained hand pose estimator: frame -> the same 21 joints per hand we condition on.

Used as the main evaluation instrument. Trained on real EgoDex frames with the (lag- and
calibration-corrected) projected ARKit joints as labels, so its joint definitions match the
world model's conditioning exactly and it is in-domain (egocentric, Vision Pro, hands on objects).

Output: 42 heatmap LOGITS (left 0-20, right 21-41, MediaPipe order) at stride 4 - each joint is
a softmax distribution over pixels, trained with cross-entropy against a Gaussian (plain MSE
regression sits at 'predict zeros' for a long time) - plus a per-hand 'present' logit. Coordinates are decoded to the 1080p image frame used everywhere else.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STRIDE = 4
NUM_OUT = 42
IN_WH = (848, 480)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class HandPoseNet(nn.Module):
    """ResNet backbone + FPN-style decoder back up to stride 4."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, width: int = 128):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        chans = [l[-1].conv3.out_channels if hasattr(l[-1], "conv3") else l[-1].conv2.out_channels
                 for l in self.layers]
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chans])
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.heat = nn.Conv2d(width, NUM_OUT, 1)
        self.present = nn.Linear(chans[-1], 2)
        self.backbone_name = backbone

    def forward(self, x):
        """x: (B,3,H,W) float in [0,1]. Returns heatmaps (B,42,H/4,W/4), present logits (B,2)."""
        x = (x - _MEAN.to(x)) / _STD.to(x)
        x = self.stem(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        y = self.lat[3](feats[3])
        for i in (2, 1, 0):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](feats[i])
        return self.heat(self.smooth(y)), self.present(feats[3].mean((2, 3)))


def decode(heat: torch.Tensor, image_wh=(1920, 1080), logits: bool = True):
    """Sub-pixel joint locations from heatmaps: argmax, then a parabola fit to the LOG heatmap
    around it, per axis. A Gaussian's log is exactly a parabola, so this is unbiased (a windowed
    centroid is pulled toward the argmax cell by up to ~1/4 cell).
    heat: logits (network output) or, with logits=False, positive heatmaps (e.g. targets).
    Returns uv (B,2,21,2) in image_wh pixels and peak probability (B,2,21)."""
    B, J, H, W = heat.shape
    lg = heat.float() if logits else torch.log(heat.float().clamp(min=1e-6))
    flat = lg.flatten(2)
    idx = flat.argmax(-1)
    peak = torch.softmax(flat, -1).max(-1).values if logits else heat.flatten(2).max(-1).values
    py, px = idx // W, idx % W
    b = torch.arange(B, device=heat.device)[:, None]
    j = torch.arange(J, device=heat.device)[None]

    def offset(c, n, size, along_x):
        lo, hi = (c - 1).clamp(0, size - 1), (c + 1).clamp(0, size - 1)
        if along_x:
            f0, fm, fp = lg[b, j, py, c], lg[b, j, py, lo], lg[b, j, py, hi]
        else:
            f0, fm, fp = lg[b, j, c, px], lg[b, j, lo, px], lg[b, j, hi, px]
        den = fm - 2 * f0 + fp
        off = torch.where(den < -1e-6, 0.5 * (fm - fp) / den, torch.zeros_like(den))
        ok = (c > 0) & (c < size - 1)  # no fit at the border
        return torch.where(ok, off.clamp(-0.5, 0.5), torch.zeros_like(off))

    gx = px.float() + offset(px, None, W, True)
    gy = py.float() + offset(py, None, H, False)
    u = (gx + 0.5) * image_wh[0] / W - 0.5
    v = (gy + 0.5) * image_wh[1] / H - 0.5
    uv = torch.stack([u, v], -1).view(B, 2, 21, 2)
    return uv, peak.view(B, 2, 21)


def target_heatmaps(uv_in: np.ndarray, valid: np.ndarray, in_wh=IN_WH, sigma: float = 2.0):
    """Gaussian targets at stride 4. uv_in: (2,21,2) pixels in the network input image."""
    W, H = in_wh[0] // STRIDE + (in_wh[0] % STRIDE > 0), in_wh[1] // STRIDE + (in_wh[1] % STRIDE > 0)
    g = (uv_in.reshape(-1, 2) + 0.5) / STRIDE - 0.5
    xs = np.arange(W, dtype=np.float32)[None, None, :]
    ys = np.arange(H, dtype=np.float32)[None, :, None]
    d2 = (xs - g[:, 0, None, None]) ** 2 + (ys - g[:, 1, None, None]) ** 2
    return (np.exp(-d2 / (2 * sigma ** 2)) * valid.reshape(-1, 1, 1)).astype(np.float32)


class Estimator:
    """Load a trained checkpoint and predict on uint8 RGB frames of any size."""

    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.in_wh = tuple(ck["in_wh"])
        self.net = HandPoseNet(ck["backbone"], pretrained=False)
        self.net.load_state_dict(ck["model"])
        self.net = self.net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch: int = 32):
        """frames (T,H,W,3) uint8 RGB. Returns uv (T,2,21,2) at 1080p, joint peak (T,2,21),
        hand probability (T,2)."""
        outs = []
        for i in range(0, len(frames), batch):
            x = np.stack([cv2.resize(f, self.in_wh, interpolation=cv2.INTER_AREA)
                          if f.shape[1::-1] != self.in_wh else f for f in frames[i:i + batch]])
            x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255
            heat, pres = self.net(x)
            uv, peak = decode(heat.float())
            outs.append((uv.cpu().numpy(), peak.cpu().numpy(), torch.sigmoid(pres).cpu().numpy()))
        return tuple(np.concatenate(o) for o in zip(*outs))
