"""Train the EgoDex hand pose estimator and compare it with MediaPipe on held-out frames.

GPU:  python scripts/train_handpose.py --frames /workspace/data/handpose_frames --epochs 6 --out runs/handpose
Mac:  python scripts/train_handpose.py --frames data/handpose_smoke --backbone resnet18 --in-w 424 --in-h 240 \
          --steps 100 --batch-size 8 --device mps --out runs/handpose_smoke

Reported on val_unseen (tasks never seen in training) and val_seen, on REAL frames:
  err_px       median per-joint error at 1080p, joints 2-20 (wrist/thumb base excluded so the
               MediaPipe comparison is fair; err_px_all includes them, ours only)
  recall       fraction of visibly-labelled hands (>=10 valid joints) found (present prob > 0.5)
  pck20        fraction of those joints within 20 px (1080p)
  erased_present  on the same frames with every hand painted over by background: fraction of
               hands still reported present. Must be low, or 'recall' on generated frames means
               nothing (EgoDex almost always shows hands, so a lazy 'present' head says yes always).
This real-frame accuracy is the floor the world model is judged against.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex.handpose import HandPoseNet, decode, target_heatmaps  # noqa: E402

FAIR = list(range(2, 21))


def _worker_init(_):
    """Give each DataLoader worker its own augmentation RNG (torch seeds workers distinctly)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def erase_hand(img, uv, valid, h, rng):
    """Paint hand h over with a same-size patch from elsewhere in the image (no hands there)."""
    H, W = img.shape[:2]
    p = uv[h][valid[h]]
    pad = 0.25 * (p.max(0) - p.min(0)) + 10
    x0, y0 = np.maximum(p.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(p.max(0) + pad, [W, H]).astype(int)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4 or bw >= W or bh >= H:
        return False
    if rng.random() < 0.5:
        # Smooth inpainting (no rectangular seam), alternating with patch copies (real texture),
        # so neither fill style is a shortcut for "no hand here".
        mask = np.zeros((H, W), np.uint8)
        mask[y0:y1, x0:x1] = 255
        img[:] = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
        return True
    boxes = [(uv[k][valid[k]].min(0), uv[k][valid[k]].max(0)) for k in range(2) if valid[k].sum()]
    for _ in range(20):
        sx, sy = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        if all(sx + bw < lo[0] - 10 or sx > hi[0] + 10 or sy + bh < lo[1] - 10 or sy > hi[1] + 10 for lo, hi in boxes):
            patch = img[sy:sy + bh, sx:sx + bw].copy()
            alpha = np.ones((bh, bw), np.float32)
            f = max(2, min(bw, bh) // 8)  # feathered edge so the seam is not a give-away
            alpha = cv2.blur(cv2.copyMakeBorder(alpha[f:-f, f:-f], f, f, f, f, cv2.BORDER_CONSTANT, value=0), (2 * f + 1,) * 2)
            reg = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = (alpha[..., None] * patch + (1 - alpha[..., None]) * reg).astype(np.uint8)
            return True
    return False


class Frames(Dataset):
    def __init__(self, root, labels, rows, in_wh, augment, seed=0, erase_p=0.2, erase_all=False):
        self.root, self.L, self.rows, self.in_wh, self.aug = Path(root), labels, rows, in_wh, augment
        self.erase_p, self.erase_all = erase_p, erase_all
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def _augment(self, img, uv, valid):
        H, W = img.shape[:2]
        rng = self.rng
        s = rng.uniform(0.85, 1.15)
        ang = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
        M[:, 2] += rng.uniform(-0.08, 0.08, 2) * [W, H]
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        uv = uv @ M[:, :2].T + M[:, 2]
        valid = valid & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
        img = img.astype(np.float32)
        img = img * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)  # contrast / brightness
        gray = img.mean(-1, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.7, 1.3)  # saturation
        img = np.clip(img, 0, 255).astype(np.uint8)
        # Generated frames come out of a VAE: softer and blockier than real video.
        if rng.random() < 0.3:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if rng.random() < 0.3:
            f = rng.uniform(2, 4)
            small = cv2.resize(img, (int(W / f), int(H / f)), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.3:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return img, uv, valid

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / self.L["files"][r])), cv2.COLOR_BGR2RGB)
        if img.shape[1::-1] != self.in_wh:
            img = cv2.resize(img, self.in_wh, interpolation=cv2.INTER_AREA)
        W, H = self.in_wh
        uv = (self.L["uv"][r] + 0.5) * [W / 1920, H / 1080] - 0.5  # 1080p -> input pixels
        valid = self.L["valid"][r].copy()
        if self.aug:
            img, uv, valid = self._augment(img, uv, valid)
        if self.aug or self.erase_all:
            img = np.ascontiguousarray(img)
            rng = self.rng if not self.erase_all else np.random.default_rng(i)
            for h in range(2):
                if valid[h].sum() >= 5 and (self.erase_all or rng.random() < self.erase_p):
                    if erase_hand(img, uv, valid, h, rng):
                        valid[h] = False
        return {"img": torch.from_numpy(img).permute(2, 0, 1),
                "heat": torch.from_numpy(target_heatmaps(uv, valid, self.in_wh)),
                "mask": torch.from_numpy(valid.reshape(-1).astype(np.float32)),
                "present": torch.from_numpy((valid.sum(-1) >= 5).astype(np.float32)),
                "uv1080": torch.from_numpy(((uv + 0.5) * [1920 / W, 1080 / H] - 0.5).astype(np.float32)),
                "valid": torch.from_numpy(valid)}


def loss_fn(heat, pres, b):
    if heat.shape[-2:] != b["heat"].shape[-2:]:
        heat = F.interpolate(heat, size=b["heat"].shape[-2:], mode="bilinear", align_corners=False)
    # Per-joint spatial softmax + cross-entropy against the normalized Gaussian target.
    logp = torch.log_softmax(heat.float().flatten(2), -1)
    tgt = b["heat"].flatten(2)
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-6)
    ce = -(tgt * logp).sum(-1)  # (B,42)
    l_heat = (ce * b["mask"]).sum() / b["mask"].sum().clamp(min=1)
    l_pres = F.binary_cross_entropy_with_logits(pres.float(), b["present"])
    return l_heat + l_pres, l_heat.item(), l_pres.item()


@torch.no_grad()
def erased_present(net, ds, device, amp, n=500):
    """Fraction of (erased) hands still reported present."""
    net.eval()
    L = ds.L
    era = Frames(ds.root, L, ds.rows[:n], ds.in_wh, False, erase_all=True)
    orig_valid = (L["valid"][ds.rows[:n]].sum(-1) >= 5)  # hands that were there before erasing
    probs = []
    for b in DataLoader(era, batch_size=32):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, pres = net(b["img"].to(device).float() / 255)
        probs.append(torch.sigmoid(pres.float()).cpu().numpy())
    net.train()
    still = np.concatenate(probs) > 0.5
    erased = orig_valid & ~np.stack([era[i]["present"].numpy() > 0 for i in range(len(era))])
    return round(float(still[erased].mean()), 3) if erased.any() else None


def scores(pred_uv, prob, uv, valid):
    """Aggregate stats over hands that are visibly labelled (>=10 valid joints)."""
    labelled = valid.sum(-1) >= 10  # (N,2)
    found = labelled & (prob > 0.5)
    err = np.linalg.norm(pred_uv - uv, axis=-1)  # (N,2,21)
    m_fair = found[..., None] & valid & np.isin(np.arange(21), FAIR)[None, None]
    m_all = found[..., None] & valid
    return {"recall": round(float(found.sum() / max(labelled.sum(), 1)), 3),
            "err_px": round(float(np.median(err[m_fair])), 1) if m_fair.any() else None,
            "err_px_all": round(float(np.median(err[m_all])), 1) if m_all.any() else None,
            "pck20": round(float((err[m_fair] < 20).mean()), 3) if m_fair.any() else None,
            "hands": int(labelled.sum())}


@torch.no_grad()
def evaluate(net, loader, device, amp):
    net.eval()
    P, PR, U, V, imgs = [], [], [], [], []
    for b in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            heat, pres = net(b["img"].to(device).float() / 255)
        uv, _ = decode(heat.float())
        P.append(uv.cpu().numpy()); PR.append(torch.sigmoid(pres.float()).cpu().numpy())
        U.append(b["uv1080"].numpy()); V.append(b["valid"].numpy())
    net.train()
    return scores(np.concatenate(P), np.concatenate(PR), np.concatenate(U), np.concatenate(V))


def mediapipe_scores(ds, n):
    """MediaPipe on the same real frames, matched to hands by position (its handedness labels
    assume mirrored images), scored with the same rules."""
    from egodex.handdet import HandDetector, match_to_hands
    det = HandDetector()
    P, PR, U, V = [], [], [], []
    for i in range(min(n, len(ds))):
        b = ds[i]
        img = b["img"].permute(1, 2, 0).numpy()
        W, H = img.shape[1], img.shape[0]
        uv, valid = b["uv1080"].numpy(), b["valid"].numpy()
        m = match_to_hands(det(img), uv * [W / 1920, H / 1080], valid, max_dist=120 * W / 848)
        p = np.zeros((2, 21, 2), np.float32)
        pr = np.zeros(2, np.float32)
        for h, duv in m.items():
            p[h] = (duv + 0.5) * [1920 / W, 1080 / H] - 0.5
            pr[h] = 1.0
        P.append(p); PR.append(pr); U.append(uv); V.append(valid)
    det.close()
    return scores(np.stack(P), np.stack(PR), np.stack(U), np.stack(V))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--in-w", type=int, default=848)
    ap.add_argument("--in-h", type=int, default=480)
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--steps", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-frames", type=int, default=3000)
    ap.add_argument("--mediapipe-frames", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    amp = device.type == "cuda"
    in_wh = (a.in_w, a.in_h)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    L = dict(np.load(Path(a.frames) / "labels.npz"))
    rng = np.random.default_rng(0)
    tr_rows = np.nonzero(L["split"] == "train")[0]
    val = {s: rng.permutation(np.nonzero(L["split"] == s)[0])[: a.val_frames] for s in ("val_unseen", "val_seen")}
    print(f"train frames {len(tr_rows)}, " + ", ".join(f"{k} {len(v)}" for k, v in val.items()), flush=True)

    train = DataLoader(Frames(a.frames, L, tr_rows, in_wh, True), batch_size=a.batch_size, shuffle=True,
                       drop_last=True, num_workers=a.num_workers, pin_memory=amp,
                       persistent_workers=a.num_workers > 0,
                       worker_init_fn=_worker_init)
    vloaders = {s: DataLoader(Frames(a.frames, L, r, in_wh, False), batch_size=a.batch_size,
                              num_workers=a.num_workers) for s, r in val.items() if len(r)}

    net = HandPoseNet(a.backbone).to(device)
    steps = a.steps or int(a.epochs * len(train))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    log = open(out / "log.jsonl", "a")
    step, t0 = 0, time.time()
    net.train()
    while step < steps:
        for b in train:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                heat, pres = net(b["img"].float() / 255)
            loss, lh, lp = loss_fn(heat, pres, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "loss_heat": lh, "loss_present": lp, "img_per_s": 50 * a.batch_size / (time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                t0 = time.time()
            if step % a.val_every == 0 or step == steps:
                rec = {"step": step, **{s: {**evaluate(net, l, device, amp),
                                            "erased_present": erased_present(net, l.dataset, device, amp)}
                                        for s, l in vloaders.items()}}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save({"model": net.state_dict(), "backbone": a.backbone, "in_wh": in_wh, "step": step},
                           out / "handpose.pt")
            if step >= steps:
                break

    final = {"ours": {s: {**evaluate(net, l, device, amp), "erased_present": erased_present(net, l.dataset, device, amp)}
                      for s, l in vloaders.items()}}
    if a.mediapipe_frames:
        final["mediapipe"] = {s: mediapipe_scores(l.dataset, a.mediapipe_frames) for s, l in vloaders.items()}
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
