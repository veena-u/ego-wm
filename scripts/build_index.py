"""Build the episode index, quality filters, splits and action normalization stats.

python scripts/build_index.py ~/Downloads/test [/data/egodex/part1 ...] --out data/

Writes:
  data/index.csv          one row per episode: metadata, quality metrics, keep flag, split
  data/action_stats.npz   per-dim mean/std of the action vector over train clips
  data/summary.json       counts per split / filter reason
Splits: held-out TASKS (val_unseen) measure generalization to new activities; a few episodes
of training tasks (val_seen) measure fit. Assignment is deterministic (hash of task / path).
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex import Episode, list_episodes  # noqa: E402
from egodex.conditioning import ACTION_DIM, make_clip  # noqa: E402
from egodex.index import valid_clip_starts  # noqa: E402

# Filter thresholds (see summary.json for how many episodes each removes).
MIN_FRAMES = 33          # >= one 17-frame clip at 15 fps
MAX_SPIKES = 3           # isolated glitches are fine: clips overlapping them are skipped
MIN_HAND_VISIBLE = 0.8   # fraction of frames with at least one hand mostly in view
MIN_CONF = 0.30          # mean joint confidence, only when confidences exist
UNSEEN_TASK_FRAC = 0.10
SEEN_VAL_FRAC = 0.05


def _frac(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def episode_row(path: Path) -> dict:
    ep = Episode(path)
    n = min(ep.T, ep.num_video_frames())
    inview = ep.in_view(range(n))  # (n,2,21)
    hand_vis = inview.mean(-1) > 0.5  # (n,2)
    any_vis = hand_vis.any(-1).mean() if n else 0.0
    R = np.einsum("tij,tkj->tik", ep.cam_to_world[1:n, :3, :3], ep.cam_to_world[: n - 1, :3, :3])
    head_deg = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    conf = float(np.nanmean(ep.confidence[:n])) if ep.confidence is not None else float("nan")
    spikes = ep.spike_frames()
    row = {
        # Folder name is the canonical task: the HDF5 'task' attr is inconsistent
        # (e.g. 'basic fold' vs basic_fold, 'Dry Hands' vs dry_hands_towel).
        "path": str(path), "task": path.parent.name, "task_attr": ep.attrs.get("task", ""),
        "description": ep.description,
        "environment": ep.attrs.get("environment", ""), "session": ep.attrs.get("session_name", ""),
        "n_frames": n, "seconds": round(n / 30, 2),
        "has_conf": ep.confidence is not None, "mean_conf": round(conf, 4),
        "spikes": len(spikes), "spike_frames": ";".join(map(str, spikes.tolist())), "any_hand_visible": round(float(any_vis), 4),
        "left_visible": round(float(hand_vis[:, 0].mean()), 4) if n else 0.0,
        "right_visible": round(float(hand_vis[:, 1].mean()), 4) if n else 0.0,
        "head_deg_per_frame_p90": round(float(np.percentile(head_deg, 90)), 4) if len(head_deg) else 0.0,
    }
    reasons = []
    clip_frames = (MIN_FRAMES - 1) // 2 + 1
    if len(valid_clip_starts(n, spikes, clip_frames)) == 0:
        reasons.append("no_clean_clip")
    if n < MIN_FRAMES:
        reasons.append("too_short")
    if row["spikes"] > MAX_SPIKES:
        reasons.append("tracking_spikes")
    if any_vis < MIN_HAND_VISIBLE:
        reasons.append("hands_out_of_view")
    if row["has_conf"] and conf < MIN_CONF:
        reasons.append("low_confidence")
    row["keep"] = not reasons
    row["drop_reason"] = "|".join(reasons)
    return row


def assign_splits(rows):
    tasks = sorted({r["task"] for r in rows})
    unseen = {t for t in tasks if _frac("task:" + t) < UNSEEN_TASK_FRAC}
    for r in rows:
        if r["task"] in unseen:
            r["split"] = "val_unseen"
        elif _frac("ep:" + "/".join(Path(r["path"]).parts[-3:])) < SEEN_VAL_FRAC:  # machine-independent
            r["split"] = "val_seen"
        else:
            r["split"] = "train"
    return unseen


def _clip_actions(args):
    path, start, nf = args
    ep = Episode(path)
    return make_clip(ep, start, nf, read_video=False)["actions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="one or more EgoDex split folders (test, part1, ...)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stat-clips", type=int, default=3000)
    ap.add_argument("--clip-frames", type=int, default=33)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    eps = [e for root in a.roots for e in list_episodes(root)]
    with Pool(a.workers) as pool:
        rows = pool.map(episode_row, eps, chunksize=8)
    unseen = assign_splits(rows)

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Action normalization stats from random clips of kept train episodes.
    rng = np.random.default_rng(0)
    train = [r for r in rows if r["keep"] and r["split"] == "train"]
    jobs = []
    for i in rng.integers(0, len(train), a.stat_clips):
        r = train[i]
        spikes = [int(x) for x in r["spike_frames"].split(";") if x]
        starts = valid_clip_starts(r["n_frames"], spikes, a.clip_frames)
        if len(starts):
            jobs.append((r["path"], int(rng.choice(starts)), a.clip_frames))
    with Pool(a.workers) as pool:
        acts = np.concatenate(pool.map(_clip_actions, jobs, chunksize=16))
    mean, std = acts.mean(0), acts.std(0)
    std = np.where(std < 1e-4, 1.0, std)  # e.g. ref-frame head pose is constant
    assert acts.shape[1] == ACTION_DIM
    np.savez(out / "action_stats.npz", mean=mean, std=std, n=len(acts))

    kept = [r for r in rows if r["keep"]]
    summary = {
        "episodes": len(rows), "kept": len(kept),
        "hours_total": round(sum(r["seconds"] for r in rows) / 3600, 2),
        "hours_kept": round(sum(r["seconds"] for r in kept) / 3600, 2),
        "drop_reasons": Counter(x for r in rows for x in r["drop_reason"].split("|") if x),
        "kept_per_split": Counter(r["split"] for r in kept),
        "hours_kept_per_split": {s: round(sum(r["seconds"] for r in kept if r["split"] == s) / 3600, 2)
                                 for s in ("train", "val_seen", "val_unseen")},
        "unseen_tasks": sorted(unseen),
        "kept_with_clean_33_frame_clip_at_15fps": sum(
            len(valid_clip_starts(r["n_frames"], [int(x) for x in r["spike_frames"].split(";") if x], 33)) > 0
            for r in kept),
        "tasks": {s: len({r["task"] for r in kept if r["split"] == s}) for s in ("train", "val_seen", "val_unseen")},
        "action_stat_frames": int(len(acts)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
