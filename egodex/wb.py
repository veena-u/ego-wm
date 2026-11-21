"""Optional Weights & Biases logging. Off unless a project name is given; the API key comes from
`wandb login` / WANDB_API_KEY on the machine, never from this repo.

The run id is stored in <out_dir>/wandb_id.txt so a resumed training run continues the same
W&B run instead of starting a new one.
"""
from __future__ import annotations

import uuid
from pathlib import Path


def init(project: str | None, out_dir, config: dict, job_type: str):
    if not project:
        return None
    import wandb

    out_dir = Path(out_dir)
    idf = out_dir / "wandb_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else uuid.uuid4().hex[:12]
    idf.write_text(run_id)
    return wandb.init(project=project, id=run_id, resume="allow", name=out_dir.name, config=config,
                      job_type=job_type, dir=str(out_dir))


def log(run, rec: dict, step: int | None = None):
    if run is None:
        return
    flat = {}
    for k, v in rec.items():
        if isinstance(v, dict):  # e.g. {"val_unseen": {"recall": ..}} -> "val_unseen/recall"
            flat.update({f"{k}/{kk}": vv for kk, vv in v.items() if isinstance(vv, (int, float)) or vv is None})
        elif k != "step" and (isinstance(v, (int, float)) or v is None):
            flat[k] = v
    run.log(flat, step=step)
