"""Mac: put what the auto-launched training machine needs into r2:<bucket>/bootstrap/
(code bundle with data/action_stats.npz, the step-8000 start weights + checksum, handpose.pt).
Rerun after changing code that the training machine should use.
  .venv/bin/python scripts/make_bootstrap.py
"""
import hashlib
import io
import os
import tarfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "RUNBOOK.md", "requirements.txt", "egodex", "scripts", "models",
           "data/action_stats.npz", "data/calibration.json"]


def env():
    e = {}
    for line in (Path.home() / ".config/ego-wm/r2.env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip()
    return e


def main():
    e = env()
    s3 = boto3.client("s3", endpoint_url=e["R2_ENDPOINT"], aws_access_key_id=e["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=e["R2_SECRET_ACCESS_KEY"], region_name="auto")
    b = e["R2_BUCKET"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in INCLUDE:
            p = ROOT / name
            if p.exists():
                t.add(p, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".DS_Store") else ti)
    s3.put_object(Bucket=b, Key="bootstrap/ego-wm.tar.gz", Body=buf.getvalue())
    print(f"code bundle: {len(buf.getvalue()) / 1e6:.1f} MB")
    w = ROOT / "runs/v2/model_bf16_step8000.pt"
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    s3.put_object(Bucket=b, Key="bootstrap/model_bf16_step8000.pt.sha256", Body=sha.encode())
    for f in (w, ROOT / "runs/handpose/handpose.pt"):
        key = f"bootstrap/{f.name}"
        try:
            head = s3.head_object(Bucket=b, Key=key)
            if head["ContentLength"] == f.stat().st_size:
                print(f"{key}: already there")
                continue
        except Exception:
            pass
        s3.upload_file(str(f), b, key)
        print(f"{key}: uploaded {f.stat().st_size / 1e9:.2f} GB")
    print(f"start weights sha256 {sha[:16]}")


if __name__ == "__main__":
    main()
