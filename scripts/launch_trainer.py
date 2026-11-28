"""Rent the training machine on vast.ai and point it at the encoded data in R2.

Run by the last encoder machine when all encoding is done (scripts/encode_worker.sh), or by hand:
  python3 scripts/launch_trainer.py --dry-run      # show the offers it would pick from; rents nothing
  python3 scripts/launch_trainer.py                # rent one

Guard rails: only 4-GPU machines with >= 80 GB per GPU, CUDA >= 13.2 (our torch build), >= 99%
reliability, >= 1 Gbps, on-demand; a hard $/hour cap per GPU type (CAPS); refuses to rent unless the
account credit covers MIN_HOURS of it; one machine per run (the caller holds a lock in R2). After
renting it waits for the machine to come up and for its bootstrap to report in R2; a machine that
never starts is destroyed and the next offer tried; one that starts but whose bootstrap never
reports is stopped (kept for debugging) and the launch gives up.

The new machine gets, as instance environment variables, the R2 credentials, the W&B key and
MAX_TRAIN_HOURS; its onstart (scripts/trainer_onstart.sh) downloads the code bundle from R2
(scripts/make_bootstrap.py puts it there) and runs scripts/trainer_bootstrap.sh.
Credentials are read from env, else from ~/.config/ego-wm/ (or /root/.config/ego-wm/):
  vast.env (VAST_API_KEY: a scoped key: misc + instance_read + instance_write), r2.env, wandb.env
  (WANDB_API_KEY; else ~/.netrc).
Prints "LAUNCHED <id> <gpu> <$/h>" as its last line on success.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://console.vast.ai/api/v0"
IMAGE = "vastai/pytorch:cuda-13.2.1-auto"  # same image the encoders run (torch 2.15 cu132 venv)
CAPS = {  # max $/hour for the whole 4-GPU machine, by GPU type (all >= 80 GB per GPU)
    "B200": 18.0, "H200": 14.0, "H200 NVL": 14.0, "H100 SXM": 12.0, "H100 NVL": 12.0, "H100 PCIE": 10.0,
    "A100 SXM4": 8.0, "A100X": 8.0, "A100 PCIE": 7.0, "A800 PCIE": 6.0,
    "RTX PRO 6000 WS": 7.0, "RTX PRO 6000 S": 7.0, "RTX PRO 6000 Max-Q": 6.0,
}
# encoded training data ~0.9 TB (~3.4 MB per recording, 2 clips each, ~255k recordings; measured in R2)
# + validation 20 GB + checkpoints (~100 GB over 3 days) + headroom
DISK_GB = 1600
MIN_HOURS = 12
ROOT = Path(__file__).resolve().parents[1]


def load_env(name: str) -> dict:
    out = {}
    for d in (Path.home() / ".config/ego-wm", Path("/root/.config/ego-wm")):
        p = d / name
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
            break
    return out


def secrets() -> dict:
    s = {}
    vast = load_env("vast_scoped.env") or load_env("vast.env")
    s["VAST_API_KEY"] = os.environ.get("VAST_API_KEY") or vast.get("VAST_API_KEY")
    r2 = load_env("r2.env")
    for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        s[k] = os.environ.get(k) or r2.get(k)
    w = os.environ.get("WANDB_API_KEY") or load_env("wandb.env").get("WANDB_API_KEY")
    if not w and (Path.home() / ".netrc").exists():
        toks = (Path.home() / ".netrc").read_text().split()
        for i, t in enumerate(toks):
            if t == "machine" and i + 1 < len(toks) and ("wandb" in toks[i + 1] or "coreweave" in toks[i + 1]):
                for j in range(i + 2, min(i + 8, len(toks) - 1)):
                    if toks[j] == "password":
                        w = toks[j + 1]
                        break
                break
    s["WANDB_API_KEY"] = w
    missing = [k for k, v in s.items() if not v]
    if missing:
        sys.exit(f"missing credentials: {missing}")
    return s


def call(key, method, path, body=None, base=API):
    req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}")
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


def offers(key, exclude=()):
    q = {"num_gpus": {"eq": 4}, "gpu_ram": {"gte": 79000}, "cuda_max_good": {"gte": 13.2},
         "reliability": {"gte": 0.99}, "rentable": {"eq": True}, "rented": {"eq": False},
         "disk_space": {"gte": DISK_GB}, "inet_down": {"gte": 1000}, "cpu_cores_effective": {"gte": 32},
         "type": "on-demand", "allocated_storage": DISK_GB, "order": [["dph_total", "asc"]], "limit": 200}
    res = call(key, "POST", "/bundles/", q).get("offers", [])
    ok = []
    for o in res:
        cap = CAPS.get(o.get("gpu_name"))
        if cap is None or o.get("dph_total", 1e9) > cap or o.get("machine_id") in exclude:
            continue
        # value: deep-learning throughput per dollar (H100 ~2x A100 per GPU for this model)
        o["_value"] = (o.get("dlperf") or o.get("total_flops") or 1) / o["dph_total"]
        ok.append(o)
    ok.sort(key=lambda o: -o["_value"])
    return ok


def show(o):
    return (f"#{o['id']} {o['num_gpus']}x {o['gpu_name']} {o.get('gpu_ram', 0) / 1000:.0f}GB  ${o['dph_total']:.2f}/h  "
            f"rel {o.get('reliability', 0):.3f}  cpu {o.get('cpu_cores_effective', 0):.0f}  disk {o.get('disk_space', 0):.0f}GB  "
            f"down {o.get('inet_down', 0):.0f}Mbps  {o.get('geolocation', '')}  value {o['_value']:.1f}")


def instance(key, iid):
    d = call(key, "GET", f"/instances/?owner=me&limit=25", base="https://console.vast.ai/api/v1")
    for i in d.get("instances", []):
        if i.get("id") == iid:
            return i
    return None


def r2_has(s, obj):
    """Does r2:<bucket>/<obj> exist? (plain S3 HEAD via boto3 if present, else rclone)."""
    try:
        import boto3  # noqa: F401
        c = boto3.client("s3", endpoint_url=s["R2_ENDPOINT"], aws_access_key_id=s["R2_ACCESS_KEY_ID"],
                         aws_secret_access_key=s["R2_SECRET_ACCESS_KEY"], region_name="auto")
        try:
            c.head_object(Bucket=s["R2_BUCKET"], Key=obj)
            return True
        except Exception:
            return False
    except ImportError:
        import subprocess
        out = subprocess.run(["bash", "-c", f'. scripts/r2.sh && r2 lsf "r2:$R2_BUCKET/{obj}"'], cwd=ROOT,
                             capture_output=True, text=True)
        return bool(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-train-hours", type=float, default=float(os.environ.get("MAX_TRAIN_HOURS", 72)))
    ap.add_argument("--boot-timeout-min", type=int, default=40, help="instance must be running by then")
    ap.add_argument("--bootstrap-timeout-min", type=int, default=90, help="bootstrap must report by then")
    a = ap.parse_args()
    s = secrets()
    key = s["VAST_API_KEY"]
    credit = call(key, "GET", "/users/current/").get("credit", 0)
    tried = []
    while True:
        cand = offers(key, exclude=tried)
        print(f"credit ${credit:.2f}; {len(cand)} offers within caps {CAPS}", flush=True)
        for o in cand[:8]:
            print("  " + show(o), flush=True)
        if not cand:
            sys.exit("no offer within the caps right now")
        o = cand[0]
        if credit < MIN_HOURS * o["dph_total"]:
            sys.exit(f"credit ${credit:.2f} < {MIN_HOURS} h x ${o['dph_total']:.2f}/h: not renting (add credit)")
        if a.dry_run:
            print("dry run: would rent " + show(o))
            return
        onstart = (ROOT / "scripts/trainer_onstart.sh").read_text()
        env = {k: s[k] for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "WANDB_API_KEY")}
        env["MAX_TRAIN_HOURS"] = str(a.max_train_hours)
        body = {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": "ego-wm-trainer",
                "onstart": onstart, "env": env, "runtype": "ssh_direc ssh_proxy"}
        r = call(key, "PUT", f"/asks/{o['id']}/", body)
        iid = r.get("new_contract")
        if not r.get("success") or not iid:
            print(f"rent failed: {r}", flush=True)
            tried.append(o.get("machine_id"))
            continue
        print(f"rented instance {iid}: " + show(o), flush=True)
        t0 = time.time()
        status = None
        while time.time() - t0 < a.boot_timeout_min * 60:
            i = instance(key, iid) or {}
            status = i.get("actual_status")
            if status == "running":
                break
            time.sleep(60)
        if status != "running":
            print(f"instance {iid} not running after {a.boot_timeout_min} min (status {status}); destroying, next offer",
                  flush=True)
            call(key, "DELETE", f"/instances/{iid}/")
            tried.append(o.get("machine_id"))
            continue
        print(f"instance {iid} running; waiting for its bootstrap to report in R2", flush=True)
        while time.time() - t0 < a.bootstrap_timeout_min * 60:
            if r2_has(s, f"trainer/{iid}/started"):
                print(f"LAUNCHED {iid} {o['gpu_name']} {o['dph_total']:.2f}", flush=True)
                return
            time.sleep(60)
        print(f"instance {iid}: bootstrap never reported; stopping it (kept for debugging)", flush=True)
        call(key, "PUT", f"/instances/{iid}/", {"state": "stopped"})
        sys.exit(2)


if __name__ == "__main__":
    main()
