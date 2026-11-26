"""Live demo client (Mac): your webcam hands drive the world model.

  # 1. tunnel to the GPU machine running scripts/demo_server.py
  ssh -p PORT root@HOST -L 8765:localhost:8765
  # 2. run
  python scripts/demo_client.py                      # laptop webcam facing you (selfie)
  python scripts/demo_client.py --view ego           # camera looking out from your eyes
  python scripts/demo_client.py --list               # scenes to start from; then --scene N
  python scripts/demo_client.py --replay EPISODE     # no webcam: an EgoDex recording's hands
  python scripts/demo_client.py --local CKPT --data DIR   # model in-process (testing)

Window: left your camera with the tracked hands, right the generated world. Keys: q quit,
r restart the scene. Hands are sampled at 15 fps (the model's rate). Until the model is distilled
it generates slower than real time: the world then runs behind, always using your LATEST hand
poses (older ones are skipped), so it stays responsive rather than falling ever further behind.
"""
from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex.data import BONES  # noqa: E402


class Remote:
    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.f = self.sock.makefile("rwb")

    def call(self, msg):
        self.f.write((json.dumps(msg) + "\n").encode())
        self.f.flush()
        r = json.loads(self.f.readline())
        if "error" in r:
            raise RuntimeError(r["error"])
        return r


class Local:
    """Same calls as Remote, with the model in this process (tiny models / testing)."""

    def __init__(self, ckpt, data, text_emb=None):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from demo_server import Demo, jpg
        self.demo, self.jpg, self.sess = Demo(ckpt, data, text_emb), jpg, None

    def call(self, msg):
        d = self.demo
        if msg["op"] == "scenes":
            return {"scenes": d.scene_list(msg.get("n", 20))}
        if msg["op"] == "start":
            frame0, p0, text = d.scene(int(msg.get("scene", 0)), msg.get("n", 20))
            self.sess = d.session(text, msg.get("cfg", 3.0), msg.get("steps", 10))
            self.sess.start([frame0], [p0])
            return {"frames": [self.jpg(frame0)], "text": text, "fpl": d.model.fpl,
                    "sampler": d.sampler(msg.get("cfg", 3.0), msg.get("steps", 10)),
                    "size": [frame0.shape[1], frame0.shape[0]]}
        t0 = time.time()
        frames = self.sess.step(d.poses(msg["poses"]))
        return {"frames": [self.jpg(x) for x in frames], "ms": round(1000 * (time.time() - t0)),
                "reanchors": self.sess.reanchors}


def unjpg(s):
    import base64
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)  # BGR


class Replay:
    """An EgoDex episode as a stand-in webcam: its video frames and its true hand poses."""

    def __init__(self, path):
        from egodex.data import Episode
        from egodex.conditioning import joint_validity
        self.ep, self.t, self.jv = Episode(path), 0, joint_validity

    def read(self):
        if self.t >= self.ep.T - 1:
            return None, None
        fr = self.ep.read_frames([self.t], (640, 360))[0]
        v = self.jv(self.ep, [self.t])[0]
        pose = {"joints": self.ep.joints_cam(self.t).tolist(), "found": v.any(-1).tolist()}
        self.t += 2  # 15 fps
        return fr, pose


def draw_hands(bgr, uv, found):
    for h in range(2):
        if not found[h]:
            continue
        pts = uv[h].astype(int)
        for a, b in BONES:
            cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), (0, 255, 0) if h else (255, 128, 0), 2)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="localhost:8765")
    ap.add_argument("--local", default=None, help="checkpoint: run the model in this process")
    ap.add_argument("--data", default=None, help="with --local: latents dir for scenes")
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--view", default="selfie", choices=["selfie", "ego"])
    ap.add_argument("--hfov", type=float, default=65.0, help="webcam horizontal field of view (deg)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--replay", default=None, help="EgoDex episode path (no .mp4/.hdf5) instead of a webcam")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print the scenes and exit")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N model steps (testing)")
    ap.add_argument("--headless", action="store_true", help="no window; write --record only (testing)")
    ap.add_argument("--record", default=None, help="also save the side-by-side video here (.mp4)")
    a = ap.parse_args()

    backend = Local(a.local, a.data, a.text_emb) if a.local else Remote(a.server)
    if a.list:
        for s in backend.call({"op": "scenes"})["scenes"]:
            print(f"{s['i']:3d}  {s['text']}")
        return

    if a.replay:
        src, live = Replay(a.replay), None
    else:
        from egodex.live import LiveHands
        live = LiveHands(view=a.view, webcam_hfov_deg=a.hfov)
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            sys.exit("could not open the webcam (macOS: allow camera access for your terminal app)")

    r = backend.call({"op": "start", "scene": a.scene, "cfg": a.cfg, "steps": a.steps})
    fpl, (W, H) = r["fpl"], r["size"]
    print(f"scene {a.scene}: {r['text']}  ({r['sampler']})")
    world = unjpg(r["frames"][0])

    poses: list = []  # newest poses, 15 fps
    lock = threading.Lock()
    shown: queue.Queue = queue.Queue()
    stop = threading.Event()
    stats = {"ms": None, "steps": 0, "skipped": 0, "reanchors": 0}

    def worker():
        while not stop.is_set():
            with lock:
                if len(poses) < fpl:
                    batch = None
                else:
                    batch = poses[-fpl:]
                    stats["skipped"] += len(poses) - fpl
                    poses.clear()
            if batch is None:
                time.sleep(0.005)
                continue
            rr = backend.call({"op": "step", "poses": batch})
            stats.update(ms=rr["ms"], reanchors=rr["reanchors"], steps=stats["steps"] + 1)
            for fr in rr["frames"]:
                shown.put(unjpg(fr))
            if a.max_steps and stats["steps"] >= a.max_steps:
                stop.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    writer = None
    period, next_t = 1 / 15, time.time()
    try:
        while not stop.is_set():
            if a.replay:
                rgb, pose = src.read()
                if rgb is None:
                    break
                cam = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                from egodex.live import egodex_intrinsics, pose_dict
                pd = pose_dict(np.array(pose["joints"]), np.array(pose["found"]), egodex_intrinsics())
                uv_cam = pd["uv"] * (cam.shape[1] / 1920)
            else:
                ok, bgr = cap.read()
                if not ok:
                    break
                pd = live(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                pose = {"joints": pd["joints"].tolist(), "found": pd["found"].tolist()}
                cam = live.mirror(bgr)
                uv_cam = pd["uv"] * (cam.shape[1] / 1920)  # where the model will draw the hands
            with lock:
                poses.append(pose)
            try:
                while True:
                    world = shown.get_nowait()
            except queue.Empty:
                pass
            cam = cv2.resize(draw_hands(cam.copy(), uv_cam, pd["found"]), (W, H))
            view = np.concatenate([cam, cv2.resize(world, (W, H))], 1)
            msg = (f"model {stats['ms']} ms/step, {stats['steps']} steps, skipped {stats['skipped']} poses, "
                   f"re-anchors {stats['reanchors']}")
            cv2.putText(view, msg, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if a.record:
                if writer is None:
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 15, (2 * W, H))
                writer.write(view)
            if not a.headless:
                cv2.imshow("ego-wm", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    with lock:
                        poses.clear()
                    world = unjpg(backend.call({"op": "start", "scene": a.scene, "cfg": a.cfg,
                                                "steps": a.steps})["frames"][0])
            next_t += period  # pace at 15 fps
            time.sleep(max(0.0, next_t - time.time()))
            if a.replay and a.headless:  # offline test: let the model keep up with the replay
                while not stop.is_set() and len(poses) >= fpl:
                    time.sleep(0.01)
    finally:
        stop.set()
        th.join(timeout=60)
        if writer is not None:
            writer.release()
        if not a.replay:
            cap.release()
        cv2.destroyAllWindows()
    print(f"done: {stats['steps']} model steps, last {stats['ms']} ms/step, re-anchors {stats['reanchors']}")


if __name__ == "__main__":
    main()
