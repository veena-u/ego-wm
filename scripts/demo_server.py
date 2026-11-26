"""Live demo server (GPU machine): hand poses in, generated frames out, over a local TCP socket.

  python scripts/demo_server.py --ckpt runs/v3/weights/stepN.pt --data /workspace/data/latents \
      --text-emb /workspace/data/text_emb.pt --port 8765
Mac:  ssh -p PORT root@HOST -L 8765:localhost:8765      then   python scripts/demo_client.py

Listens on 127.0.0.1 only: reach it through the ssh tunnel. Protocol: one JSON object per line.
  -> {"op": "scenes", "n": 20}                      <- {"scenes": [{"i", "text"}...]}
  -> {"op": "start", "scene": i, "cfg": 3, "steps": 10}  (or "image": base64 jpeg + "text")
                                                    <- {"frames": [jpeg...], "text", "size"}
  -> {"op": "step", "poses": [{"joints": 2x21x3, "found": [l, r]} x fpl]}
                                                    <- {"frames": [jpeg x fpl], "ms", "reanchors"}
Joints are meters in the EgoDex camera frame (egodex/live.py makes them from a webcam).
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egodex.conditioning import ACTION_DIM, NUM_SPATIAL_CHANNELS  # noqa: E402
from egodex.dataset import LatentClips, TextEmbeddings  # noqa: E402
from egodex.live import egodex_intrinsics, pose_dict  # noqa: E402
from egodex.live_session import LiveSession, VAEStream  # noqa: E402
from egodex.model import ActionWorldModel, load_wan, tiny_wan  # noqa: E402
from egodex.vae import load_vae  # noqa: E402


def jpg(rgb, q=85) -> str:
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(b.tobytes()).decode()


def unjpg(s: str):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Demo:
    """Model + VAE + scenes; used by the server, and in-process by demo_client.py --local."""

    def __init__(self, ckpt, data, text_emb=None, device=None, action_stats="data/action_stats.npz",
                 splits=("val_unseen", "val_seen")):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        margs = ck["args"]
        tiny = margs["model"] == "tiny"
        tr = tiny_wan(ck["latent_channels"], 64) if tiny else load_wan(margs["wan_repo"], dtype=torch.float32)
        m = ActionWorldModel(tr, NUM_SPATIAL_CHANNELS, ACTION_DIM, ck["frames_per_latent"], margs["causal"])
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        dt = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = m.to(self.device, dt).eval()
        self.shift = margs.get("shift", 3.0)
        self.student = ck.get("student")  # distilled: few-step sampler with guidance baked in
        self.action_stats = action_stats
        self.text = TextEmbeddings(text_emb, 64 if tiny else 4096, 16 if tiny else 512)
        self.ds = LatentClips(data, splits, action_stats=action_stats, text_emb=text_emb,
                              text_dim=64 if tiny else 4096, text_len=16 if tiny else 512)
        meta = self.ds._load(0)["meta"]
        self.vae = load_vae(meta["vae"], self.device, dtype=dt, tiling=False)
        # one clip per episode, spread over tasks: the scene list
        seen, self.scenes = set(), []
        for idx, (fi, ci) in enumerate(self.ds.items):
            if fi not in seen and ci == 0:
                seen.add(fi)
                self.scenes.append(idx)
        self.K = egodex_intrinsics()

    def scene_list(self, n=20):
        out = []
        for k, idx in enumerate(self.scenes[:: max(1, len(self.scenes) // n)][:n]):
            out.append({"i": k, "text": self.ds._load(self.ds.items[idx][0])["text"]})
        return out

    def scene(self, k, n=20):
        """Scene k of scene_list(n): its first frame and that frame's hand pose."""
        idx = self.scenes[:: max(1, len(self.scenes) // n)][:n][k]
        fi, ci = self.ds.items[idx]
        d = self.ds._load(fi)
        z0 = d["latents"][ci][:, :1].float()[None].to(self.device, next(self.vae.parameters()).dtype
                                                      if list(self.vae.parameters()) else torch.float32)
        vs = VAEStream(self.vae)
        frame0 = vs.decode_next(z0)[0]
        act = d["actions"][ci][0].numpy()  # raw (unnormalized) action vector of video frame 0
        joints = act[9:135].reshape(2, 21, 3).astype(np.float64)
        found = d["valid"][ci][0].numpy().any(-1)
        return frame0, pose_dict(joints, found, self.K), d["text"]

    def sampler(self, cfg, steps):
        if self.student:
            return f"distilled student, {self.student['steps']} steps, guidance {self.student['cfg']} built in"
        return f"{steps} steps, guidance {cfg}"

    def session(self, text, cfg=3.0, steps=10, seed=0):
        emb = self.text(text)[None].to(self.device, next(self.model.parameters()).dtype)
        if self.student:
            return LiveSession(self.model, self.vae, emb, cfg=1.0, action_stats=self.action_stats, seed=seed,
                               student_sigmas=self.student["sigmas"])
        return LiveSession(self.model, self.vae, emb, cfg=cfg, steps=steps, shift=self.shift,
                           action_stats=self.action_stats, seed=seed)

    def poses(self, msg_poses):
        return [pose_dict(np.asarray(p["joints"], np.float64), np.asarray(p["found"], bool), self.K) for p in msg_poses]


def serve(demo: Demo, port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print(f"demo server on 127.0.0.1:{port} ({len(demo.scenes)} scenes)", flush=True)
    while True:
        conn, _ = srv.accept()
        print("client connected", flush=True)
        f = conn.makefile("rwb")
        sess = None
        try:
            for line in f:
                msg = json.loads(line)
                if msg["op"] == "scenes":
                    reply = {"scenes": demo.scene_list(msg.get("n", 20))}
                elif msg["op"] == "start":
                    if "image" in msg:
                        frame0 = unjpg(msg["image"])
                        p0 = demo.poses([msg["pose"]])[0]
                        text = msg.get("text", "")
                    else:
                        frame0, p0, text = demo.scene(int(msg.get("scene", 0)), msg.get("n", 20))
                    sess = demo.session(msg.get("text") or text, msg.get("cfg", 3.0), msg.get("steps", 10))
                    sess.start([frame0], [p0])
                    H, W = frame0.shape[:2]
                    reply = {"frames": [jpg(frame0)], "text": msg.get("text") or text, "size": [W, H],
                             "fpl": demo.model.fpl, "sampler": demo.sampler(msg.get("cfg", 3.0), msg.get("steps", 10))}
                elif msg["op"] == "step":
                    t0 = time.time()
                    frames = sess.step(demo.poses(msg["poses"]))
                    reply = {"frames": [jpg(x) for x in frames], "ms": round(1000 * (time.time() - t0)),
                             "reanchors": sess.reanchors}
                else:
                    reply = {"error": f"unknown op {msg['op']}"}
                f.write((json.dumps(reply) + "\n").encode())
                f.flush()
        except (ConnectionError, BrokenPipeError) as e:
            print(f"client gone: {e}", flush=True)
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="latents dir for scenes (the validation split)")
    ap.add_argument("--text-emb", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    serve(Demo(a.ckpt, a.data, a.text_emb, a.device), a.port)


if __name__ == "__main__":
    main()
