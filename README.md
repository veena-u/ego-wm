# ego-wm

An action-conditioned video world model trained on EgoDex. It starts from one real frame. After
that, each new frame is generated from the past frames plus the hand pose and head motion *for that
frame only*. It never needs future poses, so poses can arrive live from a hand tracker, one frame
at a time. It is a fine-tuned Wan 2.1 1.3B (8× VAE) with:
- the hands fed in as per-joint heatmaps, per-hand depth and forearm (wrist→elbow) maps at latent
  resolution (46 channels per frame);
- a vector action passed through AdaLN, made of:
  - head motion relative to the previous frame (so a session of any length looks the same as training),
  - 3D hand joints and elbows in the current camera frame,
  - per-hand visibility;
- a separate noise level per frame (Diffusion Forcing);
- block-causal attention: a frame sees only itself and earlier frames.

The model is causal, but it is not yet real-time. Still missing:
- a KV cache and few-step distillation,
- training on its own rollouts (Self-Forcing) so errors don't compound,
- a rolling context window for sessions longer than the 2.2 s training clips.
Each latent frame covers 4 video frames, which adds about 267 ms of buffering at 15 fps.
