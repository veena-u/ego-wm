"""Streaming (interactive) generation with a KV cache.

The causal world model generates one latent frame at a time: frame k attends only to frames <= k,
and frames before k are always fed CLEAN (sigma = 0). So once a frame is finished, its keys and
values in every layer never change again - they depend only on itself and earlier clean frames.
We compute them once and cache them, and each new frame only runs the transformer over its own
tokens. This is exact: with no eviction the output equals the full-recompute `generate` loop
(checked by scripts/test_stream.py).

Cost per new latent frame: (steps + 1) transformer passes over ONE frame's tokens, attending to
the cached window - instead of `steps` passes over ALL frames so far. The +1 pass writes the
finished frame into the cache.

Rolling window: keep at most `window` past frames (optionally pinning the first `sink` frames,
like attention sinks in streaming LLMs), so memory and per-frame cost stay constant and the
stream can run indefinitely. Positions stay absolute; RoPE makes attention depend only on
relative position, so evicting old frames doesn't shift anything. The model was trained with
<= 8 past latent frames, so window <= 8 keeps it in-distribution.
"""
from __future__ import annotations

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttnProcessor, _get_qkv_projections

from .model import ActionWorldModel


def _rope(x, freqs_cos, freqs_sin):
    # Identical to diffusers' WanAttnProcessor.apply_rotary_emb
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


class _KVCache:
    def __init__(self, n_layers, window=None, sink=0):
        self.k = [[] for _ in range(n_layers)]  # per layer: list of per-frame (B,T,H,Dh)
        self.v = [[] for _ in range(n_layers)]
        self.window, self.sink = window, sink
        self.write = False

    def __len__(self):
        return len(self.k[0])

    def evict(self):
        if self.window is None:
            return
        for kv in (self.k, self.v):
            for frames in kv:
                while len(frames) > self.window:
                    del frames[self.sink]


class _CachedSelfAttn(WanAttnProcessor):
    """Self-attention for the new frame's tokens over [cached past frames, new frame]."""

    def __init__(self, layer, cache: _KVCache):
        super().__init__()
        self.layer, self.cache = layer, cache

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        q, k, v = _get_qkv_projections(attn, hidden_states, None)
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k, v = (t.unflatten(2, (attn.heads, -1)) for t in (q, k, v))
        if rotary_emb is not None:
            q, k = _rope(q, *rotary_emb), _rope(k, *rotary_emb)
        ck, cv = self.cache.k[self.layer], self.cache.v[self.layer]
        K = torch.cat(ck + [k], 1) if ck else k
        V = torch.cat(cv + [v], 1) if cv else v
        if self.cache.write:
            ck.append(k)
            cv.append(v)
        out = dispatch_attention_fn(q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](out))


class StreamingWorldModel:
    """Usage:
        s = StreamingWorldModel(model, text, window=8)
        s.push(first_latent, spatial0, action0)          # real context frame(s)
        for each new hand pose:
            z = s.next(spatial_t, action_t, steps=30)    # (1,C,1,h,w), also appended to the cache
    spatial/action are for ONE latent frame: (1,S*fpl,1,h,w) and (1,1,A*fpl).
    """

    def __init__(self, model: ActionWorldModel, text, window: int | None = None, sink: int = 0,
                 cfg: float = 1.0):
        """cfg > 1: classifier-free guidance on the HAND conditioning (spatial maps + action vector),
        v = v_uncond + cfg * (v_cond - v_uncond). The unconditional branch zeroes both, exactly as
        conditioning dropout did in training; the text prompt is kept in both branches. The two
        branches run as one batch of 2 with separate caches (their past K/V differ)."""
        assert model.causal, "streaming needs the block-causal model"
        self.model, self.cfg = model, cfg
        self.text = torch.cat([text, text]) if cfg != 1.0 else text
        blocks = model.tr.blocks
        self.cache = _KVCache(len(blocks), window, sink)
        self._procs = [_CachedSelfAttn(i, self.cache) for i in range(len(blocks))]
        self.t = 0  # absolute index of the next latent frame

    def _cond(self, spatial, action):
        if self.cfg == 1.0:
            return spatial, action
        return torch.cat([spatial, torch.zeros_like(spatial)]), torch.cat([action, torch.zeros_like(action)])

    def _run(self, x, sigma, spatial, action, write):
        blocks = self.model.tr.blocks
        saved = [b.attn1.processor for b in blocks]
        for b, p in zip(blocks, self._procs):
            b.attn1.processor = p
        self.cache.write = write
        try:
            sig = torch.full((x.shape[0], 1), float(sigma), device=x.device)
            return self.model(x, sig, spatial, action, self.text, frame_offset=self.t)
        finally:
            self.cache.write = False
            for b, p in zip(blocks, saved):
                b.attn1.processor = p

    def _v(self, x, sigma, spatial, action, write):
        sp, ac = self._cond(spatial, action)
        if self.cfg == 1.0:
            return self._run(x, sigma, sp, ac, write)
        v = self._run(torch.cat([x, x]), sigma, sp, ac, write)
        vc, vu = v.chunk(2)
        return vu + self.cfg * (vc - vu)

    @torch.no_grad()
    def push(self, latent, spatial, action):
        """Append a known clean latent frame (context) to the cache."""
        self._v(latent, 0.0, spatial, action, write=True)
        self.t += 1
        self.cache.evict()

    @torch.no_grad()
    def next(self, spatial, action, steps=30, shift=3.0, generator=None, noise=None):
        """Generate the next latent frame from noise given the cached past, then cache it."""
        from_sig = torch.linspace(1, 0, steps + 1)
        sig = (shift * from_sig / (1 + (shift - 1) * from_sig)).tolist()
        p = next(self.model.parameters())
        shape = (spatial.shape[0], self.model.latent_channels, 1, *spatial.shape[-2:])
        x = noise.clone() if noise is not None else torch.randn(shape, device=p.device, generator=generator, dtype=p.dtype)
        for i in range(steps):
            v = self._v(x, sig[i], spatial, action, write=False)
            x = x + (sig[i + 1] - sig[i]) * v.to(x.dtype)
        self.push(x, spatial, action)  # sigma=0 pass writes the finished frame's K/V
        return x

    @torch.no_grad()
    def next_few(self, spatial, action, sigmas, generator=None, noise=None):
        """Few-step sampler of a distilled student (scripts/distill.py): at each noise level predict
        x0, then re-noise it to the next level with fresh noise; the last x0 is the frame."""
        p = next(self.model.parameters())
        shape = (spatial.shape[0], self.model.latent_channels, 1, *spatial.shape[-2:])
        x = noise.clone() if noise is not None else torch.randn(shape, device=p.device, generator=generator, dtype=p.dtype)
        for i, s in enumerate(sigmas):
            x0 = x - s * self._v(x, s, spatial, action, write=False).to(x.dtype)
            if i + 1 < len(sigmas):
                n = torch.randn(shape, device=p.device, generator=generator, dtype=p.dtype)
                x = (1 - sigmas[i + 1]) * x0 + sigmas[i + 1] * n
        self.push(x0, spatial, action)
        return x0


@torch.no_grad()
def stream_generate(model, ctx, spatial, actions, text, n_ctx=1, steps=30, shift=3.0, generator=None,
                    cfg=1.0, window=None, student_sigmas=None):
    """Drop-in for the full-recompute `generate` (same arguments and noise draws), via the cache.
    student_sigmas: a distilled student's few-step sampler instead (guidance is baked in)."""
    s = StreamingWorldModel(model, text, window=window, cfg=cfg)
    for k in range(n_ctx):
        s.push(ctx[:, :, k:k + 1], spatial[:, :, k:k + 1], actions[:, k:k + 1])
    out = [ctx[:, :, :n_ctx]]
    for k in range(n_ctx, ctx.shape[2]):
        if student_sigmas is not None:
            out.append(s.next_few(spatial[:, :, k:k + 1], actions[:, k:k + 1], student_sigmas, generator).to(ctx.dtype))
        else:
            out.append(s.next(spatial[:, :, k:k + 1], actions[:, k:k + 1], steps, shift, generator).to(ctx.dtype))
    return torch.cat(out, 2)
