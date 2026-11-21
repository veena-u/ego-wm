"""Action-conditioned world model on top of a pretrained Wan 2.1 transformer.

What is added to the pretrained network (all zero-initialized, so at step 0 the model is
exactly the pretrained one):
  1. Spatial hand conditioning: the patch embedding gets extra input channels for the
     per-joint heatmaps + depth of every video frame that a latent frame covers.
  2. Vector actions (head pose, 3D joints, visibility) -> MLP -> added to the timestep
     embedding of every token of that latent frame, so they modulate AdaLN.
  3. Per-frame noise levels (Diffusion Forcing): each latent frame has its own sigma; clean
     context frames have sigma=0. diffusers' Wan already accepts per-token timesteps.
  4. Optional block-causal attention: a latent frame attends only to itself and earlier
     frames, which is what lets the model roll out frame-by-frame interactively.

Training objective: rectified flow as in Wan. x_sigma = (1 - sigma) x0 + sigma * eps, the
network predicts v = eps - x0, timestep = 1000 * sigma.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_wan import WanAttnProcessor, WanTransformer3DModel


class _MaskedSelfAttn(WanAttnProcessor):
    """Wan attention processor that applies `self.mask` to self-attention only."""

    def __init__(self):
        super().__init__()
        self.mask = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is None and attention_mask is None:
            attention_mask = self.mask
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


def block_causal_mask(num_frames: int, tokens_per_frame: int, device) -> torch.Tensor:
    f = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    return (f[None, :] <= f[:, None])[None, None]  # (1,1,S,S) True = may attend


class ActionWorldModel(nn.Module):
    def __init__(self, transformer: WanTransformer3DModel, spatial_channels: int, action_dim: int,
                 frames_per_latent: int, causal: bool = True):
        super().__init__()
        self.tr = transformer
        self.fpl = frames_per_latent
        self.causal = causal
        D = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        self.latent_channels = transformer.config.in_channels
        self.spatial_in = spatial_channels * frames_per_latent

        # Separate conv for the hand channels, added to the pretrained patch embedding (conv is
        # linear, so this equals widening it) - keeps pretrained weights in their own LR group.
        pe = transformer.patch_embedding
        self.spatial_embed = nn.Conv3d(self.spatial_in, pe.out_channels, pe.kernel_size, pe.stride, bias=False)
        nn.init.zeros_(self.spatial_embed.weight)

        self.action_mlp = nn.Sequential(nn.Linear(action_dim * frames_per_latent, D), nn.SiLU(), nn.Linear(D, D))
        nn.init.zeros_(self.action_mlp[-1].weight)
        nn.init.zeros_(self.action_mlp[-1].bias)

        self._proc = _MaskedSelfAttn()
        for blk in transformer.blocks:
            blk.attn1.processor = self._proc

    def enable_gradient_checkpointing(self):
        self.tr.enable_gradient_checkpointing()

    def new_parameters(self):
        return list(self.spatial_embed.parameters()) + list(self.action_mlp.parameters())

    def forward(self, x, sigma, spatial, actions, text):
        """x: (B,C,F,h,w) noisy latents; sigma: (B,F) in [0,1]; spatial: (B,S*fpl,F,h,w);
        actions: (B,F,A*fpl) normalized; text: (B,L,text_dim). Returns predicted v (B,C,F,h,w)."""
        tr, cfg = self.tr, self.tr.config
        B, C, Fr, h, w = x.shape
        p_t, p_h, p_w = cfg.patch_size
        ppf, pph, ppw = Fr // p_t, h // p_h, w // p_w
        tpf = pph * ppw  # tokens per latent frame

        rotary = tr.rope(x)
        hs = tr.patch_embedding(x) + self.spatial_embed(spatial.to(x.dtype))
        hs = hs.flatten(2).transpose(1, 2).contiguous()

        ce = tr.condition_embedder
        t = (sigma * 1000).float()  # (B,F)
        tt = ce.timesteps_proj(t.flatten()).unflatten(0, (B, Fr))
        te_dtype = next(iter(ce.time_embedder.parameters())).dtype
        temb = ce.time_embedder(tt.to(te_dtype))  # (B,F,D)
        temb = temb + self.action_mlp(actions.to(temb.dtype))
        temb = temb.repeat_interleave(tpf, dim=1).type_as(hs)  # (B,S,D)
        tproj = ce.time_proj(ce.act_fn(temb)).unflatten(2, (6, -1))  # (B,S,6,D)
        enc = ce.text_embedder(text.to(hs.dtype))

        # The mask must stay set after forward: gradient checkpointing re-runs the blocks during
        # backward. Every forward overwrites it, so it never leaks between calls.
        self._proc.mask = block_causal_mask(ppf, tpf, x.device) if self.causal else None
        for blk in tr.blocks:
            if torch.is_grad_enabled() and tr.gradient_checkpointing:
                hs = tr._gradient_checkpointing_func(blk, hs, enc, tproj, rotary)
            else:
                hs = blk(hs, enc, tproj, rotary)

        shift, scale = (tr.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        hs = (tr.norm_out(hs.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(hs)
        hs = tr.proj_out(hs)
        hs = hs.reshape(B, ppf, pph, ppw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def sample_sigmas(B, F, n_ctx, shift, device, generator=None):
    """Diffusion Forcing noise levels: independent per latent frame, shifted toward high noise
    like Wan (shift>1). The first n_ctx[b] frames are clean context (sigma=0)."""
    u = torch.rand(B, F, device=device, generator=generator)
    sigma = shift * u / (1 + (shift - 1) * u)
    ctx = torch.arange(F, device=device)[None] < n_ctx[:, None]
    return sigma.masked_fill(ctx, 0.0), ~ctx


def flow_loss(model, x0, spatial, actions, text, n_ctx, shift=3.0, generator=None):
    B, C, Fr = x0.shape[:3]
    sigma, loss_mask = sample_sigmas(B, Fr, n_ctx, shift, x0.device, generator)
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    s = sigma[:, None, :, None, None].to(x0.dtype)
    xt = (1 - s) * x0 + s * eps
    v = model(xt, sigma, spatial, actions, text)
    per_frame = F.mse_loss(v.float(), (eps - x0).float(), reduction="none").mean(dim=(1, 3, 4))  # (B,F)
    return (per_frame * loss_mask).sum() / loss_mask.sum().clamp(min=1), per_frame.detach(), sigma


def tiny_wan(latent_channels: int, text_dim: int = 64) -> WanTransformer3DModel:
    """Randomly initialized miniature Wan for CPU/Mac smoke tests."""
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=48,
                                 in_channels=latent_channels, out_channels=latent_channels, text_dim=text_dim,
                                 freq_dim=64, ffn_dim=128, num_layers=2, rope_max_seq_len=256)


def load_wan(repo: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", dtype=torch.bfloat16) -> WanTransformer3DModel:
    return WanTransformer3DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dtype)
