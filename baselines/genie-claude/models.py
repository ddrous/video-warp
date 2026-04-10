"""
models.py — Genie (Discrete) for MovingMNIST
=============================================

Faithful discrete-token Genie (Bruce et al., ICML 2024), implemented
in JAX + Equinox.  All quantisation uses the straight-through gradient
estimator (Bengio et al., 2013).

Architecture
------------
  VectorQuantizer        : codebook + straight-through STE
  GenieEncoder           : CNN → flat → linear → d_vq
  GenieDecoder           : d_vq → linear → reshape → transposed-CNN
  GenieVQVAE             : encoder + VQ + decoder  (Phase 1)
  GenieIDM               : MLP + VQ for discrete action tokens  (Phase 2)
  GenieDynamicsTransformer : causal Transformer on (z_idx, a_idx) sequences
  GenieGCM               : GRU prior over action tokens  (Phase 3)
  Genie                  : full model combining all components
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────
# 1. Vector Quantizer  (straight-through estimator)
# ─────────────────────────────────────────────────────────────────

class VectorQuantizer(eqx.Module):
    """
    VQ bottleneck with straight-through gradient estimator.

    Implements the VQ-VAE objective from van den Oord et al. (2017):
        L_vq = ||sg(z_e) - e||^2 + beta * ||z_e - sg(e)||^2

    where z_e is the encoder output and e is the nearest codebook entry.

    For the forward pass, the straight-through trick is applied so that
    gradients flow to the encoder as if no quantisation occurred:
        z_q = z_e + sg(e - z_e)       (forward = e, backward ≈ z_e)
    """
    codebook: jax.Array                 # (K, d)
    K: int   = eqx.field(static=True)  # codebook size
    d: int   = eqx.field(static=True)  # embedding dimension
    beta: float = eqx.field(static=True)  # commitment loss weight

    def __init__(self, K: int, d: int, beta: float = 0.25, *, key):
        self.K    = K
        self.d    = d
        self.beta = beta
        # Uniform init; kaiming-like scale keeps norms comparable to encoder
        self.codebook = jax.random.uniform(key, (K, d), minval=-1/math.sqrt(d),
                                            maxval=1/math.sqrt(d))

    def quantize(self, z_e: jnp.ndarray):
        """
        z_e : (..., d)
        Returns
        -------
        indices  : (...,)      nearest-neighbor indices ∈ {0..K-1}
        z_q_st   : (..., d)   quantised embedding with straight-through gradient
        vq_loss  : scalar      codebook_loss + beta * commitment_loss
        """
        flat = z_e.reshape(-1, self.d)           # (N, d)

        # Squared Euclidean distance via ||a-b||^2 = ||a||^2 - 2a·b + ||b||^2
        z_sq = jnp.sum(flat ** 2, axis=-1, keepdims=True)          # (N,1)
        c_sq = jnp.sum(self.codebook ** 2, axis=-1, keepdims=True).T  # (1,K)
        zc   = flat @ self.codebook.T                                # (N,K)
        dists = z_sq - 2.0 * zc + c_sq                              # (N,K)

        idx_flat = jnp.argmin(dists, axis=-1)     # (N,)
        e_flat   = self.codebook[idx_flat]         # (N, d)

        # Straight-through: forward uses e, backward uses z_e
        z_q_flat_st = flat + jax.lax.stop_gradient(e_flat - flat)

        # VQ-VAE losses
        codebook_loss  = jnp.mean((e_flat - jax.lax.stop_gradient(flat)) ** 2)
        commitment_loss = jnp.mean((jax.lax.stop_gradient(e_flat) - flat) ** 2)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Reshape to original shape
        indices  = idx_flat.reshape(z_e.shape[:-1])
        z_q_st   = z_q_flat_st.reshape(z_e.shape)
        return indices, z_q_st, vq_loss

    def decode(self, indices: jnp.ndarray) -> jnp.ndarray:
        """indices : (...,) → e : (..., d)"""
        return self.codebook[indices]


# ─────────────────────────────────────────────────────────────────
# 2. Encoder / Decoder
# ─────────────────────────────────────────────────────────────────

class GenieEncoder(eqx.Module):
    """CNN encoder: (C, H, W) → (d_vq,)"""
    conv_layers: list
    proj: eqx.nn.Linear

    def __init__(self, in_C: int, d_vq: int,
                 hidden_width: int, depth: int, *, key):
        keys = jax.random.split(key, depth + 1)
        convs, ch_in, ch_out = [], in_C, hidden_width
        for i in range(depth):
            convs.append(eqx.nn.Conv2d(ch_in, ch_out, kernel_size=3,
                                        stride=2, padding=1, key=keys[i]))
            ch_in  = ch_out
            ch_out = min(ch_out * 2, 512)
        self.conv_layers = convs
        # Compute flattened dimension with a dummy forward
        dummy = jnp.zeros((in_C, 64, 64))
        for l in self.conv_layers:
            dummy = jax.nn.relu(l(dummy))
        flat_dim = dummy.reshape(-1).shape[0]
        self.proj = eqx.nn.Linear(flat_dim, d_vq, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (C, H, W) → z_e: (d_vq,)"""
        for l in self.conv_layers:
            x = jax.nn.relu(l(x))
        return self.proj(x.reshape(-1))


class GenieDecoder(eqx.Module):
    """Transposed-CNN decoder: (d_vq,) → (C, H, W)"""
    proj: eqx.nn.Linear
    deconv_layers: list
    out_conv: eqx.nn.Conv2d
    bot_C:  int   = eqx.field(static=True)
    bot_HW: tuple = eqx.field(static=True)

    def __init__(self, d_vq: int, out_C: int,
                 H: int, W: int,
                 hidden_width: int, depth: int, *, key):
        keys = jax.random.split(key, depth + 2)
        H_bot = H  // (2 ** depth)
        W_bot = W  // (2 ** depth)
        C_bot = min(hidden_width * (2 ** (depth - 1)), 512)
        self.bot_C  = C_bot
        self.bot_HW = (H_bot, W_bot)

        self.proj = eqx.nn.Linear(d_vq, C_bot * H_bot * W_bot, key=keys[0])
        deconvs, ch = [], C_bot
        for i in range(depth):
            ch_out = max(ch // 2, hidden_width)
            deconvs.append(eqx.nn.ConvTranspose2d(
                ch, ch_out, kernel_size=4, stride=2, padding=1, key=keys[i + 1]))
            ch = ch_out
        self.deconv_layers = deconvs
        self.out_conv = eqx.nn.Conv2d(ch, out_C, kernel_size=3, padding=1, key=keys[-1])

    def __call__(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """z_q: (d_vq,) → x_hat: (C, H, W)  in [0, 1]"""
        H_bot, W_bot = self.bot_HW
        x = jax.nn.relu(self.proj(z_q)).reshape(self.bot_C, H_bot, W_bot)
        for l in self.deconv_layers:
            x = jax.nn.relu(l(x))
        # return jax.nn.sigmoid(self.out_conv(x))
        return self.out_conv(x)


# ─────────────────────────────────────────────────────────────────
# 3. IDM / Latent Action Model  (discrete actions via VQ)
# ─────────────────────────────────────────────────────────────────

class GenieIDM(eqx.Module):
    """
    Inverse Dynamics Model producing discrete action tokens.

    Maps (z_q_t, z_q_{t+1}) → (action_idx, action_embedding, vq_loss).
    An internal MLP first maps the pair to an action-space vector,
    which is then quantised via a separate action codebook.
    """
    mlp: eqx.nn.MLP
    action_vq: VectorQuantizer

    def __init__(self, d_vq: int, action_K: int,
                 action_d: int, beta: float, *, key):
        k1, k2 = jax.random.split(key)
        self.mlp       = eqx.nn.MLP(d_vq * 2, action_d,
                                      width_size=d_vq * 2, depth=3, key=k1)
        self.action_vq = VectorQuantizer(action_K, action_d, beta=beta, key=k2)

    def __call__(self, z_q_t: jnp.ndarray, z_q_tp1: jnp.ndarray):
        """
        Returns (action_idx: int,  action_emb: (action_d,),  vq_loss: scalar)
        """
        a_cont = self.mlp(jnp.concatenate([z_q_t, z_q_tp1]))   # (action_d,)
        a_idx, a_emb, vq_loss = self.action_vq.quantize(a_cont)
        return a_idx, a_emb, vq_loss


# ─────────────────────────────────────────────────────────────────
# 4. Causal Transformer Block
# ─────────────────────────────────────────────────────────────────

class CausalBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attn:  eqx.nn.MultiheadAttention
    ff1:   eqx.nn.Linear
    ff2:   eqx.nn.Linear

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.attn  = eqx.nn.MultiheadAttention(n_heads, d_model, key=k1)
        self.ff1   = eqx.nn.Linear(d_model, d_ff,    key=k2)
        self.ff2   = eqx.nn.Linear(d_ff,    d_model, key=k3)

    def __call__(self, x: jnp.ndarray,
                 mask: jnp.ndarray) -> jnp.ndarray:
        """x: (T, d_model)  mask: (T, T) bool, True = attend"""
        xn = jax.vmap(self.norm1)(x)
        x  = x + self.attn(xn, xn, xn, mask=mask)
        xn = jax.vmap(self.norm2)(x)
        x  = x + jax.vmap(lambda v: self.ff2(jax.nn.gelu(self.ff1(v))))(xn)
        return x


# ─────────────────────────────────────────────────────────────────
# 5. Dynamics Transformer
# ─────────────────────────────────────────────────────────────────

class GenieDynamicsTransformer(eqx.Module):
    """
    Causal autoregressive transformer that predicts the next frame token
    from the current frame token and the action taken.

    Input sequence  : [(z_1, a_1), (z_2, a_2), …, (z_{T-1}, a_{T-1})]
    Target sequence : [z_2, z_3, …, z_T]
    Loss            : cross-entropy over frame codebook

    The action embedding and frame embedding are summed after projection
    to produce one token per timestep, keeping the sequence length = T-1.
    """
    frame_emb:  eqx.nn.Embedding   # (frame_K, d_model)
    action_emb: eqx.nn.Embedding   # (action_K, d_model)
    input_proj: eqx.nn.Linear      # (2*d_model → d_model)
    pos_emb:    jax.Array           # (max_T, d_model)  learnable
    blocks:     list
    out_head:   eqx.nn.Linear      # (d_model → frame_K)

    frame_K:  int = eqx.field(static=True)
    action_K: int = eqx.field(static=True)
    d_model:  int = eqx.field(static=True)
    max_T:    int = eqx.field(static=True)

    def __init__(self, frame_K: int, action_K: int,
                 d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_T: int, *, key):
        keys = jax.random.split(key, n_layers + 5)
        self.frame_K  = frame_K
        self.action_K = action_K
        self.d_model  = d_model
        self.max_T    = max_T

        self.frame_emb  = eqx.nn.Embedding(frame_K,  d_model, key=keys[0])
        self.action_emb = eqx.nn.Embedding(action_K, d_model, key=keys[1])
        self.input_proj = eqx.nn.Linear(2 * d_model, d_model, key=keys[2])
        self.pos_emb    = jax.random.normal(keys[3], (max_T, d_model)) * 0.02
        self.blocks     = [CausalBlock(d_model, n_heads, d_ff, key=keys[4 + i])
                           for i in range(n_layers)]
        self.out_head   = eqx.nn.Linear(d_model, frame_K, key=keys[-1])

    def __call__(self, z_idx: jnp.ndarray,
                 a_idx: jnp.ndarray) -> jnp.ndarray:
        """
        z_idx : (T-1,)  frame token indices  z_{1..T-1}
        a_idx : (T-1,)  action token indices a_{1..T-1}
        Returns logits : (T-1, frame_K)  — predict z_{2..T}
        """
        T = z_idx.shape[0]
        z_e = jax.vmap(self.frame_emb)(z_idx)      # (T-1, d_model)
        a_e = jax.vmap(self.action_emb)(a_idx)     # (T-1, d_model)
        # Fuse: concat → project
        tok = jax.vmap(self.input_proj)(
            jnp.concatenate([z_e, a_e], axis=-1))  # (T-1, d_model)
        tok = tok + self.pos_emb[:T]
        # Causal mask: position i attends to positions 0..i
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        for blk in self.blocks:
            tok = blk(tok, mask)
        return jax.vmap(self.out_head)(tok)         # (T-1, frame_K)


# ─────────────────────────────────────────────────────────────────
# 6. Generative Control Model  (GCM — Phase 3)
# ─────────────────────────────────────────────────────────────────

class GenieGCM(eqx.Module):
    """
    GRU-based behavioural prior over discrete action tokens.

    Predicts the action distribution at step t from:
        (frame token z_t, previous action token a_{t-1})
    """
    frame_emb:  eqx.nn.Embedding   # (frame_K, d_h)
    action_emb: eqx.nn.Embedding   # (action_K, d_h)
    rnn:        eqx.nn.GRUCell
    out_head:   eqx.nn.MLP

    frame_K:  int = eqx.field(static=True)
    action_K: int = eqx.field(static=True)
    d_h:      int = eqx.field(static=True)

    def __init__(self, frame_K: int, action_K: int, d_h: int, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.frame_K  = frame_K
        self.action_K = action_K
        self.d_h      = d_h
        self.frame_emb  = eqx.nn.Embedding(frame_K,  d_h, key=k1)
        self.action_emb = eqx.nn.Embedding(action_K, d_h, key=k2)
        self.rnn        = eqx.nn.GRUCell(2 * d_h, d_h, key=k3)
        self.out_head   = eqx.nn.MLP(d_h + d_h, action_K,
                                      width_size=d_h, depth=2, key=k4)

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros(self.d_h)

    def step(self, h: jnp.ndarray,
             z_idx: jnp.ndarray,
             a_prev_idx: jnp.ndarray):
        """
        One GCM step.
        Returns (h_new, logits_over_action_K)
        """
        z_e  = self.frame_emb(z_idx)
        a_e  = self.action_emb(a_prev_idx)
        h_new = self.rnn(jnp.concatenate([z_e, a_e]), h)
        logits = self.out_head(jnp.concatenate([h_new, z_e]))  # condition on z too
        return h_new, logits


# ─────────────────────────────────────────────────────────────────
# 7. Full Genie Model
# ─────────────────────────────────────────────────────────────────

class Genie(eqx.Module):
    """
    Genie (Discrete): complete model combining all five components.

    Phase 1 : train encoder + VQ + decoder  (VQVAE)
    Phase 2 : train IDM + DynamicsTransformer  (encoder/VQ/decoder frozen)
    Phase 3 : train GCM  (all other components frozen)
    """
    encoder:  GenieEncoder
    decoder:  GenieDecoder
    vq:       VectorQuantizer
    idm:      GenieIDM
    dynamics: GenieDynamicsTransformer
    gcm:      GenieGCM

    frame_K:  int = eqx.field(static=True)
    action_K: int = eqx.field(static=True)
    d_vq:     int = eqx.field(static=True)

    def __init__(self, cfg: dict,
                 frame_shape: Tuple[int, int, int], *, key):
        H, W, C = frame_shape
        d_vq      = cfg["vq_dim"]
        frame_K   = cfg["frame_codebook_size"]
        action_K  = cfg["action_codebook_size"]
        action_d  = cfg["action_dim"]
        vq_beta   = cfg.get("vq_beta", 0.25)
        act_beta  = cfg.get("action_beta", 0.25)
        hw        = cfg["cnn_hidden_width"]
        cdep      = cfg["cnn_depth"]

        self.frame_K  = frame_K
        self.action_K = action_K
        self.d_vq     = d_vq

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        self.encoder  = GenieEncoder(C, d_vq, hw, cdep, key=k1)
        self.decoder  = GenieDecoder(d_vq, C, H, W, hw, cdep, key=k2)
        self.vq       = VectorQuantizer(frame_K, d_vq, beta=vq_beta, key=k3)
        self.idm      = GenieIDM(d_vq, action_K, action_d, act_beta, key=k4)

        d_model  = cfg["dyn_d_model"]
        n_heads  = cfg["dyn_num_heads"]
        n_layers = cfg["dyn_num_layers"]
        d_ff     = d_model * cfg["dyn_ffn_mult"]

        self.dynamics = GenieDynamicsTransformer(
            frame_K, action_K, d_model, n_heads, n_layers, d_ff,
            max_T=40, key=k5)
        self.gcm = GenieGCM(frame_K, action_K, cfg["gcm_hidden"], key=k6)

    # ── Convenience wrappers ─────────────────────────────────────

    def encode_frame(self, frame: jnp.ndarray):
        """
        frame : (H, W, C)
        → (z_idx: int,  z_q_st: (d_vq,),  vq_loss: scalar,  z_e: (d_vq,))
        """
        z_e = self.encoder(jnp.transpose(frame, (2, 0, 1)))  # channels-first
        z_idx, z_q_st, vq_loss = self.vq.quantize(z_e)
        return z_idx, z_q_st, vq_loss, z_e

    def decode_frame(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """z_q: (d_vq,) → frame: (H, W, C)"""
        return jnp.transpose(self.decoder(z_q), (1, 2, 0))

    def decode_from_index(self, z_idx: int) -> jnp.ndarray:
        """z_idx: int → frame: (H, W, C)"""
        return self.decode_frame(self.vq.decode(z_idx))

    def extract_action(self, z_q_t: jnp.ndarray, z_q_tp1: jnp.ndarray):
        """→ (action_idx, action_emb, vq_loss)"""
        return self.idm(z_q_t, z_q_tp1)