"""
models.py — Genie-Continuous Architecture for MovingMNIST
==========================================================

Implements the three-component Genie architecture (Bruce et al., ICML 2024)
adapted for *continuous* latent actions, eliminating the need for a VQ
codebook and enabling end-to-end differentiable training.

Components
----------
Encoder          : CNN  frame → latent z_t ∈ R^{d_z}
Decoder          : Transposed-CNN  z_t → frame  (pixel-space renderer)
IDM              : MLP  (z_t, z_{t+1}) → action u_t ∈ R^{d_u}
DynamicsTransformer : Causal Transformer  (z_{1:t}, u_{1:t}) → ẑ_{t+1}
GCM              : GRU/LSTM/Transformer  (z_t, u_{t-1}) → û_t (Phase 3)
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_causal_mask(T: int) -> jnp.ndarray:
    """Upper-triangular boolean mask: position i may only attend to j <= i."""
    return jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))


# ─────────────────────────────────────────────────────────────────
# 1. Encoder  (CNN)
# ─────────────────────────────────────────────────────────────────

class GenieEncoder(eqx.Module):
    """
    Strided-convolution encoder.
    Input : (C, H, W) frame  (channels-first for eqx.nn.Conv2d)
    Output: z ∈ R^{d_z}
    """
    conv_layers: list
    proj: eqx.nn.Linear

    def __init__(self, in_channels: int, d_z: int,
                 hidden_width: int, depth: int, key):
        keys = jax.random.split(key, depth + 1)
        convs = []
        ch_in = in_channels
        ch_out = hidden_width
        for i in range(depth):
            convs.append(eqx.nn.Conv2d(ch_in, ch_out,
                                        kernel_size=3, stride=2,
                                        padding=1, key=keys[i]))
            ch_in = ch_out
            ch_out = min(ch_out * 2, 512)
        self.conv_layers = convs
        # Compute flat dim with a dummy pass
        dummy = jnp.zeros((in_channels, 64, 64))
        for l in self.conv_layers:
            dummy = jax.nn.relu(l(dummy))
        flat = dummy.reshape(-1).shape[0]
        self.proj = eqx.nn.Linear(flat, d_z, key=keys[depth])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for l in self.conv_layers:
            x = jax.nn.relu(l(x))
        return self.proj(x.reshape(-1))


# ─────────────────────────────────────────────────────────────────
# 2. Decoder  (Transposed CNN)
# ─────────────────────────────────────────────────────────────────

class GenieDecoder(eqx.Module):
    """
    Transposed-convolution decoder.
    Input : z ∈ R^{d_z}
    Output: (C, H, W) frame in [0, 1]
    """
    proj: eqx.nn.Linear
    deconv_layers: list
    out_layer: eqx.nn.Conv2d
    spatial_shape: tuple = eqx.field(static=True)
    bottleneck_channels: int = eqx.field(static=True)

    def __init__(self, d_z: int, out_channels: int,
                 spatial_shape: Tuple[int, int],
                 hidden_width: int, depth: int, key):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 2)

        # Determine spatial size after depth stride-2 downsamples
        H_bot = H // (2 ** depth)
        W_bot = W // (2 ** depth)
        ch_bot = min(hidden_width * (2 ** (depth - 1)), 512)
        self.bottleneck_channels = ch_bot
        self.spatial_shape = (H_bot, W_bot)

        flat_dim = ch_bot * H_bot * W_bot
        self.proj = eqx.nn.Linear(d_z, flat_dim, key=keys[0])

        deconvs = []
        ch_in = ch_bot
        for i in range(depth):
            ch_out = ch_in // 2 if ch_in > hidden_width else hidden_width
            deconvs.append(eqx.nn.ConvTranspose2d(
                ch_in, ch_out, kernel_size=4, stride=2,
                padding=1, key=keys[i + 1]))
            ch_in = ch_out
        self.deconv_layers = deconvs
        self.out_layer = eqx.nn.Conv2d(
            ch_in, out_channels, kernel_size=3,
            stride=1, padding=1, key=keys[-1])

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        H_bot, W_bot = self.spatial_shape
        x = jax.nn.relu(self.proj(z))
        x = x.reshape(self.bottleneck_channels, H_bot, W_bot)
        for l in self.deconv_layers:
            x = jax.nn.relu(l(x))
        # return jax.nn.sigmoid(self.out_layer(x))   # output in [0, 1]
        return self.out_layer(x)   # output in [0, 1]


# ─────────────────────────────────────────────────────────────────
# 3. Inverse Dynamics Model  (IDM / LAM)
# ─────────────────────────────────────────────────────────────────

class GenieIDM(eqx.Module):
    """
    Latent Action Model (continuous).
    u_t = IDM(z_t, z_{t+1})
    """
    mlp: eqx.nn.MLP

    def __init__(self, d_z: int, d_u: int, key):
        self.mlp = eqx.nn.MLP(
            in_size=d_z * 2, out_size=d_u,
            width_size=d_z * 2, depth=3, key=key)

    def __call__(self, z_t: jnp.ndarray, z_tp1: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(jnp.concatenate([z_t, z_tp1], axis=-1))


# ─────────────────────────────────────────────────────────────────
# 4. Dynamics Transformer
# ─────────────────────────────────────────────────────────────────

class CausalMHABlock(eqx.Module):
    """Single causal multi-head attention + FFN block."""
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attn:  eqx.nn.MultiheadAttention
    ffn_in:  eqx.nn.Linear
    ffn_out: eqx.nn.Linear
    d_model: int = eqx.field(static=True)
    ffn_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.d_model = d_model
        self.ffn_dim = d_model * ffn_mult
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.attn  = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=d_model, key=k1)
        self.ffn_in  = eqx.nn.Linear(d_model, self.ffn_dim, key=k2)
        self.ffn_out = eqx.nn.Linear(self.ffn_dim, d_model, key=k3)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # x: (T, d_model)
        T = x.shape[0]
        # Self-attention with causal mask
        x_norm = jax.vmap(self.norm1)(x)
        # eqx MultiheadAttention expects (seq, d) inputs
        attn_out = self.attn(
            query=x_norm, key_=x_norm, value=x_norm,
            mask=mask)
        x = x + attn_out
        # FFN
        x_norm2 = jax.vmap(self.norm2)(x)
        ffn = jax.vmap(lambda v: self.ffn_out(
            jax.nn.gelu(self.ffn_in(v))))(x_norm2)
        return x + ffn


class GenieDynamicsTransformer(eqx.Module):
    """
    Causal Transformer dynamics model.

    Takes a sequence of (z_t, u_t) pairs and predicts z_{t+1} for each t.
    At inference time, predicts step-by-step autoregressively.

    Input tokens  : concat(z_t, u_t)  projected to R^{d_model}
    Output tokens : ẑ_{t+1} ∈ R^{d_z}
    """
    input_proj: eqx.nn.Linear
    pos_emb: jax.Array          # (max_T, d_model)
    blocks: list
    out_proj: eqx.nn.Linear
    d_model: int = eqx.field(static=True)
    d_z: int = eqx.field(static=True)
    d_u: int = eqx.field(static=True)
    max_T: int = eqx.field(static=True)

    def __init__(self, d_z: int, d_u: int,
                 d_model: int, num_heads: int,
                 num_layers: int, ffn_mult: int,
                 max_T: int, key):
        keys = jax.random.split(key, num_layers + 3)
        self.d_model = d_model
        self.d_z = d_z
        self.d_u = d_u
        self.max_T = max_T

        self.input_proj = eqx.nn.Linear(d_z + d_u, d_model, key=keys[0])
        # Learnable positional embeddings
        self.pos_emb = jax.random.normal(keys[1], (max_T, d_model)) * 0.02
        self.blocks = [
            CausalMHABlock(d_model, num_heads, ffn_mult, key=keys[2 + i])
            for i in range(num_layers)
        ]
        self.out_proj = eqx.nn.Linear(d_model, d_z, key=keys[-1])

    def __call__(self, z_seq: jnp.ndarray,
                 u_seq: jnp.ndarray) -> jnp.ndarray:
        """
        z_seq : (T, d_z)   — encoded frames
        u_seq : (T, d_u)   — latent actions (u_t explains z_t → z_{t+1})
        Returns ẑ_{2:T+1} : (T, d_z)  — predicted next frames
        """
        T = z_seq.shape[0]
        # Build input tokens
        tokens_in = jnp.concatenate([z_seq, u_seq], axis=-1)   # (T, d_z+d_u)
        tokens = jax.vmap(self.input_proj)(tokens_in)           # (T, d_model)
        tokens = tokens + self.pos_emb[:T]                      # add pos emb

        mask = _make_causal_mask(T)   # (T, T) causal mask
        for block in self.blocks:
            tokens = block(tokens, mask)

        return jax.vmap(self.out_proj)(tokens)                  # (T, d_z)


# ─────────────────────────────────────────────────────────────────
# 5. Generative Control Model  (GCM — Phase 3)
# ─────────────────────────────────────────────────────────────────

class GenieGCM(eqx.Module):
    """
    Autoregressive prior over actions for deployment (Phase 3).
    Predicts û_t from (z_t, u_{t-1}) using a GRU hidden state.

    For a Transformer variant, the full context (z_{1:t}, u_{1:t-1})
    is used via in-context learning.
    """
    rnn: eqx.Module           # GRUCell or LSTMCell
    action_head: eqx.nn.MLP
    d_hidden: int = eqx.field(static=True)
    d_z: int = eqx.field(static=True)
    d_u: int = eqx.field(static=True)
    rnn_type: str = eqx.field(static=True)

    def __init__(self, d_z: int, d_u: int,
                 d_hidden: int, rnn_type: str, key):
        k1, k2 = jax.random.split(key)
        self.d_z = d_z
        self.d_u = d_u
        self.d_hidden = d_hidden
        self.rnn_type = rnn_type.upper()

        input_dim = d_z + d_u
        if self.rnn_type == "LSTM":
            self.rnn = eqx.nn.LSTMCell(input_dim, d_hidden, key=k1)
        elif self.rnn_type == "GRU":
            self.rnn = eqx.nn.GRUCell(input_dim, d_hidden, key=k1)
        else:
            raise ValueError(f"Unsupported GCM rnn_type: {rnn_type}")

        self.action_head = eqx.nn.MLP(
            in_size=d_hidden + d_z,
            out_size=d_u,
            width_size=d_hidden,
            depth=2, key=k2)

    def initial_state(self) -> jnp.ndarray:
        if self.rnn_type == "LSTM":
            return (jnp.zeros(self.d_hidden), jnp.zeros(self.d_hidden))
        return jnp.zeros(self.d_hidden)

    def step(self, hidden, z_t: jnp.ndarray,
             u_prev: jnp.ndarray):
        """One GCM step: (hidden, z_t, u_{t-1}) → (hidden', û_t)."""
        rnn_in = jnp.concatenate([z_t, u_prev], axis=-1)
        hidden_new = self.rnn(rnn_in, hidden)
        # Extract the actual hidden vector for LSTM
        h = hidden_new[0] if self.rnn_type == "LSTM" else hidden_new
        u_pred = self.action_head(jnp.concatenate([h, z_t], axis=-1))
        return hidden_new, u_pred


# ─────────────────────────────────────────────────────────────────
# 6. Full Genie Model
# ─────────────────────────────────────────────────────────────────

class Genie(eqx.Module):
    """
    Genie-Continuous: full model wrapping all five components.

    Phase 1 : train encoder + decoder
    Phase 2 : train IDM + DynamicsTransformer  (encoder frozen)
    Phase 3 : train GCM  (encoder + IDM + Transformer frozen)
    """
    encoder:  GenieEncoder
    decoder:  GenieDecoder
    idm:      GenieIDM
    dynamics: GenieDynamicsTransformer
    gcm:      GenieGCM

    # Scalar metadata (static)
    d_z: int = eqx.field(static=True)
    d_u: int = eqx.field(static=True)

    def __init__(self, cfg: dict,
                 frame_shape: Tuple[int, int, int],
                 key):
        H, W, C = frame_shape
        d_z  = cfg["latent_dim"]
        d_u  = cfg["action_dim"]
        self.d_z = d_z
        self.d_u = d_u

        hw   = cfg["cnn_hidden_width"]
        cdep = cfg["cnn_depth"]

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.encoder = GenieEncoder(
            in_channels=C, d_z=d_z,
            hidden_width=hw, depth=cdep, key=k1)

        self.decoder = GenieDecoder(
            d_z=d_z, out_channels=C,
            spatial_shape=(H, W),
            hidden_width=hw, depth=cdep, key=k2)

        self.idm = GenieIDM(d_z=d_z, d_u=d_u, key=k3)

        max_T = 40   # maximum sequence length
        self.dynamics = GenieDynamicsTransformer(
            d_z=d_z, d_u=d_u,
            d_model=cfg["dyn_d_model"],
            num_heads=cfg["dyn_num_heads"],
            num_layers=cfg["dyn_num_layers"],
            ffn_mult=cfg["dyn_ffn_mult"],
            max_T=max_T, key=k4)

        self.gcm = GenieGCM(
            d_z=d_z, d_u=d_u,
            d_hidden=cfg["gcm_hidden"],
            rnn_type=cfg["gcm_type"], key=k5)

    # ── convenience ──

    def encode(self, frame: jnp.ndarray) -> jnp.ndarray:
        """frame: (H, W, C)  →  z: (d_z,)"""
        return self.encoder(jnp.transpose(frame, (2, 0, 1)))

    def decode(self, z: jnp.ndarray,
               shape: Tuple[int, int, int]) -> jnp.ndarray:
        """z: (d_z,)  →  frame: (H, W, C)"""
        out = self.decoder(z)           # (C, H, W)
        return jnp.transpose(out, (1, 2, 0))

    def extract_action(self, z_t: jnp.ndarray,
                        z_tp1: jnp.ndarray) -> jnp.ndarray:
        """IDM: u_t = IDM(z_t, z_{t+1})"""
        return self.idm(z_t, z_tp1)

    def predict_next_latent(self, z_seq: jnp.ndarray,
                             u_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Transformer dynamics: given z_{1:T} and u_{1:T},
        returns ẑ_{2:T+1} of shape (T, d_z).
        """
        return self.dynamics(z_seq, u_seq)

    def predict_action_gcm(self, gcm_hidden, z_t, u_prev):
        """One GCM step during autonomous rollout."""
        return self.gcm.step(gcm_hidden, z_t, u_prev)