"""
models.py — Genie (Discrete) for MovingMNIST
=============================================

Implements the full Genie architecture:
  - VectorQuantizer (STE)
  - CNN Encoder / Decoder (for frames)
  - Tokenizer (VQ-VAE for videos)
  - Latent Action Model (IDM) with pixel reconstruction
  - Dynamics Transformer (causal, predicts next frame tokens)
  - Generative Control Module (RNN / Transformer based)
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional, Union

# ------------------------------------------------------------
# 1. Vector Quantizer (straight-through estimator)
# ------------------------------------------------------------
class VectorQuantizer(eqx.Module):
    codebook: jax.Array
    K: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, K: int, d: int, beta: float = 0.25, *, key):
        self.K = K
        self.d = d
        self.beta = beta
        self.codebook = jax.random.uniform(key, (K, d), minval=-1/math.sqrt(d),
                                           maxval=1/math.sqrt(d))

    def quantize(self, z_e: jnp.ndarray):
        flat = z_e.reshape(-1, self.d)
        z_sq = jnp.sum(flat ** 2, axis=-1, keepdims=True)
        c_sq = jnp.sum(self.codebook ** 2, axis=-1, keepdims=True).T
        dists = z_sq - 2.0 * flat @ self.codebook.T + c_sq
        idx_flat = jnp.argmin(dists, axis=-1)
        e_flat = self.codebook[idx_flat]
        z_q_flat = flat + jax.lax.stop_gradient(e_flat - flat)
        codebook_loss = jnp.mean((e_flat - jax.lax.stop_gradient(flat)) ** 2)
        commitment_loss = jnp.mean((jax.lax.stop_gradient(e_flat) - flat) ** 2)
        vq_loss = codebook_loss + self.beta * commitment_loss
        indices = idx_flat.reshape(z_e.shape[:-1])
        z_q = z_q_flat.reshape(z_e.shape)
        return indices, z_q, vq_loss

    def decode(self, indices: jnp.ndarray) -> jnp.ndarray:
        return self.codebook[indices]

# ------------------------------------------------------------
# 2. Frame Encoder / Decoder (CNN)
# ------------------------------------------------------------
class FrameEncoder(eqx.Module):
    conv_layers: list
    proj: eqx.nn.Linear

    def __init__(self, in_channels: int, d_vq: int, hidden_width: int, depth: int, *, key):
        keys = jax.random.split(key, depth + 1)
        convs = []
        ch_in, ch_out = in_channels, hidden_width
        for i in range(depth):
            convs.append(eqx.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, key=keys[i]))
            ch_in, ch_out = ch_out, min(ch_out * 2, 512)
        self.conv_layers = convs
        dummy = jnp.zeros((in_channels, 64, 64))
        for l in self.conv_layers:
            dummy = jax.nn.relu(l(dummy))
        flat_dim = dummy.reshape(-1).shape[0]
        self.proj = eqx.nn.Linear(flat_dim, d_vq, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for l in self.conv_layers:
            x = jax.nn.relu(l(x))
        return self.proj(x.reshape(-1))

class FrameDecoder(eqx.Module):
    proj: eqx.nn.Linear
    deconv_layers: list
    out_conv: eqx.nn.Conv2d
    bot_C: int = eqx.field(static=True)
    bot_HW: Tuple[int, int] = eqx.field(static=True)

    def __init__(self, d_vq: int, out_channels: int, H: int, W: int,
                 hidden_width: int, depth: int, *, key):
        keys = jax.random.split(key, depth + 2)
        H_bot, W_bot = H // (2 ** depth), W // (2 ** depth)
        C_bot = min(hidden_width * (2 ** (depth - 1)), 512)
        self.bot_C, self.bot_HW = C_bot, (H_bot, W_bot)
        self.proj = eqx.nn.Linear(d_vq, C_bot * H_bot * W_bot, key=keys[0])
        deconvs = []
        ch = C_bot
        for i in range(depth):
            ch_out = max(ch // 2, hidden_width)
            deconvs.append(eqx.nn.ConvTranspose2d(ch, ch_out, kernel_size=4, stride=2, padding=1, key=keys[i+1]))
            ch = ch_out
        self.deconv_layers = deconvs
        self.out_conv = eqx.nn.Conv2d(ch, out_channels, kernel_size=3, padding=1, key=keys[-1])

    def __call__(self, z_q: jnp.ndarray) -> jnp.ndarray:
        H_bot, W_bot = self.bot_HW
        x = jax.nn.relu(self.proj(z_q)).reshape(self.bot_C, H_bot, W_bot)
        for l in self.deconv_layers:
            x = jax.nn.relu(l(x))
        return self.out_conv(x)   # no sigmoid, use MSE loss

# ------------------------------------------------------------
# 3. Video Tokenizer (combines encoder, VQ, decoder)
# ------------------------------------------------------------
class CausalTemporalBlock(eqx.Module):
    """A standard causal transformer block over the sequence length T."""
    blocks: list
    pos_emb: jax.Array

    def __init__(self, d_model: int, n_heads: int, n_layers: int, max_T: int, *, key):
        keys = jax.random.split(key, n_layers + 1)
        self.pos_emb = jax.random.normal(keys[0], (max_T, d_model)) * 0.02
        # Re-using the CausalBlock already defined in your models.py
        self.blocks = [CausalBlock(d_model, n_heads, d_model * 4, key=keys[i+1]) 
                       for i in range(n_layers)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is (T, d_model)
        T = x.shape[0]
        x = x + self.pos_emb[:T]
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        for blk in self.blocks:
            x = blk(x, mask)
        return x

class VideoTokenizer(eqx.Module):
    encoder: FrameEncoder
    temporal_encoder: CausalTemporalBlock  # NEW: Temporal attention
    vq: VectorQuantizer
    temporal_decoder: CausalTemporalBlock  # NEW: Temporal attention
    decoder: FrameDecoder

    def __init__(self, frame_shape: Tuple[int, int, int], d_vq: int, K_frame: int,
                 vq_beta: float, hidden_width: int, depth: int, max_T: int = 40, *, key):
        H, W, C = frame_shape
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        
        # 1. Spatial CNN
        self.encoder = FrameEncoder(C, d_vq, hidden_width, depth, key=k1)
        # 2. Causal temporal mixing BEFORE quantization
        self.temporal_encoder = CausalTemporalBlock(d_vq, n_heads=4, n_layers=2, max_T=max_T, key=k2)
        # 3. Vector Quantizer
        self.vq = VectorQuantizer(K_frame, d_vq, beta=vq_beta, key=k3)
        # 4. Causal temporal mixing AFTER quantization
        self.temporal_decoder = CausalTemporalBlock(d_vq, n_heads=4, n_layers=2, max_T=max_T, key=k4)
        # 5. Spatial De-CNN
        self.decoder = FrameDecoder(d_vq, C, H, W, hidden_width, depth, key=k5)

    def encode(self, video: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """video: (T,H,W,C) -> (indices (T,), z_q (T,d_vq), vq_loss)"""
        # Encode frames spatially, independently
        z_spatial = jax.vmap(self.encoder)(jnp.transpose(video, (0, 3, 1, 2)))
        
        # Mix temporally with causal attention!
        # Now z_e[t] contains information from z_spatial[0]...z_spatial[t]
        z_e = self.temporal_encoder(z_spatial)
        
        # Quantize the temporally-aware embeddings
        indices, z_qs, vq_loss = self.vq.quantize(z_e)
        return indices, z_qs, vq_loss

    def decode(self, z_qs: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
        # Mix temporally before decoding back to pixels
        z_temporal = self.temporal_decoder(z_qs)
        
        # Decode frames spatially, independently
        frames = jax.vmap(self.decoder)(z_temporal)
        return jnp.transpose(frames, (0, 2, 3, 1))

# ------------------------------------------------------------
# 4. Latent Action Model (IDM) with pixel reconstruction
# ------------------------------------------------------------
class LatentActionModel(eqx.Module):
    idm: eqx.Module          # GenieIDM from earlier
    decoder: FrameDecoder    # shares decoder with tokenizer? no, separate for pixel reconstruction

    def __init__(self, d_vq: int, action_K: int, action_d: int, action_beta: float,
                 frame_shape: Tuple[int, int, int], hidden_width: int, depth: int, *, key):
        H, W, C = frame_shape
        k1, k2 = jax.random.split(key)
        # IDM: maps (z_t, z_{t+1}) -> action token
        self.idm = GenieIDM(d_vq, action_K, action_d, action_beta, key=k1)
        # Separate decoder to reconstruct next frame from (z_t, action)
        self.decoder = FrameDecoder(action_d + d_vq, C, H, W, hidden_width, depth, key=k2)

    def encode(self, frame_pairs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """frame_pairs: (T-1, 2*C, H, W) -> (a_idx, a_emb, vq_loss)"""
        # This would require the tokenizer to get z_t, z_{t+1}; here we assume outside
        # For simplicity in phase2, we implement a method that takes (z_t, z_{t+1})
        pass

    def decode(self, prev_frame: jnp.ndarray, a_emb: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
        """prev_frame: (C,H,W)  a_emb: (action_d,) -> (C,H,W)"""
        z_prev = self.tokenizer.encoder(jnp.transpose(prev_frame, (2,0,1)))   # need tokenizer ref
        # Actually we need the full model to pass the tokenizer; better to restructure.
        # We'll handle this inside phase2 directly.

# To avoid circular dependencies, we keep the IDM and LAM simple.
class GenieIDM(eqx.Module):
    mlp: eqx.nn.MLP
    action_vq: VectorQuantizer

    def __init__(self, d_vq: int, action_K: int, action_d: int, beta: float, *, key):
        k1, k2 = jax.random.split(key)
        self.mlp = eqx.nn.MLP(d_vq * 2, action_d, width_size=d_vq*2, depth=3, key=k1)
        self.action_vq = VectorQuantizer(action_K, action_d, beta=beta, key=k2)

    def __call__(self, z_t: jnp.ndarray, z_tp1: jnp.ndarray):
        a_cont = self.mlp(jnp.concatenate([z_t, z_tp1]))
        a_idx, a_emb, vq_loss = self.action_vq.quantize(a_cont)
        return a_idx, a_emb, vq_loss

# ------------------------------------------------------------
# 5. Causal Dynamics Transformer
# ------------------------------------------------------------
class CausalBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=k1)
        self.ff1 = eqx.nn.Linear(d_model, d_ff, key=k2)
        self.ff2 = eqx.nn.Linear(d_ff, d_model, key=k3)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        xn = jax.vmap(self.norm1)(x)
        x = x + self.attn(xn, xn, xn, mask=mask)
        xn = jax.vmap(self.norm2)(x)
        x = x + jax.vmap(lambda v: self.ff2(jax.nn.gelu(self.ff1(v))))(xn)
        return x

class DynamicsTransformer(eqx.Module):
    frame_emb: eqx.nn.Embedding
    action_emb: eqx.nn.Embedding
    input_proj: eqx.nn.Linear
    pos_emb: jax.Array
    blocks: list
    out_head: eqx.nn.Linear
    mask_token_id: int = eqx.field(static=True)
    dummy_action_id: int = eqx.field(static=True)
    frame_K: int = eqx.field(static=True)
    action_K: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(self, frame_K: int, action_K: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_T: int, mask_token_id: int,
                 dummy_action_id: int, *, key):
        keys = jax.random.split(key, n_layers + 5)
        self.frame_K = frame_K
        self.action_K = action_K
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.dummy_action_id = dummy_action_id
        self.frame_emb = eqx.nn.Embedding(frame_K, d_model, key=keys[0])
        self.action_emb = eqx.nn.Embedding(action_K, d_model, key=keys[1])
        self.input_proj = eqx.nn.Linear(2 * d_model, d_model, key=keys[2])
        self.pos_emb = jax.random.normal(keys[3], (max_T, d_model)) * 0.02
        self.blocks = [CausalBlock(d_model, n_heads, d_ff, key=keys[4+i]) for i in range(n_layers)]
        self.out_head = eqx.nn.Linear(d_model, frame_K, key=keys[-1])

    def __call__(self, z_idx: jnp.ndarray, a_idx: jnp.ndarray) -> jnp.ndarray:
        T = z_idx.shape[0]
        z_e = jax.vmap(self.frame_emb)(z_idx)
        a_e = jax.vmap(self.action_emb)(a_idx)
        tok = jax.vmap(self.input_proj)(jnp.concatenate([z_e, a_e], axis=-1))
        tok = tok + self.pos_emb[:T]
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        for blk in self.blocks:
            tok = blk(tok, mask)
        return jax.vmap(self.out_head)(tok)

# ------------------------------------------------------------
# 6. Generative Control Module (GCM) – RNN / Transformer
# ------------------------------------------------------------
class RNNCell(eqx.Module):
    weight_ih: eqx.nn.Linear
    weight_hh: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, key):
        k1, k2 = jax.random.split(key)
        self.weight_ih = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=k1)
        self.weight_hh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k2)

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.tanh(self.weight_ih(x) + self.weight_hh(h))

class RNNController(eqx.Module):
    rnn_cell: eqx.Module
    action_decoder: eqx.nn.MLP
    d_model: int = eqx.field(static=True)
    rnn_type: str = eqx.field(static=True)

    def __init__(self, lam_dim: int, mem_dim: int, latent_dim: int, out_dim: int,
                 rnn_type: str, key):
        self.d_model = mem_dim
        self.rnn_type = rnn_type.upper()
        k1, k2 = jax.random.split(key, 2)
        input_dim = latent_dim + lam_dim
        if self.rnn_type == "LSTM":
            self.rnn_cell = eqx.nn.LSTMCell(input_dim, mem_dim, key=k1)
        elif self.rnn_type == "GRU":
            self.rnn_cell = eqx.nn.GRUCell(input_dim, mem_dim, key=k1)
        else:
            self.rnn_cell = RNNCell(input_dim, mem_dim, key=k1)
        decode_input_dim = mem_dim + latent_dim
        self.action_decoder = eqx.nn.MLP(decode_input_dim, out_dim, width_size=mem_dim, depth=2, key=k2)

    def reset(self, T: int):
        if self.rnn_type == "LSTM":
            return (jnp.zeros(self.d_model), jnp.zeros(self.d_model))
        else:
            return jnp.zeros(self.d_model)

    def encode(self, state, step_idx, z, a):
        rnn_input = jnp.concatenate([z, a], axis=-1)
        return self.rnn_cell(rnn_input, state)

    def decode(self, state, step_idx, z_current):
        h = state[0] if self.rnn_type == "LSTM" else state
        decode_input = jnp.concatenate([h, z_current], axis=-1)
        return self.action_decoder(decode_input)

class TransformerBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, key):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(n_heads, d_model, key=k1)
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model*4, depth=1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        x_norm = jax.vmap(self.ln1)(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x

class TransformerController(eqx.Module):
    proj_in: eqx.nn.Linear
    pos_emb: jax.Array
    blocks: tuple
    output_proj: eqx.nn.Linear
    d_model: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)

    def __init__(self, lam_dim: int, mem_dim: int, latent_dim: int, out_dim: int,
                 max_len: int, num_heads: int, num_layers: int, key):
        self.d_model = mem_dim
        self.max_len = max_len
        k1, k2, k3 = jax.random.split(key, 3)
        self.proj_in = eqx.nn.Linear(latent_dim + lam_dim, mem_dim, key=k1)
        self.pos_emb = jax.random.normal(k2, (max_len, mem_dim)) * 0.02
        keys = jax.random.split(k3, num_layers)
        self.blocks = tuple(TransformerBlock(mem_dim, num_heads, k) for k in keys)
        self.output_proj = eqx.nn.Linear(mem_dim, out_dim, key=k3)

    def reset(self, T: int) -> jnp.ndarray:
        return jnp.zeros((T, self.d_model))

    def encode(self, buffer: jnp.ndarray, step_idx: int, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        token = self.proj_in(jnp.concatenate([z, a], axis=-1))
        return buffer.at[step_idx - 1].set(token)

    def decode(self, buffer: jnp.ndarray, step_idx: int, z_current: jnp.ndarray) -> jnp.ndarray:
        T = buffer.shape[0]
        zero_action = jnp.zeros(self.proj_in.in_features - z_current.shape[0])
        query_token = self.proj_in(jnp.concatenate([z_current, zero_action], axis=-1))
        temp_buffer = buffer.at[step_idx - 1].set(query_token)
        x = temp_buffer + self.pos_emb[:T]
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        for block in self.blocks:
            x = block(x, mask)
        return self.output_proj(x[step_idx - 1])

class GenerativeControlModule(eqx.Module):
    seq_model: eqx.Module
    gcm_type: str = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)

    def __init__(self, lam_dim: int, mem_dim: int, latent_dim: int, out_dim: int,
                 gcm_type: str, max_len: int = 40, num_heads: int = 4,
                 num_layers: int = 3, *, key):
        self.lam_dim = lam_dim
        self.gcm_type = gcm_type.upper()
        if self.gcm_type in ["GRU", "LSTM", "RNN"]:
            self.seq_model = RNNController(lam_dim, mem_dim, latent_dim, out_dim,
                                           rnn_type=gcm_type, key=key)
        elif self.gcm_type == "TRANSFORMER":
            self.seq_model = TransformerController(lam_dim, mem_dim, latent_dim, out_dim,
                                                   max_len, num_heads, num_layers, key=key)
        else:
            raise ValueError("gcm_type must be GRU, LSTM, RNN, or TRANSFORMER")

    def reset(self, T: int):
        return self.seq_model.reset(T)

    def encode(self, buffer, step_idx, z, a):
        return self.seq_model.encode(buffer, step_idx, z, a)

    def decode(self, buffer, step_idx, z_current):
        return self.seq_model.decode(buffer, step_idx, z_current)

# ------------------------------------------------------------
# 7. Full Genie Model
# ------------------------------------------------------------
class Genie(eqx.Module):
    tokenizer: VideoTokenizer
    idm: GenieIDM
    dynamics: DynamicsTransformer
    gcm: GenerativeControlModule

    def __init__(self, cfg: dict, frame_shape: Tuple[int, int, int], *, key):
        H, W, C = frame_shape
        d_vq = cfg["vq_dim"]
        frame_K = cfg["frame_codebook_size"]
        action_K = cfg["action_codebook_size"]
        vq_beta = cfg["vq_beta"]
        action_beta = cfg["action_beta"]
        hidden_width = cfg["cnn_hidden_width"]
        depth = cfg["cnn_depth"]
        action_d = cfg["action_dim"]
        keys = jax.random.split(key, 6)

        self.tokenizer = VideoTokenizer(frame_shape, d_vq, frame_K, vq_beta,
                                        hidden_width, depth, key=keys[0])
        self.idm = GenieIDM(d_vq, action_K, action_d, action_beta, key=keys[1])

        d_model = cfg["dyn_d_model"]
        n_heads = cfg["dyn_num_heads"]
        n_layers = cfg["dyn_num_layers"]
        d_ff = d_model * cfg["dyn_ffn_mult"]
        max_T = cfg.get("max_seq_len", 40)
        mask_token_id = frame_K  # use K as mask id
        dummy_action_id = action_K
        self.dynamics = DynamicsTransformer(frame_K, action_K, d_model, n_heads,
                                            n_layers, d_ff, max_T, mask_token_id,
                                            dummy_action_id, key=keys[2])

        mem_dim = cfg["gcm_mem_dim"]
        gcm_type = cfg["gcm_type"]
        self.gcm = GenerativeControlModule(action_d, mem_dim, d_vq, action_K,
                                           gcm_type, max_T, cfg["gcm_num_heads"],
                                           cfg["gcm_num_layers"], key=keys[3])
