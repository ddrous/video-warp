import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from jax.flatten_util import ravel_pytree

def fourier_encode(x, num_freqs):
    freqs = 2.0 ** jnp.arange(num_freqs)
    angles = x[..., None] * freqs[None, None, :] * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)

def get_activation(name):
    if name == 'sin': return jnp.sin
    if name == 'gelu': return jax.nn.gelu
    return jax.nn.relu

class RootMLP(eqx.Module):
    layers: list
    activation: callable = eqx.field(static=True)  # <-- Add this!

    def __init__(self, in_size, out_size, width, depth, activation_name, key):
        self.activation = get_activation(activation_name)
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

class WeightCNN(eqx.Module):
    layers: list
    theta_base: jax.Array

    def __init__(self, in_channels, out_dim, spatial_shape, theta_base, key, hidden_width=32, depth=4):
        self.theta_base = theta_base
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        conv_layers = []
        current_in = in_channels
        current_out = hidden_width
        
        for i in range(depth):
            conv_layers.append(
                eqx.nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1, key=keys[i])
            )
            current_in = current_out
            current_out *= 2
            
        dummy_x = jnp.zeros((in_channels, H, W))
        for layer in conv_layers:
            dummy_x = layer(dummy_x)

        flat_dim = dummy_x.reshape(-1).shape[0]
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = x.reshape(-1)
        offset = self.layers[-1](x)
        return offset

class ForwardDynamicsModule(eqx.Module):
    mlp_A: Optional[eqx.nn.MLP]
    mlp_B: Optional[eqx.nn.MLP]
    giant_mlp: Optional[eqx.nn.MLP]
    split_forward: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, lam_dim, split_forward, key):
        self.split_forward = split_forward
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Hardcoded FDM dimensions as requested
        fdm_depth = 3
        fdm_width = dyn_dim * 2
        
        if split_forward:
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=fdm_width, depth=fdm_depth, key=k1)
            self.mlp_B = eqx.nn.MLP(lam_dim, dyn_dim, width_size=fdm_width, depth=fdm_depth, key=k2)
            self.giant_mlp = None
        else:
            self.mlp_A = None
            self.mlp_B = None
            self.giant_mlp = eqx.nn.MLP(dyn_dim + lam_dim, dyn_dim, width_size=fdm_width, depth=fdm_depth, key=k3)

    def __call__(self, z_prev, a):
        if self.split_forward:
            z_a = self.mlp_A(z_prev)
            z_b = self.mlp_B(a)
            return (z_a, z_b), z_a + z_b
        else:
            out = self.giant_mlp(jnp.concatenate([z_prev, a], axis=-1))
            return None, out

class InverseDynamicsModule(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key):
        self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=3, key=key)
        
    def __call__(self, z_prev, z_target):
        return self.mlp(jnp.concatenate([z_prev, z_target], axis=-1))

class VanillaRNNCell(eqx.Module):
    weight_ih: eqx.nn.Linear
    weight_hh: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.weight_ih = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=k1)
        self.weight_hh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k2)

    def __call__(self, input: jax.Array, hidden: jax.Array) -> jax.Array:
        return jax.nn.tanh(self.weight_ih(input) + self.weight_hh(hidden))

class RNNController(eqx.Module):
    d_model: int
    rnn_type: str = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    
    rnn_cell: eqx.Module
    action_decoder: eqx.nn.MLP

    def __init__(self, lam_dim, mem_dim, latent_dim, out_dim, key, rnn_type="GRU", **kwargs):
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        self.rnn_type = rnn_type.upper()
        
        k1, k2 = jax.random.split(key, 2)
        
        input_dim = latent_dim + lam_dim
        if self.rnn_type == "LSTM":
            self.rnn_cell = eqx.nn.LSTMCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "GRU":
            self.rnn_cell = eqx.nn.GRUCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "RNN":
            self.rnn_cell = VanillaRNNCell(input_dim, self.d_model, key=k1)
        else:
            raise ValueError("Unsupported rnn_type. Must be 'LSTM', 'GRU', or 'RNN'.")

        decode_input_dim = self.d_model + latent_dim
        
        self.action_decoder = eqx.nn.MLP(
            in_size=decode_input_dim, out_size=out_dim, 
            width_size=self.d_model * 1, depth=1, key=k2
        )

    def reset(self, T):
        if self.rnn_type == "LSTM":
            return (jnp.zeros((self.d_model,)), jnp.zeros((self.d_model,)))
        else:
            return jnp.zeros((self.d_model,))

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

    def __init__(self, d_model, num_heads, key):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k1
        )
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model * 4, depth=1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask):
        x_norm = jax.vmap(self.ln1)(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x

class TransformerController(eqx.Module):
    """
    Autoregressive Transformer Module for Latent Actions (GCM).
    """
    d_model: int
    max_len: int
    pos_emb: jax.Array
    blocks: tuple
    proj_in: eqx.nn.Linear
    
    lam_dim: int = eqx.field(static=True)
    icl_decoding: bool = eqx.field(static=True)
    
    action_mlp: Optional[eqx.nn.MLP]
    output_proj: Optional[eqx.nn.Linear]

    def __init__(self, lam_dim, mem_dim, latent_dim, out_dim, key, max_len=20, num_heads=4, num_blocks=4):
        self.max_len = max_len
        self.icl_decoding = True
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        
        self.proj_in = eqx.nn.Linear(latent_dim + lam_dim, self.d_model, key=k1)
        self.pos_emb = jax.random.normal(k2, (max_len, self.d_model)) * 0.02
        
        block_keys = jax.random.split(k3, num_blocks)
        self.blocks = tuple(TransformerBlock(self.d_model, num_heads, bk) for bk in block_keys)

        if self.icl_decoding:
            self.action_mlp = None
            self.output_proj = eqx.nn.Linear(self.d_model, out_dim, key=k6)
        else:
            self.action_mlp = eqx.nn.MLP(self.d_model + latent_dim, out_dim, width_size=self.d_model * 2, depth=3, key=k4)
            self.output_proj = None

    def reset(self, T):
        return jnp.zeros((T, self.d_model))

    def encode(self, buffer, step_idx, z, a):
        token = self.proj_in(jnp.concatenate([z, a], axis=-1))
        return buffer.at[step_idx - 1].set(token)

    def decode(self, buffer, step_idx, z_current):
        T = buffer.shape[0]
        if self.icl_decoding:
            zero_action = jnp.zeros((self.lam_dim,), dtype=z_current.dtype)
            query_token = self.proj_in(jnp.concatenate([z_current, zero_action], axis=-1))
            temp_buffer = buffer.at[step_idx - 1].set(query_token)
            
            x = temp_buffer + self.pos_emb[:T]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            
            for block in self.blocks:
                x = block(x, mask)
            context = x[step_idx - 1]
            return self.output_proj(context)
        else:
            def compute_context():
                x = buffer + self.pos_emb[:T]
                mask = jnp.tril(jnp.ones((T, T), dtype=bool))
                for block in self.blocks:
                    x = block(x, mask)
                return x[step_idx - 2]
                
            context = jax.lax.cond(step_idx > 1, compute_context, lambda: jnp.zeros(self.d_model))
            return self.action_mlp(jnp.concatenate([context, z_current], axis=-1))


class GenerativeControlModule(eqx.Module):
    d_model: int
    gcm_type: str = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)

    seq_model: eqx.Module

    def __init__(self, lam_dim, mem_dim, latent_dim, out_dim, key, gcm_type="GRU", **kwargs):
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        self.gcm_type = gcm_type.upper()

        if self.gcm_type.lower() in ["lstm", "gru", "rnn"]:
            self.seq_model = RNNController(lam_dim, mem_dim, latent_dim, out_dim, key=key, rnn_type=gcm_type)
        elif self.gcm_type.lower() == "transformer":
            self.seq_model = TransformerController(lam_dim, mem_dim, latent_dim, out_dim, key=key, **kwargs)
        else:
            raise ValueError("Unsupported gcm_type. Must be 'LSTM', 'GRU', 'RNN', or 'TRANSFORMER'.")

    def decode_gcm(self, buffer, step_idx, z_current):
        return self.seq_model.decode(buffer, step_idx, z_current)
    def encode_gcm(self, buffer, step_idx, z_current, a):
        return self.seq_model.encode(buffer, step_idx, z_current, a)
    def reset_gcm(self, T):
        return self.seq_model.reset(T)


class LatentActionModule(eqx.Module):
    """ Latent Action Model holding both IDM and GCM. """
    idm: InverseDynamicsModule
    gcm: Optional[eqx.Module]
    discrete_actions: bool = eqx.field(static=True)

    idm_embeddings: Optional[eqx.nn.Embedding]
    gcm_embeddings: Optional[eqx.nn.Embedding]

    action_bridge: Optional[eqx.nn.MLP]
    translate_actions: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, lam_dim, mem_dim, num_actions, init_gcm, gcm_type, key):
        k1, k2 = jax.random.split(key)
        self.discrete_actions = num_actions is not None
        self.idm = InverseDynamicsModule(dyn_dim, lam_dim, key=k1)

        if type(num_actions) == tuple and len(num_actions) == 2: 
            num_actions_idm, num_actions_gcm = num_actions
        else:
            num_actions_idm = num_actions_gcm = num_actions

        if self.discrete_actions:
            emb_weights = jnp.zeros((num_actions_idm, lam_dim))
            self.idm_embeddings = eqx.nn.Embedding(weight=emb_weights, key=k2)
        else:
            self.idm_embeddings = None

        self.translate_actions = num_actions_gcm != num_actions_idm
        if init_gcm:
            # gcm_input_dim = lam_dim*num_actions_idm if self.translate_actions else lam_dim
            # gcm_lam_dim = 1*num_actions_idm if self.translate_actions else lam_dim
            gcm_lam_dim = lam_dim
            # self.gcm = GenerativeControlModule(lam_dim, mem_dim, dyn_dim, gcm_lam_dim, key=k2, rnn_type="GRU")
            self.gcm = GenerativeControlModule(lam_dim, mem_dim, dyn_dim, gcm_lam_dim, key=k2, gcm_type=gcm_type, max_len=32, num_heads=4, num_blocks=4)
        else:
            self.gcm = None

        if self.gcm and self.discrete_actions:
            emb_weights_gcm = jnp.zeros((num_actions_gcm, gcm_lam_dim))
            if self.translate_actions:
                self.gcm_embeddings = eqx.nn.Embedding(weight=emb_weights_gcm, key=k2)
                self.action_bridge = eqx.nn.MLP(gcm_lam_dim+dyn_dim, lam_dim, width_size=dyn_dim, depth=2, key=k2)
            else:
                self.gcm_embeddings = None
                self.action_bridge = None
        else:
            self.gcm_embeddings = None
            self.action_bridge = None


    def quantise_idm_action(self, raw_action):
        dists = jnp.sum((raw_action - self.idm_embeddings.weight) ** 2, axis=-1)
        closest_idx = jnp.argmin(dists)
        return raw_action, self.idm_embeddings(closest_idx)

    def quantise_gcm_action(self, raw_action):
        if self.translate_actions:
            dists = jnp.sum((raw_action - self.gcm_embeddings.weight) ** 2, axis=-1)
            closest_idx = jnp.argmin(dists)
            quant_action = self.gcm_embeddings(closest_idx)
        else:
            dists = jnp.sum((raw_action - self.idm_embeddings.weight) ** 2, axis=-1)
            closest_idx = jnp.argmin(dists)
            quant_action = self.idm_embeddings(closest_idx)         ## Use IDM emneddings directly
        return raw_action, quant_action

    def decode_idm(self, z_prev, z_target):
        raw_action = self.idm(z_prev, z_target)
        if not self.discrete_actions:
            return raw_action, raw_action
        return self.quantise_idm_action(raw_action)

    def decode_gcm(self, buffer, step_idx, z_current):
        raw_action = self.gcm.decode_gcm(buffer, step_idx, z_current)
        if not self.discrete_actions:
            return raw_action, raw_action
        return self.quantise_gcm_action(raw_action)

    def encode_gcm(self, buffer, step_idx, z_current, a):
        return self.gcm.encode_gcm(buffer, step_idx, z_current, a)
    
    def reset_gcm(self, T):
        return self.gcm.reset_gcm(T)

class VWARP(eqx.Module):
    encoder: WeightCNN
    transition_model: ForwardDynamicsModule
    action_model: LatentActionModule

    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    split_forward: bool = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    mem_dim: int = eqx.field(static=True)
    use_action_residuals: bool = eqx.field(static=True)

    use_time_in_root: bool = eqx.field(static=True)

    def __init__(self, config, frame_shape, key, init_gcm=True):
        k_root, k_enc, k_lam, k_fwd = jax.random.split(key, 4)
        
        self.frame_shape = frame_shape
        self.num_freqs = config["num_fourier_freqs"]
        self.lam_dim = config["lam_space"]
        self.split_forward = config["split_forward"]
        self.mem_dim = config["mem_space"]
        self.use_action_residuals = config.get("use_action_residuals", False)
        self.use_time_in_root = config.get("use_time_in_root", False)

        H, W, C = frame_shape
        coord_dim = 2 + 2 * 2 * self.num_freqs 
        add_time = 1 if config.get("use_time_in_root", False) else 0
        
        activation_name = config.get("root_activation", "relu")
        template_root = RootMLP(coord_dim+add_time, C, config["root_width"], config["root_depth"], activation_name, k_root)
        
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]

        self.encoder = WeightCNN(
            in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), 
            theta_base=flat_params, key=k_enc, 
            hidden_width=config["cnn_hidden_width"], depth=config["cnn_depth"]
        )

        self.transition_model = ForwardDynamicsModule(self.d_theta, self.lam_dim, self.split_forward, key=k_fwd)

        num_actions_idm = config["phase_2"]["num_actions"]
        num_actions_gcm = config["phase_3"]["num_actions"]
        num_actions = (num_actions_idm, num_actions_gcm) if config["discrete_actions"] else None
        self.action_model = LatentActionModule(
            self.d_theta, self.lam_dim, self.mem_dim, 
            num_actions=num_actions, init_gcm=init_gcm, key=k_lam, gcm_type=config["phase_3"]["gcm_type"]
        )

    def render_pixels(self, theta, coords):
        def render_pt(th, coord):
            root = self.unravel_fn(th)
            encoded_spatial = fourier_encode(coord[1:], self.num_freqs)
            
            if self.use_time_in_root:
                encoded_coord = jnp.concatenate([coord[:1], encoded_spatial], axis=-1)
            else:
                encoded_coord = encoded_spatial
                
            return root(encoded_coord)
        return jax.vmap(render_pt, in_axes=(None, 0))(theta, coords)

    def render_frame(self, theta_offset, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 3)
        theta = theta_offset + self.encoder.theta_base

        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)

    def inference_rollout(self, ref_video, coords_grid, context_ratio=0.0):
        T = ref_video.shape[0]
        init_frame = ref_video[0]
        
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        m_init = self.action_model.reset_gcm(T)
        z_A_init = self.transition_model.mlp_A(z_init) if self.split_forward else None

        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t, z_tA = carry
            o_tp1, step_idx = scan_inputs

            time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            pred_out = self.render_frame(z_t, coords_grid_t)

            is_context = (step_idx / T) <= context_ratio

            def true_fn():
                return self.action_model.decode_idm(z_t, self.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
            def false_fn():
                raw_a, quant_a = self.action_model.decode_gcm(m_t, step_idx, z_t)
                if self.action_model.translate_actions:
                    raw_a = self.action_model.action_bridge(jnp.concatenate([raw_a, z_t], axis=-1))
                    quant_a = self.action_model.action_bridge(jnp.concatenate([quant_a, z_t], axis=-1))
                return raw_a, quant_a

            a_t_raw, a_t = jax.lax.cond(
                is_context,
                lambda: true_fn(),
                lambda: false_fn()
            )

            m_tp1 = self.action_model.encode_gcm(m_t, step_idx, z_t, a_t)
            (z_tp1A, _), z_tp1 = self.transition_model(z_t, a_t)

            return (z_tp1, m_tp1, z_tp1A), ((a_t_raw, a_t), z_t, pred_out)

        scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
        _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, z_A_init), scan_inputs)

        return actions, pred_latents, pred_video

    def __call__(self, ref_videos, coords_grid, context_ratio=0.0):
        batched_fn = jax.vmap(self.inference_rollout, in_axes=(0, None, None))
        return batched_fn(ref_videos, coords_grid, context_ratio)
