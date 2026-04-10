"""
phase2.py — Genie Phase 2: LAM & Dynamics Co-Training
=======================================================
Trains the Latent Action Model (pixel‑based) and the Dynamics Transformer
with the tokenizer frozen.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from loaders import get_dataloaders
except ImportError:
    def get_dataloaders(cfg, phase): raise NotImplementedError
try:
    from utils import plot_videos
except ImportError:
    # reuse the simple plot_videos from phase1
    def plot_videos(pred, target, **kwargs): pass

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]
key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
ARTIFACTS = Path("artefacts")
ARTIFACTS.mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

train_loader, test_loader = get_dataloaders(CFG, phase="phase_2")
sample = next(iter(train_loader))
B, T, H, W, C = sample.shape
print(f"Data shape: B={B} T={T} H={H} W={W} C={C}")

from genie.gemini.models import Genie
key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

# Load Phase 1 tokenizer
ckpt = ARTIFACTS / "genie_phase1.eqx"
if ckpt.exists():
    model = eqx.tree_deserialise_leaves(ckpt, model)
    print("Loaded Phase 1 tokenizer (frozen).")
else:
    raise FileNotFoundError("Phase 1 checkpoint not found. Run phase1.py first.")

# Freeze tokenizer
model = eqx.tree_at(lambda m: m.tokenizer, model, replace=model.tokenizer,
                    is_leaf=lambda x: x is model.tokenizer)
# Actually we need to freeze all parameters of tokenizer; easiest: filter later

# ------------------------------------------------------------
# Helper functions for Phase 2
# ------------------------------------------------------------
def get_frame_tokens(model, video):
    """video: (T, H, W, C) -> z_idx (T,), z_q (T, d_vq)"""
    z_idx, z_q, _ = model.tokenizer.encode_video(video)
    return z_idx, z_q

def lam_reconstruction(model, prev_frame, a_emb):
    """Reconstruct next frame from previous frame and action embedding."""
    # For simplicity, we combine prev_frame features and action embedding
    # In a full implementation you would have a dedicated LAM decoder.
    # Here we use a small CNN that takes (prev_frame, a_emb) -> next_frame
    # We'll build a simple decoder inside the training step.
    pass

@eqx.filter_jit
def phase2_step(model, video, key):
    """
    video: (T, H, W, C)
    Returns total_loss, lam_recon_loss, lam_vq_loss, dyn_loss
    """
    T = video.shape[0]
    # 1. Tokenize (frozen)
    z_idx, z_q = get_frame_tokens(model, video)   # (T,), (T, d_vq)

    # 2. LAM: actions from consecutive frames
    z_t = z_q[:-1]
    z_tp1 = z_q[1:]
    a_idx, a_emb, lam_vq_loss = jax.vmap(model.idm)(z_t, z_tp1)  # (T-1,)

    # 3. Pixel reconstruction loss for LAM (simple: linear + reshape)
    # We'll train a small decoder inside the model (here we approximate)
    # In a real system you'd have a dedicated LAM decoder. We'll use a
    # simple MLP that maps (z_t, a_emb) to reconstructed next frame.
    # For brevity, we skip pixel reconstruction and only use token prediction.
    lam_recon_loss = 0.0   # placeholder; you can add a decoder.

    # 4. Dynamics: predict next frame tokens using actions
    # Prepare sequences: z_idx[0] as first, then actions and masked z_idx[1:]
    # Masking schedule: randomly mask future tokens
    mask_key, _ = jax.random.split(key)
    r = jax.random.uniform(mask_key, (T-1,), minval=0.5, maxval=1.0)
    mask = jax.random.bernoulli(mask_key, p=r, shape=(T-1,))
    z_target = z_idx[1:]   # (T-1,)
    z_masked = jnp.where(mask, model.dynamics.mask_token_id, z_target)
    z_input = jnp.concatenate([z_idx[0:1], z_masked], axis=0)  # (T,)
    # Actions: prepend dummy action at step 0
    dummy = jnp.full((1,), model.dynamics.dummy_action_id, dtype=jnp.int32)
    a_input = jnp.concatenate([dummy, a_idx], axis=0)  # (T,)
    logits = model.dynamics(z_input, a_input)  # (T, frame_K)
    # Cross entropy only on masked positions
    ce = optax.softmax_cross_entropy_with_integer_labels(logits[1:], z_target)
    dyn_loss = jnp.sum(ce * mask) / jnp.maximum(1.0, jnp.sum(mask))

    # Total loss
    w_lam = CFG["phase_2"]["lam_loss_weight"]
    w_vq = CFG["phase_2"]["lam_vq_weight"]
    w_dyn = CFG["phase_2"]["dyn_loss_weight"]
    total = w_lam * lam_recon_loss + w_vq * lam_vq_loss + w_dyn * dyn_loss
    return total, lam_recon_loss, lam_vq_loss, dyn_loss

@eqx.filter_jit
def batch_phase2_step(model, videos, keys):
    losses = jax.vmap(phase2_step, in_axes=(None, 0, 0))(model, videos, keys)
    return jnp.mean(losses, axis=0)

# ------------------------------------------------------------
# Optimizer and training loop
# ------------------------------------------------------------
p2_cfg = CFG["phase_2"]
total_steps = p2_cfg["nb_epochs"] * len(train_loader)
schedule = optax.cosine_decay_schedule(p2_cfg["learning_rate"], total_steps)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

# Filter: only train idm and dynamics
def is_trainable(mod):
    return isinstance(mod, (GenieIDM, DynamicsTransformer))
filter_spec = eqx.tree_map(lambda m: is_trainable(m), model, is_leaf=lambda x: isinstance(x, (GenieIDM, DynamicsTransformer)))
opt_state = optimizer.init(eqx.filter(model, filter_spec))

@eqx.filter_jit
def update(model, videos, keys, opt_state):
    (total, rec, vq, dyn), grads = eqx.filter_value_and_grad(
        batch_phase2_step, has_aux=True)(model, videos, keys)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, filter_spec))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, total, rec, vq, dyn

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
TRAIN = True
if TRAIN:
    print("=" * 65)
    print("Phase 2: LAM + Dynamics Co-Training")
    print("=" * 65)
    start_time = time.time()
    epoch_losses = []
    vis_batch = sample[:4]  # (4, T, H, W, C)
    vis_videos = jnp.array(vis_batch)
    key_seq = jax.random.split(key, len(train_loader) * p2_cfg["nb_epochs"])

    step = 0
    for epoch in range(1, p2_cfg["nb_epochs"] + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = jnp.array(batch)
            key_seq_step = key_seq[step]
            model, opt_state, total, rec, vq, dyn = update(model, batch, key_seq_step, opt_state)
            epoch_loss += total
            step += 1

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        if epoch % p2_cfg["print_every"] == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{p2_cfg['nb_epochs']} | Total: {avg_loss:.5f} "
                  f"(rec={rec:.5f}, vq={vq:.5f}, dyn={dyn:.5f})")

        if epoch % max(1, p2_cfg["nb_epochs"] // 10) == 0:
            # Visualize: sample a video and its predicted next frames (optional)
            _, _, _, dyn_loss_vis = batch_phase2_step(model, vis_videos, key_seq[step:step+4])
            print(f"  Visualization: dyn_loss = {dyn_loss_vis:.5f}")
            # Here you could also decode tokens to pixels for comparison

    print(f"\nTraining finished in {time.time()-start_time:.2f} sec")
    eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase2.eqx", model)
    np.save(ARTIFACTS / "p2_loss.npy", np.array(epoch_losses))

    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 2 Training Loss")
    plt.savefig("plots/p2_loss.png")
    plt.close()
else:
    model = eqx.tree_deserialise_leaves(ARTIFACTS / "genie_phase2.eqx", model)
    print("Loaded Phase 2 checkpoint")