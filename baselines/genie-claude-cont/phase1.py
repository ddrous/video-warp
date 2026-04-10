#%%
"""
phase1.py — Genie Phase 1: Tokenizer Pre-training
===================================================

Trains the Encoder and Decoder jointly to minimise pixel-space
reconstruction loss (MSE + optional SSIM) on individual frames.
After convergence, every frame o_t can be faithfully represented
by a compact latent code z_t = Encoder(o_t), which provides a
stable foundation for Phase 2 dynamics learning.

Usage (from the genie/ directory):
    python phase1.py
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

from loaders import get_dataloaders
from utils import plot_videos, count_trainable_params
from genie.gemini.models import Genie

# ─────────────────────────────────────────────────────────────────
# Config & Seeding
# ─────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED   = CFG["seed"]
key    = jax.random.PRNGKey(SEED)
np.random.seed(SEED)

ARTIFACTS = Path("artefacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
(Path("plots")).mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
train_loader, test_loader = get_dataloaders(CFG, phase="phase_1")
sample_batch = next(iter(train_loader))
if sample_batch.ndim == 5:          # (B, T, H, W, C) — pick frames
    B, T, H, W, C = sample_batch.shape
    print(f"Phase-1 frame batch shape: ({B*T}, {H}, {W}, {C})")
else:                               # (B, H, W, C) — already frames
    B, H, W, C = sample_batch.shape
    print(f"Phase-1 frame batch shape: {sample_batch.shape}")

#%%
# ─────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────
key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

enc_params = count_trainable_params(model.encoder)
dec_params = count_trainable_params(model.decoder)
print(f"\nEncoder params : {enc_params:,}")
print(f"Decoder params : {dec_params:,}")
print(f"Total Genie params (all) : {count_trainable_params(model):,}\n")

# ─────────────────────────────────────────────────────────────────
# Phase 1 Forward  (per-frame reconstruction)
# ─────────────────────────────────────────────────────────────────
@eqx.filter_jit
def p1_forward_single(model, frame):
    """Reconstruct a single frame. frame: (H, W, C)."""
    z = model.encode(frame)                     # (d_z,)
    recon = model.decode(z, frame.shape)        # (H, W, C)
    loss  = jnp.mean((recon - frame) ** 2)
    return loss, recon

@eqx.filter_jit
def p1_loss_batch(model, frames):
    """frames: (B, H, W, C)."""
    losses, _ = jax.vmap(p1_forward_single, in_axes=(None, 0))(model, frames)
    return jnp.mean(losses)

#%%
# ─────────────────────────────────────────────────────────────────
# Optimiser — only encoder + decoder trainable
# ─────────────────────────────────────────────────────────────────
p1_cfg   = CFG["phase_1"]
schedule = optax.cosine_decay_schedule(
    p1_cfg["learning_rate"],
    decay_steps=p1_cfg["nb_epochs"] * len(train_loader))
optimiser = optax.adam(schedule)

# Freeze everything except encoder and decoder
filter_spec = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda m: (m.encoder, m.decoder),
    filter_spec,
    replace=(
        jax.tree_util.tree_map(eqx.is_inexact_array, model.encoder),
        jax.tree_util.tree_map(eqx.is_inexact_array, model.decoder),
    )
)

trainable, frozen = eqx.partition(model, filter_spec)
opt_state = optimiser.init(eqx.filter(trainable, eqx.is_array))

# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────
@eqx.filter_jit
def train_step(model, frames, opt_state):
    def loss_fn(trainable):
        m = eqx.combine(trainable, frozen)
        return p1_loss_batch(m, frames)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        eqx.filter(model, filter_spec))
    updates, opt_state_new = optimiser.update(
        grads, opt_state, eqx.filter(model, filter_spec))
    new_trainable = optax.apply_updates(
        eqx.filter(model, filter_spec), updates)
    new_model = eqx.combine(new_trainable, frozen)
    return new_model, opt_state_new, loss


@eqx.filter_jit
def eval_step(model, frames):
    _, recons = jax.vmap(p1_forward_single, in_axes=(None, 0))(model, frames)
    return recons

print("=" * 60)
print("Phase 1: Tokenizer Pre-training")
print("=" * 60)
train_losses, test_losses = [], []

for epoch in range(1, p1_cfg["nb_epochs"] + 1):
    epoch_start = time.time()
    batch_losses = []

    for batch in train_loader:
        # Flatten time dimension if needed
        if batch.ndim == 5:
            B2, T2, H2, W2, C2 = batch.shape
            batch = batch.reshape(B2 * T2, H2, W2, C2)
        batch = jnp.array(batch)
        model, opt_state, loss = train_step(model, batch, opt_state)
        batch_losses.append(float(loss))

    # Validation
    val_losses = []
    for batch in test_loader:
        if batch.ndim == 5:
            batch = batch.reshape(-1, *batch.shape[2:])
        batch = jnp.array(batch)
        val_losses.append(float(p1_loss_batch(model, batch)))

        train_losses.append(np.mean(batch_losses))
        test_losses.append(np.mean(val_losses))

    if epoch % p1_cfg["print_every"] == 0 or epoch == 1:
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{p1_cfg['nb_epochs']} | "
              f"train MSE={train_losses[-1]:.5f}  "
              f"val MSE={test_losses[-1]:.5f}  "
              f"[{elapsed:.1f}s]", flush=True)
        
    ## Visualise reconstructions every 10 epochs
    if epoch % max(1, p1_cfg["nb_epochs"] // 10) == 0 or epoch == p1_cfg["nb_epochs"]-1:
        print("Recon shape:", sample_batch.shape, flush=True)
        recon = eval_step(model, sample_batch)

        plot_videos(
            np.expand_dims(recon, axis=1)[0] if recon.ndim == 4 else recon[0],
            np.expand_dims(sample_batch, axis=1)[0] if sample_batch.ndim == 4 else sample_batch[0],
            show_borders=True,
            corner_radius=5,
            no_rescale=True,
            cmap="grey" if CFG["dataset"].lower() == "movingmnist" else "viridis",
            plot_ref=True, show_titles=True, save_name=f"plots/p1_epoch{epoch+1}.png"
        )

# ─────────────────────────────────────────────────────────────────
# Save encoder & decoder
# ─────────────────────────────────────────────────────────────────
eqx.tree_serialise_leaves(ARTIFACTS / "genie_encoder.eqx", model.encoder)
eqx.tree_serialise_leaves(ARTIFACTS / "genie_decoder.eqx", model.decoder)
eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase1.eqx",  model)
print(f"\n✅  Saved encoder/decoder to {ARTIFACTS}/")

#%%
# ─────────────────────────────────────────────────────────────────
# Visualise reconstructions
# ─────────────────────────────────────────────────────────────────
def _vis_reconstructions(model, loader, n=4, fname="plots/p1_reconstructions.png"):
    batch = next(iter(loader))
    if batch.ndim == 5:
        batch = batch.reshape(-1, *batch.shape[2:])
    batch = jnp.array(batch[:n])

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        z     = model.encode(batch[i])
        recon = model.decode(z, batch[i].shape)
        orig  = np.array(batch[i])
        rec   = np.array(recon)
        cmap  = "gray" if orig.shape[-1] == 1 else None
        axes[0, i].imshow(orig[..., 0] if cmap else orig, cmap=cmap)
        axes[0, i].set_title(f"GT t={i+1}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec[..., 0] if cmap else rec, cmap=cmap)
        axes[1, i].set_title(f"Recon t={i+1}", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("Phase 1 — Tokenizer Reconstructions", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

_vis_reconstructions(model, test_loader)

# Loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train MSE", linewidth=2)
ax.plot(test_losses,  label="Val MSE",   linewidth=2, linestyle="--")
# ax.set_xlabel("Epoch")
ax.set_xlabel("Train Steps")
ax.set_yscale("log")
ax.set_ylabel("MSE")
ax.set_title("Phase 1 — Tokenizer Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.draw()
plt.savefig("plots/p1_loss.png", dpi=120)
# plt.close(fig)
print("  Saved plots/p1_loss.png")
print("\nPhase 1 complete.\n")
