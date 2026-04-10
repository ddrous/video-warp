"""
phase1.py — Genie Phase 1: Video Tokenizer Pre-training
========================================================
Trains the VQ-VAE based video tokenizer (per frame) on MovingMNIST.
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

#%%
# ----------------------------------------------------------------------
# Minimal loaders and plotting if not available
# ----------------------------------------------------------------------
try:
    from loaders import get_dataloaders
except ImportError:
    print("Warning: loaders module not found. Define get_dataloaders or adapt.")
    def get_dataloaders(cfg, phase):
        raise NotImplementedError

try:
    from utils import plot_videos
except ImportError:
    def plot_videos(pred, target, show_borders=True, corner_radius=5,
                    no_rescale=True, cmap="grey", plot_ref=True,
                    show_titles=True, save_name=None):
        fig, axes = plt.subplots(2, pred.shape[0], figsize=(pred.shape[0]*2, 4))
        for i in range(pred.shape[0]):
            axes[0, i].imshow(pred[i], cmap=cmap, interpolation='none')
            axes[0, i].axis('off')
            axes[1, i].imshow(target[i], cmap=cmap, interpolation='none')
            axes[1, i].axis('off')
        if show_titles:
            axes[0, 0].set_title("Predicted")
            axes[1, 0].set_title("Ground Truth")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

# ----------------------------------------------------------------------
# Load config
# ----------------------------------------------------------------------
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]
key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)

ARTIFACTS = Path("artefacts")
ARTIFACTS.mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
train_loader, test_loader = get_dataloaders(CFG, phase="phase_1")
sample_batch = next(iter(train_loader))

# Determine shape and ensure channel dimension
if sample_batch.ndim == 4:          # (B, H, W, C) or (B, H, W)
    B, H, W = sample_batch.shape[:3]
    C = sample_batch.shape[3] if sample_batch.ndim == 4 and sample_batch.shape[3] in (1,3) else 1
    T = 1
elif sample_batch.ndim == 5:        # (B, T, H, W, C) or (B, T, H, W)
    B, T, H, W = sample_batch.shape[:4]
    C = sample_batch.shape[4] if sample_batch.ndim == 5 and sample_batch.shape[4] in (1,3) else 1
else:
    raise ValueError(f"Unexpected batch shape: {sample_batch.shape}")

# If there is no channel dimension, add one
if sample_batch.ndim == 4 and sample_batch.shape[-1] not in (1,3):
    sample_batch = sample_batch[..., np.newaxis]
    C = 1
    print("Added channel dimension (C=1) to frames.")
elif sample_batch.ndim == 5 and sample_batch.shape[-1] not in (1,3):
    sample_batch = sample_batch[..., np.newaxis]
    C = 1
    print("Added channel dimension (C=1) to videos.")

print(f"Video shape: B={B} T={T} H={H} W={W} C={C}")

#%%
# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
from genie.gemini.models import Genie
from utils import count_trainable_params

key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

## Print model properties, count the number of parameters, etc.
print("Counting model parameters...")
# print("Genire Model Summary:", model)
print(f"Total parameters: {count_trainable_params(model):,}")
print(f" - Tokenizer parameters: {count_trainable_params(model.tokenizer):,}")
print(f" - IDM parameters: {count_trainable_params(model.idm):,}")
print(f" - Dynamics parameters: {count_trainable_params(model.dynamics):,}")
print(f" - GCM parameters: {count_trainable_params(model.gcm):,}")


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def compute_recon_loss(recon, target):
    return jnp.mean((recon - target) ** 2)

@eqx.filter_jit
def train_step(model, video_frames):
    """
    video_frames: (T, H, W, C)
    Returns (total_loss, recon_loss, vq_loss, recon)
    """
    # Ensure video_frames has 4 dimensions (T, H, W, C)
    if video_frames.ndim == 3:   # (H, W, C) or (H, W)
        video_frames = video_frames[None, ...]
    if video_frames.shape[-1] not in (1,3):
        video_frames = video_frames[..., np.newaxis]   # add channel

    # Fix: Updated method names to match the new ST-Transformer models.py
    indices, z_qs, vq_loss = model.tokenizer.encode(video_frames)
    recon = model.tokenizer.decode(z_qs, H, W)
    recon_loss = compute_recon_loss(recon, video_frames)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss, vq_loss, recon

@eqx.filter_jit
def batch_train_step(model, videos):
    """
    videos: (B, T, H, W, C)
    Returns (total_loss, (recon_loss, vq_loss, recons))
    where recons is (B, T, H, W, C)
    """
    total, recon, vq, recons = jax.vmap(train_step, in_axes=(None, 0))(model, videos)
    # Return loss as first element, auxiliary data as second element (tuple)
    return jnp.mean(total), (jnp.mean(recon), jnp.mean(vq), recons)

#%%
# ----------------------------------------------------------------------
# Optimizer and training loop
# ----------------------------------------------------------------------
p1_cfg = CFG["phase_1"]
total_steps = p1_cfg["nb_epochs"] * len(train_loader)
schedule = optax.cosine_decay_schedule(p1_cfg["learning_rate"], total_steps)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def update(model, videos, opt_state):
    (loss, (recon_loss, vq_loss, recons)), grads = eqx.filter_value_and_grad(
        batch_train_step, has_aux=True)(model, videos)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, recon_loss, vq_loss, recons

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
TRAIN = True
if TRAIN:
    print("=" * 65)
    print("Phase 1: Video Tokenizer Pre-training")
    print("=" * 65)
    start_time = time.time()
    epoch_losses = []
    all_losses = []
    vis_batch = sample_batch[:4]   # (4, T, H, W, C)
    vis_videos = jnp.array(vis_batch)

    for epoch in range(1, p1_cfg["nb_epochs"] + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = np.array(batch)
            # Ensure batch has channel dimension
            if batch.ndim == 4 and batch.shape[-1] not in (1,3):
                batch = batch[..., np.newaxis]
            elif batch.ndim == 5 and batch.shape[-1] not in (1,3):
                batch = batch[..., np.newaxis]
            batch = jnp.array(batch)   # (B, T, H, W, C)
            model, opt_state, loss, r_loss, v_loss, recons = update(model, batch, opt_state)
            epoch_loss += loss
            all_losses.append(loss)

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # Fix: Safely default to 1 if print_every is omitted from config.yaml
        if epoch % p1_cfg.get("print_every", 1) == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{p1_cfg['nb_epochs']} | Loss: {avg_loss:.5f} (recon={r_loss:.5f}, vq={v_loss:.5f})")

        # Visualize every 10% of epochs
        if epoch % max(1, p1_cfg["nb_epochs"] // 10) == 0:
            # Compute reconstruction for visualization (no gradients)
            _, (_, _, recons_vis) = batch_train_step(model, vis_videos)
            pred_frames = np.array(recons_vis[0])    # first video in batch
            gt_frames = np.array(vis_videos[0])
            # if C == 1:
            #     pred_frames = pred_frames[..., 0]
            #     gt_frames = gt_frames[..., 0]

            pred_frames = np.expand_dims(pred_frames, axis=1) if pred_frames.ndim == 3 else pred_frames
            gt_frames = np.expand_dims(gt_frames, axis=0) if gt_frames.ndim == 3 else gt_frames

            # print(f"Shapes for plotting: pred={pred_frames.shape}, gt={gt_frames.shape}")
            plot_videos(
                pred_frames, gt_frames,
                show_borders=True, cmap="grey", plot_ref=True,
                show_titles=True, save_name=f"plots/p1_epoch{epoch}.png"
            )

    print(f"\nTraining finished in {time.time()-start_time:.2f} sec")
    eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase1.eqx", model)
    np.save(ARTIFACTS / "p1_loss.npy", np.array(epoch_losses))

    # Plot loss curve
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 1 Training Loss")
    plt.savefig("plots/p1_loss.png")
    plt.close()
else:
    model = eqx.tree_deserialise_leaves(ARTIFACTS / "genie_phase1.eqx", model)
    print("Loaded Phase 1 checkpoint")

#%%
# ----------------------------------------------------------------------
# Plotting the loss curve for all batches (optional)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(all_losses, label="Batch Loss", alpha=0.5)
# ax.plot(np.convolve(all_losses, np.ones(10)/10, mode='valid'), label="Smoothed Loss", color='red')
ax.set_xlabel("Train Step")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.set_title("Phase 1 Training Loss (All Batches)")
ax.legend()
plt.savefig("plots/p1_all_batch_losses.png")
# plt.close() 


# ----------------------------------------------------------------------
# Final evaluation on test set
# ----------------------------------------------------------------------
test_batch = next(iter(test_loader))
if test_batch.ndim == 4 and test_batch.shape[-1] not in (1,3):
    test_batch = test_batch[..., np.newaxis]
elif test_batch.ndim == 5 and test_batch.shape[-1] not in (1,3):
    test_batch = test_batch[..., np.newaxis]

test_batch = jnp.array(test_batch[:4])
_, (_, _, recons_test) = batch_train_step(model, test_batch)
pred_frames = np.array(recons_test[0])
# pred_frames = np.expand_dims(pred_frames, axis=1)

gt_frames = np.array(test_batch[0])
gt_frames = np.expand_dims(gt_frames, axis=0)

# if C == 1:
#     pred_frames = pred_frames[..., 0]
#     gt_frames = gt_frames[..., 0]

print(f"Test batch shapes: pred={pred_frames.shape}, gt={gt_frames.shape}")

plot_videos(
    pred_frames, gt_frames,
    show_borders=True, cmap="grey", plot_ref=True,
    show_titles=True, save_name="plots/p1_test_recon.png"
)
print("Phase 1 completed. Check plots/ for visualizations.")
