#%%
"""
phase1.py — Genie Phase 1: VQ-VAE Tokenizer Pre-training
==========================================================

Trains the Encoder, Vector Quantizer, and Decoder jointly using the
VQ-VAE objective (van den Oord et al., 2017):

    L = L_recon + L_codebook + β · L_commit

where
    L_recon     = ||x - x̂||²          (pixel reconstruction)
    L_codebook  = ||sg(z_e) - e||²     (move codebook entry → encoder output)
    L_commit    = ||z_e - sg(e)||²     (encoder commits to codebook entries)

Straight-through gradient is used so encoder gradients flow through
the discrete argmax.  After convergence every frame maps to a discrete
token index z_idx ∈ {0, …, K-1}, providing the foundation for Phase 2.

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
import yaml, time
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from loaders import get_dataloaders
from utils import plot_videos, count_trainable_params
from genie.gemini.models import Genie

# ─────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"];  key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
ARTIFACTS = Path("artefacts"); ARTIFACTS.mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
train_loader, test_loader = get_dataloaders(CFG, phase="phase_1")
sample_batch = next(iter(train_loader))
if sample_batch.ndim == 5:
    B, T, H, W, C = sample_batch.shape
else:
    B, H, W, C = sample_batch.shape
print(f"Frame shape: H={H} W={W} C={C}")

#%%
# ─────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────
key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

print(f"\nEncoder   params : {count_trainable_params(model.encoder):,}")
print(f"VQ        params : {count_trainable_params(model.vq):,}  "
      f"(K={model.frame_K} × d={model.d_vq})")
print(f"Decoder   params : {count_trainable_params(model.decoder):,}")
print(f"Total (all phases): {count_trainable_params(model):,}\n")

#%%
# ─────────────────────────────────────────────────────────────────
# Forward — single frame
# ─────────────────────────────────────────────────────────────────
@eqx.filter_jit
def p1_step_single(model, frame):
    """frame: (H,W,C) → (total_loss, recon_loss, vq_loss, recon, z_idx)"""
    z_idx, z_q_st, vq_loss, _ = model.encode_frame(frame)
    recon      = model.decode_frame(z_q_st)
    recon_loss = jnp.mean((recon - frame) ** 2)
    return recon_loss + vq_loss, recon_loss, vq_loss, recon, z_idx

@eqx.filter_jit
def p1_batch_loss(model, frames):
    """frames: (B,H,W,C) → (mean_total, mean_recon, mean_vq)"""
    total, recon, vq, _, _ = jax.vmap(
        p1_step_single, in_axes=(None, 0))(model, frames)
    return jnp.mean(total), jnp.mean(recon), jnp.mean(vq)

@eqx.filter_jit
def get_recons(model, frames):
    _, _, _, recons, _ = jax.vmap(p1_step_single, in_axes=(None, 0))(model, frames)
    return recons

@eqx.filter_jit
def get_indices(model, frames):
    _, _, _, _, idxs = jax.vmap(p1_step_single, in_axes=(None, 0))(model, frames)
    return idxs

#%%
# ─────────────────────────────────────────────────────────────────
# Optimiser — encoder + VQ + decoder only
# ─────────────────────────────────────────────────────────────────
p1_cfg = CFG["phase_1"]
total_steps = p1_cfg["nb_epochs"] * len(train_loader)
schedule   = optax.cosine_decay_schedule(p1_cfg["learning_rate"], total_steps)
optimiser  = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

all_false   = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda m: (m.encoder, m.vq, m.decoder), all_false,
    replace=(
        jax.tree_util.tree_map(eqx.is_inexact_array, model.encoder),
        jax.tree_util.tree_map(eqx.is_inexact_array, model.vq),
        jax.tree_util.tree_map(eqx.is_inexact_array, model.decoder),
    ))

frozen    = eqx.filter(model, jax.tree_util.tree_map(lambda x: not x, filter_spec))
opt_state = optimiser.init(eqx.filter(model, filter_spec))

@eqx.filter_jit
def train_step(model, frames, opt_state):
    def loss_fn(trainable):
        m = eqx.combine(trainable, frozen)
        total, r_loss, v_loss = p1_batch_loss(m, frames)
        return total, (r_loss, v_loss)
    (loss, aux_loss), grads  = eqx.filter_value_and_grad(loss_fn, has_aux=True)(eqx.filter(model, filter_spec))
    upd, new_opt = optimiser.update(grads, opt_state, eqx.filter(model, filter_spec))
    return eqx.combine(optax.apply_updates(eqx.filter(model, filter_spec), upd), frozen), \
           new_opt, loss, aux_loss

# @eqx.filter_jit
# def eval_step(model, frames):
#     total, recon, vq = p1_batch_loss(model, frames)
#     return total, recon, vq

#%%
# ─────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────
def _prep_frame(f):
    """(H,W,C) or (H,W,1) → plottable numpy, returns (cmap, img)"""
    f = np.array(f)
    if f.shape[-1] == 1:
        return "gray", f[..., 0]
    return None, np.clip(f, 0, 1)

def vis_recon_grid(model, frames, step, n=8, fname=None):
    n = min(n, frames.shape[0])
    recons = np.array(get_recons(model, jnp.array(frames[:n])))
    frames_np = np.array(frames[:n])
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.5))
    for i in range(n):
        for row, arr, label in [(0, frames_np[i], "GT"), (1, recons[i], "Recon")]:
            cmap, img = _prep_frame(arr)
            axes[row, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(label, rotation=0, labelpad=28,
                                         fontsize=12, fontweight="bold")
    fig.suptitle(f"Phase 1 — VQ-VAE Reconstructions  (step {step:,})", fontsize=12)
    plt.tight_layout()
    # fname = fname or f"plots/p1_recon_step{step:06d}.png"
    fname = fname or f"plots/p1_recon.png"
    plt.savefig(fname, dpi=100, bbox_inches="tight"); 
    # plt.close(fig)
    print(f"  [vis] {fname}", flush=True)

def vis_codebook_usage(model, loader, step, max_batches=20):
    counts = np.zeros(model.frame_K, dtype=np.int64)
    for i, batch in enumerate(loader):
        if batch.ndim == 5:
            batch = batch.reshape(-1, *batch.shape[2:])
        # Ensure conversion to a pure numpy array of integers
        idxs_jax = get_indices(model, jnp.array(batch))
        idxs = np.array(idxs_jax).astype(np.int64).flatten() 
        
        # Safe accumulation
        for idx in idxs:
            if 0 <= idx < model.frame_K:
                counts[idx] += 1
        if i >= max_batches: break

    used = int((counts > 0).sum())

    # print("Used and counts:", used, counts)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(model.frame_K), counts, width=1.0,
           color="#2196F3", alpha=0.75)
    ax.set_xlabel("Codebook entry"); ax.set_ylabel("Frequency")
    ax.set_title(f"Codebook usage — {used}/{model.frame_K} entries active  "
                 f"(step {step:,})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"plots/p1_codebook_step{step:06d}.png", dpi=100, bbox_inches="tight")
    plt.savefig(f"plots/p1_codebook.png", dpi=100, bbox_inches="tight")
    # plt.close(fig)

    return used

def vis_decoded_codebook(model, step, n=16):
    K = model.frame_K
    idxs = np.linspace(0, K-1, n, dtype=int)
    fig, axes = plt.subplots(2, n//2, figsize=(2.2 * n//2, 4.5))
    axes = axes.flatten()
    for i, idx in enumerate(idxs):
        frame = np.array(model.decode_from_index(jnp.array(idx, dtype=jnp.int32)))
        cmap, img = _prep_frame(frame)
        axes[i].imshow(img, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(f"#{idx}", fontsize=7); axes[i].axis("off")
    fig.suptitle(f"Decoded codebook entries (step {step:,})", fontsize=10)
    plt.tight_layout()
    # plt.savefig(f"plots/p1_codebook_decoded_step{step:06d}.png", dpi=100, bbox_inches="tight")
    plt.savefig(f"plots/p1_codebook_decoded.png", dpi=100, bbox_inches="tight")
    # plt.close(fig)

def vis_loss_curves(xs, totals, recons, vqs):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    # ax1.plot(xs, totals, label="Total",      lw=2, color="#333")
    # ax1.plot(xs, recons, label="Recon (MSE)",lw=2, color="#2196F3", linestyle="--")
    # ax1.plot(xs, vqs,    label="VQ",         lw=2, color="#F44336", linestyle=":")
    # ax1.set_xlabel("Train step"); ax1.set_ylabel("Loss")
    # ax1.set_title("Phase 1 — Loss per step"); ax1.legend(); ax1.grid(alpha=0.3)

    # ## Only plot the 96th perceptiles of each loss
    # vqs = np.array(vqs)
    # vq_96 = np.percentile(vqs, 96)
    # vqs = np.clip(vqs, None, vq_96)

    fig, ax2  = plt.subplots(1, 1, figsize=(8, 4))
    ax2.semilogy(xs, totals, lw=2, color="#333", label="Total")
    ax2.semilogy(xs, recons, lw=2, color="#2196F3", linestyle="--", label="Recons")
    ax2.semilogy(xs, vqs,    lw=2, color="#F44336", linestyle=":", label="VQ")
    ax2.set_xlabel("Train step"); 
    ax2.set_ylabel("Losses")
    ax2.set_title("Phase 1 — Loss");

    ## Crop the y xis to only show more than 
    ax2.set_ylim(bottom=np.min(totals)*0.9, top=np.max(totals)*1.1)

    ax2.grid(alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("plots/p1_loss.png", dpi=120, bbox_inches="tight"); 
    # plt.close(fig)

#%%
# ─────────────────────────────────────────────────────────────────
# Training loop — log per step
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 1: VQ-VAE Tokenizer Pre-training")
print(f"  Frame codebook K={model.frame_K}  d_vq={model.d_vq}  β={CFG['vq_beta']}")
print(f"  Epochs={p1_cfg['nb_epochs']}  Steps/epoch={len(train_loader)}")
print("=" * 65)

xs, totals_log, recons_log, vqs_log = [], [], [], []
gstep = 0
# Fixed visualisation sample (same frames throughout training)
vis_frames = jnp.array(
    sample_batch.reshape(-1, H, W, C)[:8]
    if sample_batch.ndim == 5 else sample_batch[:8])

for epoch in range(1, p1_cfg["nb_epochs"] + 1):
    for batch in train_loader:
        if batch.ndim == 5:
            batch = batch.reshape(-1, *batch.shape[2:])
        batch = jnp.array(batch)

        model, opt_state, loss, (r_loss, v_loss) = train_step(model, batch, opt_state)
        # _, r_loss, v_loss = p1_batch_loss(model, batch)

        xs.append(gstep)
        totals_log.append(float(loss))
        recons_log.append(float(r_loss))
        vqs_log.append(float(v_loss))
        gstep += 1

    if epoch % p1_cfg["print_every"] == 0:
        print(f"  step {gstep:6d} | ep {epoch:3d} | "
                f"total={float(loss):.5f}  "
                f"recon={float(r_loss):.5f}  "
                f"vq={float(v_loss):.5f}", flush=True)

    # if epoch % p1_cfg["vis_every"] == 0:
    if epoch % max(1, p1_cfg["nb_epochs"]//10) == 0:
        # vis_recon_grid(model, vis_frames, gstep)

        # vis_decoded_codebook(model, gstep)
        # vis_loss_curves(xs, totals_log, recons_log, vqs_log)
        # used = vis_codebook_usage(model, train_loader, gstep)
        # print(f"    codebook: {used}/{model.frame_K} entries active", flush=True)

        eval_recon = get_recons(model, vis_frames)
        # print(f"    [eval] total={float(eval_total):.5f} recon={float(eval_recon):.5f}  vq={float(eval_vq):.5f}", flush=True)

        # print(f"Shapes — eval_recon: {eval_recon.shape}  sample_batch: {sample_batch.shape}")

        plot_videos(
            np.expand_dims(eval_recon, axis=1)[0] if eval_recon.ndim == 4 else eval_recon[0],
            np.expand_dims(vis_frames, axis=1)[0] if vis_frames.ndim == 4 else vis_frames[0],
            show_borders=True,
            corner_radius=5,
            no_rescale=True,
            cmap="grey" if CFG["dataset"].lower() == "movingmnist" else "viridis",
            plot_ref=True, show_titles=True, save_name=f"plots/p1_epoch{epoch+1}.png"
        )

#%%
# ─────────────────────────────────────────────────────────────────
# Final diagnostics
# ─────────────────────────────────────────────────────────────────
vis_recon_grid(model, vis_frames, gstep, fname="plots/p1_final_recon.png")
vis_decoded_codebook(model, gstep)
vis_loss_curves(xs, totals_log, recons_log, vqs_log)
vis_codebook_usage(model, train_loader, gstep)

# Validation
val_t, val_r, val_v = [], [], []
for batch in test_loader:
    if batch.ndim == 5:
        batch = batch.reshape(-1, *batch.shape[2:])
    t, r, v = p1_batch_loss(model, jnp.array(batch))
    val_t.append(float(t)); val_r.append(float(r)); val_v.append(float(v))
print(f"\nVal — total={np.mean(val_t):.5f}  "
      f"recon={np.mean(val_r):.5f}  vq={np.mean(val_v):.5f}")

eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase1.eqx", model)
print(f"✅  Saved → {ARTIFACTS}/genie_phase1.eqx")
print("\nPhase 1 complete.\n")
