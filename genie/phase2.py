#%%
"""
phase2.py — Genie Phase 2: Dynamics Training (IDM + Transformer)
=================================================================

Trains the Inverse Dynamics Model (IDM / LAM) and the Dynamics
Transformer jointly in latent space. The Encoder is frozen (loaded
from Phase 1). The Decoder is not used in this phase.

Loss
----
  L₂ = (1/T) Σ_t  ||z_{t+1}  –  ẑ_{t+1}||²
where z_{t+1}  = stop_gradient(Encoder(o_{t+1}))  (ground truth)
and   ẑ_{t+1} = DynamicsTransformer(z_{1:t}, u_{1:t})  with
      u_t      = IDM(z_t, z_{t+1}).

Usage (from the genie/ directory):
    python phase2.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from loaders import get_dataloaders
from utils import plot_videos, count_trainable_params
from genie.models import Genie


#%%

# ─────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]
key  = jax.random.PRNGKey(SEED)
np.random.seed(SEED)

ARTIFACTS = Path("artefacts")
ARTIFACTS.mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
train_loader, test_loader = get_dataloaders(CFG, phase="phase_2")
sample_batch = next(iter(train_loader))
B, T, H, W, C = sample_batch.shape
print(f"Phase-2 video batch: B={B}, T={T}, H={H}, W={W}, C={C}")

# ─────────────────────────────────────────────────────────────────
# Model — load Phase-1 tokenizer
# ─────────────────────────────────────────────────────────────────
key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

enc_path = ARTIFACTS / "genie_encoder.eqx"
if enc_path.exists():
    model = eqx.tree_at(
        lambda m: m.encoder,
        model,
        eqx.tree_deserialise_leaves(enc_path, model.encoder))
    print(f"✅  Loaded encoder from {enc_path}")
else:
    print("⚠️   No Phase-1 encoder found; training from scratch.")

print(f"IDM params      : {count_trainable_params(model.idm):,}")
print(f"Dynamics params : {count_trainable_params(model.dynamics):,}\n")

# ─────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────
def augment_videos(videos: jnp.ndarray, key) -> jnp.ndarray:
    """videos: (B, T, H, W, C)"""
    p2 = CFG["phase_2"]
    k1, k2 = jax.random.split(key)
    if p2.get("reverse_aug", True):
        do_rev = jax.random.bernoulli(k1, 0.5, (videos.shape[0],))
        videos = jax.vmap(
            lambda rev, v: jax.lax.cond(
                rev, lambda x: jnp.flip(x, axis=0), lambda x: x, v)
        )(do_rev, videos)
    if p2.get("static_aug", True):
        pad = T // 4
        add_front = jax.random.bernoulli(k2, 0.5, (videos.shape[0],))
        videos = jax.vmap(
            lambda front, v: jax.lax.cond(
                front,
                lambda x: jnp.concatenate(
                    [jnp.repeat(x[:1], pad, axis=0), x[:T-pad]], axis=0),
                lambda x: jnp.concatenate(
                    [x[pad:], jnp.repeat(x[T-pad:T-pad+1], pad, axis=0)], axis=0),
                v)
        )(add_front, videos)
    return videos

# ─────────────────────────────────────────────────────────────────
# Phase-2 Forward (single video sequence)
# ─────────────────────────────────────────────────────────────────
def p2_forward(model, video):
    """
    video : (T, H, W, C)
    Returns (latent_loss, pixel_loss, z_seq, z_pred_seq)
    """
    T_loc = video.shape[0]

    # ── Encode all frames (stop-gradient: encoder is frozen) ─────
    def encode_one(frame):
        return jax.lax.stop_gradient(model.encode(frame))

    z_seq = jax.vmap(encode_one)(video)          # (T, d_z)

    # ── IDM: extract actions from consecutive pairs ───────────────
    u_seq = jax.vmap(model.extract_action)(
        z_seq[:-1], z_seq[1:])                    # (T-1, d_u)

    # ── Dynamics Transformer: predict next latents ────────────────
    # We use z_{1:T-1} and u_{1:T-1} to predict z_{2:T}
    z_input  = z_seq[:-1]                        # (T-1, d_z)
    z_target = jax.lax.stop_gradient(z_seq[1:])  # (T-1, d_z)
    z_pred   = model.predict_next_latent(z_input, u_seq)  # (T-1, d_z)

    latent_loss = jnp.mean((z_pred - z_target) ** 2)

    return latent_loss, z_pred, z_target


# ─────────────────────────────────────────────────────────────────
# Optimiser — IDM + Dynamics only (encoder frozen)
# ─────────────────────────────────────────────────────────────────
p2_cfg    = CFG["phase_2"]
schedule  = optax.cosine_decay_schedule(
    p2_cfg["learning_rate"],
    decay_steps=p2_cfg["nb_epochs"] * len(train_loader))
optimiser = optax.adam(schedule)

# Build filter spec
all_false = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda m: (m.idm, m.dynamics),
    all_false,
    replace=(
        jax.tree_util.tree_map(eqx.is_inexact_array, model.idm),
        jax.tree_util.tree_map(eqx.is_inexact_array, model.dynamics),
    )
)
# Optionally fine-tune encoder
if p2_cfg.get("train_encoder", False):
    filter_spec = eqx.tree_at(
        lambda m: m.encoder,
        filter_spec,
        replace=jax.tree_util.tree_map(eqx.is_inexact_array, model.encoder))

trainable_params = eqx.filter(model, filter_spec)
opt_state = optimiser.init(trainable_params)

frozen_model = eqx.filter(model, jax.tree_util.tree_map(lambda x: not x, filter_spec))

#%%

@eqx.filter_jit
def train_step(model, videos_batch, aug_key):
    """videos_batch: (B, T, H, W, C)"""
    # Apply augmentation
    videos_aug = augment_videos(videos_batch, aug_key)

    def batch_loss(trainable):
        m = eqx.combine(trainable, frozen_model)
        # vmap over batch
        losses, _, _ = jax.vmap(p2_forward, in_axes=(None, 0))(m, videos_aug)
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(batch_loss)(
        eqx.filter(model, filter_spec))
    updates, new_opt_state = optimiser.update(
        grads, opt_state, eqx.filter(model, filter_spec))
    new_trainable = optax.apply_updates(
        eqx.filter(model, filter_spec), updates)
    new_model = eqx.combine(new_trainable, frozen_model)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_step(model, videos_batch):
    """videos_batch: (B, T, H, W, C)"""
    losses, z_pred_seq, z_gt_seq = jax.vmap(p2_forward, in_axes=(None, 0))(
        model, videos_batch)
    ## Decode
    recon_pred = jax.vmap(
        lambda z_seq: jax.vmap(model.decode, in_axes=(0, None))(
            z_seq, videos_batch.shape[2:]))(z_pred_seq)  # (B, T-1, H, W, C)

    return losses, recon_pred, videos_batch[:, 1:]

# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 2: Dynamics Training (IDM + Transformer)")
print("=" * 60)
train_losses, test_losses = [], []
aug_key = jax.random.PRNGKey(SEED + 100)


start_time = time.time()
for epoch in range(1, p2_cfg["nb_epochs"] + 1):
    t0 = time.time()
    batch_ls = []
    for batch in train_loader:
        aug_key, k = jax.random.split(aug_key)
        batch = jnp.array(batch)
        model, opt_state, loss = train_step(model, batch, k)
        batch_ls.append(float(loss))
        train_losses.append(float(loss))

    # Validation
    val_ls = []
    for batch in test_loader:
        batch = jnp.array(batch)
        losses, _, _ = jax.vmap(p2_forward, in_axes=(None, 0))(model, batch)
        val_ls.append(float(jnp.mean(losses)))
        test_losses.append(float(jnp.mean(losses)))

    # train_losses.append(np.mean(batch_ls))
    # test_losses.append(np.mean(val_ls))

    if epoch % p2_cfg["print_every"] == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{p2_cfg['nb_epochs']} | "
              f"train={train_losses[-1]:.6f}  "
              f"val={test_losses[-1]:.6f}  "
              f"[{time.time()-t0:.1f}s]", flush=True)

    if epoch % max(1, p2_cfg["nb_epochs"] // 10) == 0 or epoch == p2_cfg["nb_epochs"]-1:
        print("Recon shape:", sample_batch.shape, flush=True)
        recon, gt_frames = eval_step(model, sample_batch)[1:]

        plot_videos(
            np.expand_dims(recon, axis=1)[0] if recon.ndim == 4 else recon[0],
            np.expand_dims(gt_frames, axis=1)[0] if gt_frames.ndim == 4 else gt_frames[0],
            show_borders=True,
            corner_radius=5,
            no_rescale=True,
            cmap="grey" if CFG["dataset"].lower() == "movingmnist" else "viridis",
            plot_ref=True, show_titles=True, save_name=f"plots/p1_epoch{epoch+1}.png"
        )

end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅  Phase 2 training complete in HH:MM:SS = {elapsed//3600:.0f}:{(elapsed%3600)//60:.0f}:{elapsed%60:.1f}\n")

# ─────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────
eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase2.eqx", model)
print(f"\n✅  Saved Phase-2 model to {ARTIFACTS}/genie_phase2.eqx")


#%%

# ─────────────────────────────────────────────────────────────────
# Loss curve
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train", linewidth=2)
ax.plot(test_losses,  label="Val",   linewidth=2, linestyle="--")
ax.set_xlabel("Train Steps"); ax.set_ylabel("Latent MSE")
ax.set_title("Phase 2 — IDM + Dynamics Transformer Loss")
ax.set_yscale("log")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig("plots/p2_loss.png", dpi=120)
# plt.close()
print("  Saved plots/p2_loss.png")

# ─────────────────────────────────────────────────────────────────
# Visualise latent-space rollout quality
# ─────────────────────────────────────────────────────────────────
def _vis_rollout_latent(model, loader, n=4,
                        fname="plots/p2_rollout_vis.png"):
    batch = jnp.array(next(iter(loader))[:n])   # (n, T, H, W, C)
    fig, axes = plt.subplots(n, T, figsize=(2*T, 2.5*n))
    for i in range(n):
        video  = batch[i]
        _, z_pred, z_gt = p2_forward(model, video)
        for t in range(T - 1):
            recon_gt   = np.array(model.decode(z_gt[t], video.shape[1:]))
            recon_pred = np.array(model.decode(z_pred[t], video.shape[1:]))
            # Show GT on top, prediction below in the same column
            # Alternate rows: even = GT, odd = Pred
            pass
        # Simpler: show GT frames vs decoder output from predicted latent
        gt_frames   = np.array(video)            # (T, H, W, C)
        pred_frames = np.array(jax.vmap(model.decode, in_axes=(0, None))(
            z_pred, video.shape[1:]))            # (T-1, H, W, C)
        for t in range(T):
            ax = axes[i, t] if n > 1 else axes[t]
            frame = gt_frames[t] if t == 0 else (
                pred_frames[t-1] if t-1 < len(pred_frames) else gt_frames[t])
            cmap = "gray" if frame.shape[-1] == 1 else None
            ax.imshow(frame[..., 0] if cmap else frame, cmap=cmap,
                      vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"t={t+1}", fontsize=8)
    fig.suptitle("Phase 2 Rollouts (GT frame 1, then Predicted Decoder outputs)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    # plt.close()
    print(f"  Saved {fname}")

# _vis_rollout_latent(model, test_loader)
print("\nPhase 2 complete.\n")

