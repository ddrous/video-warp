#%%
"""
phase2.py — Genie Phase 2: Dynamics Training (IDM + Transformer)
=================================================================

Trains the Inverse Dynamics Model (IDM / LAM) and the Dynamics
Transformer jointly.  The VQ-VAE (encoder, VQ, decoder) is frozen.

Loss
----
    L = w_dyn · L_dyn + w_idm · L_idm_vq

where
    L_dyn    = CrossEntropy( dynamics(z_idx[:-1], a_idx),  z_idx[1:] )
               — next-frame-token prediction (classification over K)
    L_idm_vq = VQ commitment + codebook loss for the action quantizer

Why cross-entropy and not MSE?
  The frame targets are *discrete indices* (hard class labels from the
  codebook).  Cross-entropy provides a direct, strong gradient signal:
  the transformer learns to predict the exact next token rather than a
  blurry continuous average.  This is the key reason Phase 2 works with
  the discrete Genie formulation.

Teacher forcing is implicit: the input sequence always uses the
ground-truth z_idx[:-1] (not the transformer's own predictions).

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
ARTIFACTS = Path("artefacts"); ARTIFACTS.mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
train_loader, test_loader = get_dataloaders(CFG, phase="phase_2")
sample = next(iter(train_loader))
B, T, H, W, C = sample.shape
print(f"Phase-2 batch: B={B} T={T} H={H} W={W} C={C}")

#%%
# ─────────────────────────────────────────────────────────────────
# Model — load Phase-1 checkpoint
# ─────────────────────────────────────────────────────────────────
key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

ckpt = ARTIFACTS / "genie_phase1.eqx"
if ckpt.exists():
    model = eqx.tree_deserialise_leaves(ckpt, model)
    print(f"✅  Loaded Phase-1 checkpoint from {ckpt}")
else:
    print("⚠️   No Phase-1 checkpoint; using random init")

print(f"IDM       params : {count_trainable_params(model.idm):,}")
print(f"Dynamics  params : {count_trainable_params(model.dynamics):,}\n")

#%%
# ─────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────
def augment_batch(videos, key):
    """videos: (B, T, H, W, C)"""
    p2 = CFG["phase_2"]
    k1, k2 = jax.random.split(key)
    if p2.get("reverse_aug", True):
        flags = jax.random.bernoulli(k1, 0.5, (videos.shape[0],))
        videos = jax.vmap(
            lambda rev, v: jax.lax.cond(rev,
                lambda x: jnp.flip(x, axis=0), lambda x: x, v)
        )(flags, videos)
    if p2.get("static_aug", True):
        pad = T // 4
        flags = jax.random.bernoulli(k2, 0.5, (videos.shape[0],))
        videos = jax.vmap(
            lambda front, v: jax.lax.cond(front,
                lambda x: jnp.concatenate(
                    [jnp.repeat(x[:1], pad, 0), x[:T-pad]], 0),
                lambda x: jnp.concatenate(
                    [x[pad:], jnp.repeat(x[T-pad:T-pad+1], pad, 0)], 0),
                v)
        )(flags, videos)
    return videos

#%%
# ─────────────────────────────────────────────────────────────────
# Phase-2 Forward — one video sequence
# ─────────────────────────────────────────────────────────────────
def p2_forward(model, video):
    """
    video : (T, H, W, C)

    1. Encode every frame → z_idx (T,) using FROZEN VQ-VAE
    2. IDM: extract action token for each consecutive pair → a_idx (T-1,)
    3. Dynamics Transformer: predict logits over frame codebook → (T-1, K)
    4. Loss: CE(logits, z_idx[1:]) + IDM VQ loss

    Returns (total_loss, ce_loss, idm_vq_loss, z_idx, a_idx, logits)
    """
    T_loc = video.shape[0]

    # ── Encode all frames (VQ-VAE frozen) ────────────────────────
    def enc_one(frame):
        z_idx, z_q_st, vq_loss, z_e = model.encode_frame(frame)
        # stop grad: VQ-VAE is frozen
        return jax.lax.stop_gradient(z_idx), jax.lax.stop_gradient(z_q_st)

    z_idx_all, z_q_all = jax.vmap(enc_one)(video)   # (T,), (T, d_vq)

    # ── IDM: discrete action tokens ───────────────────────────────
    def idm_one(z_q_t, z_q_tp1):
        a_idx, a_emb, vq_loss = model.extract_action(z_q_t, z_q_tp1)
        return a_idx, vq_loss

    a_idx_all, idm_vq_losses = jax.vmap(idm_one)(
        z_q_all[:-1], z_q_all[1:])              # (T-1,), (T-1,)

    # ── Dynamics Transformer ──────────────────────────────────────
    # Input:  z_idx[:-1] (first T-1 frames) + a_idx (T-1 actions)
    # Target: z_idx[1:]  (next  T-1 frames)
    logits  = model.dynamics(z_idx_all[:-1], a_idx_all)  # (T-1, frame_K)
    targets = z_idx_all[1:]                               # (T-1,)

    ce_loss     = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                      logits, targets))
    idm_vq_loss = jnp.mean(idm_vq_losses)

    w_dyn = CFG["phase_2"].get("dyn_loss_weight", 1.0)
    w_idm = CFG["phase_2"].get("idm_loss_weight", 0.1)
    total = w_dyn * ce_loss + w_idm * idm_vq_loss

    return total, ce_loss, idm_vq_loss, z_idx_all, a_idx_all, logits

#%%
# ─────────────────────────────────────────────────────────────────
# Optimiser — IDM + Dynamics only
# ─────────────────────────────────────────────────────────────────
p2_cfg = CFG["phase_2"]
total_steps = p2_cfg["nb_epochs"] * len(train_loader)
schedule    = optax.cosine_decay_schedule(p2_cfg["learning_rate"], total_steps)
optimiser   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

all_false   = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda m: (m.idm, m.dynamics), all_false,
    replace=(
        jax.tree_util.tree_map(eqx.is_inexact_array, model.idm),
        jax.tree_util.tree_map(eqx.is_inexact_array, model.dynamics),
    ))
# Optionally fine-tune encoder
if p2_cfg.get("train_encoder", False):
    filter_spec = eqx.tree_at(
        lambda m: m.encoder, filter_spec,
        replace=jax.tree_util.tree_map(eqx.is_inexact_array, model.encoder))

frozen    = eqx.filter(model, jax.tree_util.tree_map(lambda x: not x, filter_spec))
opt_state = optimiser.init(eqx.filter(model, filter_spec))

aug_key = jax.random.PRNGKey(SEED + 99)

@eqx.filter_jit
def train_step(model, videos, opt_state):
    def batch_loss(trainable):
        m = eqx.combine(trainable, frozen)
        totals, _, _, _, _, _ = jax.vmap(p2_forward, in_axes=(None, 0))(m, videos)
        return jnp.mean(totals)
    loss, grads = eqx.filter_value_and_grad(batch_loss)(eqx.filter(model, filter_spec))
    upd, new_opt = optimiser.update(grads, opt_state, eqx.filter(model, filter_spec))
    return eqx.combine(optax.apply_updates(eqx.filter(model, filter_spec), upd), frozen), \
           new_opt, loss

#%%
# ─────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────
def _prep(f):
    f = np.clip(np.array(f), 0, 1)
    return ("gray", f[..., 0]) if f.shape[-1] == 1 else (None, f)

def vis_rollout(model, videos, step, n=4, fname=None):
    """
    Show GT video strips vs reconstructions decoded from *predicted* tokens.
    Top row: GT  |  Bottom row: decoded from Transformer prediction
    """
    n = min(n, videos.shape[0])
    fig, big_axes = plt.subplots(2 * n, T,
                                  figsize=(1.8 * T, 2.2 * n * 2))
    if n == 1: big_axes = big_axes[np.newaxis]

    for i in range(n):
        video = jnp.array(videos[i])
        _, ce, _, z_idx, a_idx, logits = p2_forward(model, video)

        # Greedy-decode predicted next token at each step
        pred_idx = jnp.argmax(logits, axis=-1)   # (T-1,)
        gt_idx   = jnp.array(z_idx)

        for t in range(T):
            ax_gt   = big_axes[2*i,     t]
            ax_pred = big_axes[2*i + 1, t]

            # GT frame
            cmap, img = _prep(videos[i, t])
            ax_gt.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax_gt.axis("off")
            if t == 0: ax_gt.set_ylabel("GT", rotation=0, labelpad=22,
                                          fontsize=10, fontweight="bold")

            # Predicted: decode token
            if t == 0:
                # First frame: use GT
                frame_pred = np.array(model.decode_from_index(gt_idx[0]))
            else:
                frame_pred = np.array(model.decode_from_index(pred_idx[t-1]))
            cmap2, img2 = _prep(frame_pred)
            ax_pred.imshow(img2, cmap=cmap2, vmin=0, vmax=1)
            ax_pred.axis("off")
            if t == 0: ax_pred.set_ylabel("Pred", rotation=0, labelpad=22,
                                            fontsize=10, fontweight="bold")
            if i == 0: ax_gt.set_title(f"t={t+1}", fontsize=8)

    fig.suptitle(f"Phase 2 — Dynamics Rollouts (step {step:,})", fontsize=12)
    plt.tight_layout()
    fname = fname or f"plots/p2_rollout_step{step:06d}.png"
    plt.savefig(fname, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  [vis] {fname}", flush=True)


def vis_action_usage(model, loader, step, max_batches=15):
    """Bar chart of action token usage."""
    counts = np.zeros(model.action_K, dtype=np.int64)
    for i, batch in enumerate(loader):
        batch = jnp.array(batch)
        for b in range(batch.shape[0]):
            _, _, _, _, a_idx, _ = p2_forward(model, batch[b])
            for a in np.array(a_idx).flatten():
                counts[a] += 1
        if i >= max_batches: break
    used = int((counts > 0).sum())
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(model.action_K), counts, color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Action token index"); ax.set_ylabel("Frequency")
    ax.set_title(f"Action codebook usage — {used}/{model.action_K} active  "
                 f"(step {step:,})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/p2_actions_step{step:06d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return used


def vis_loss_p2(xs, totals, ces, idm_vqs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(xs, totals,  label="Total",     lw=2, color="#333")
    ax1.plot(xs, ces,     label="CE (dyn)",  lw=2, color="#2196F3", linestyle="--")
    ax1.plot(xs, idm_vqs, label="IDM VQ",    lw=2, color="#F44336", linestyle=":")
    ax1.set_xlabel("Train step"); ax1.set_ylabel("Loss")
    ax1.set_title("Phase 2 — Loss per step"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.semilogy(xs, totals, lw=2, color="#333")
    ax2.semilogy(xs, ces,    lw=2, color="#2196F3", linestyle="--")
    ax2.set_xlabel("Train step")
    ax2.set_title("Phase 2 — Loss (log)"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/p2_loss.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def vis_token_accuracy(logits_batch, targets_batch, step):
    """Histogram of per-step token prediction accuracy."""
    acc_per_step = []
    for logits, targets in zip(np.array(logits_batch), np.array(targets_batch)):
        pred = np.argmax(logits, axis=-1)
        acc_per_step.append((pred == targets).mean())
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(acc_per_step)), acc_per_step, color="#9C27B0", alpha=0.75)
    ax.axhline(np.mean(acc_per_step), color="red", linestyle="--", linewidth=2,
               label=f"Mean={np.mean(acc_per_step):.3f}")
    ax.set_xlabel("Video in batch"); ax.set_ylabel("Token accuracy")
    ax.set_title(f"Next-token prediction accuracy (step {step:,})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/p2_accuracy_step{step:06d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

#%%
# ─────────────────────────────────────────────────────────────────
# Training loop — per step logging
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 2: Dynamics Training (IDM + Transformer)")
print(f"  Action codebook K={model.action_K}")
print(f"  Loss: CE(dyn) + IDM_VQ  (teacher forcing, discrete tokens)")
print(f"  Epochs={p2_cfg['nb_epochs']}  Steps/epoch={len(train_loader)}")
print("=" * 65)

xs, totals_log, ces_log, idmvqs_log = [], [], [], []
gstep = 0
vis_batch = jnp.array(next(iter(test_loader))[:4])

for epoch in range(1, p2_cfg["nb_epochs"] + 1):
    for batch in train_loader:
        aug_key, k = jax.random.split(aug_key)
        batch_aug = augment_batch(jnp.array(batch), k)

        model, opt_state, total = train_step(model, batch_aug, opt_state)

        # Detailed losses on un-augmented batch for clean logging
        totals, ces, idm_vqs, _, _, _ = jax.vmap(
            p2_forward, in_axes=(None, 0))(model, jnp.array(batch))
        ce_m  = float(jnp.mean(ces))
        idm_m = float(jnp.mean(idm_vqs))

        xs.append(gstep); totals_log.append(float(total))
        ces_log.append(ce_m); idmvqs_log.append(idm_m)
        gstep += 1

        if gstep % p2_cfg["print_every"] == 0:
            print(f"  step {gstep:6d} | ep {epoch:4d} | "
                  f"total={float(total):.4f}  "
                  f"CE={ce_m:.4f}  IDM_VQ={idm_m:.5f}", flush=True)

        if gstep % p2_cfg["vis_every"] == 0:
            vis_rollout(model, vis_batch, gstep)
            vis_action_usage(model, test_loader, gstep)
            vis_loss_p2(xs, totals_log, ces_log, idmvqs_log)

            # Accuracy on vis batch
            logits_b, targets_b = [], []
            for b in range(min(4, vis_batch.shape[0])):
                _, _, _, z_idx, _, logits = p2_forward(model, vis_batch[b])
                logits_b.append(np.array(logits))
                targets_b.append(np.array(z_idx[1:]))
            vis_token_accuracy(logits_b, targets_b, gstep)

#%%
# ─────────────────────────────────────────────────────────────────
# Final diagnostics
# ─────────────────────────────────────────────────────────────────
vis_rollout(model, vis_batch, gstep, fname="plots/p2_final_rollout.png")
vis_action_usage(model, test_loader, gstep)
vis_loss_p2(xs, totals_log, ces_log, idmvqs_log)

# Validation CE
val_ces = []
for batch in test_loader:
    _, ces, _, _, _, _ = jax.vmap(
        p2_forward, in_axes=(None, 0))(model, jnp.array(batch))
    val_ces.append(float(jnp.mean(ces)))
print(f"\nValidation CE={np.mean(val_ces):.4f}  "
      f"(~random baseline={np.log(model.frame_K):.2f})")

eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase2.eqx", model)
print(f"✅  Saved → {ARTIFACTS}/genie_phase2.eqx")
print("\nPhase 2 complete.\n")