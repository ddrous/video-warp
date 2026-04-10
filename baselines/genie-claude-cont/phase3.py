"""
phase3.py — Genie Phase 3: GCM Training + Inference + State-Swap
=================================================================

Phase 3 trains the Generative Control Model (GCM), a causal RNN
that learns to predict actions autoregressively, without access to
future frames.  All other components (Encoder, IDM, Dynamics
Transformer, Decoder) are frozen.

Inference modes
---------------
1. Context-conditioned rollout : first ρ·T steps use IDM (oracle
   actions), the remainder use GCM (autonomous prediction).
2. Fully autonomous rollout    : ρ = 0; GCM drives everything.
3. State-swap experiment       : states from video A, actions from
   video B injected mid-rollout to test for state-action entanglement.

Usage (from the genie/ directory):
    python phase3.py
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
from genie.gemini.models import Genie

# ─────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]
key  = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
ARTIFACTS = Path("artefacts");  ARTIFACTS.mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
train_loader, test_loader = get_dataloaders(CFG, phase="phase_3")
sample = next(iter(train_loader))
B, T, H, W, C = sample.shape
print(f"Phase-3 batch: B={B}, T={T}, H={H}, W={W}, C={C}")

# ─────────────────────────────────────────────────────────────────
# Model — load Phase-2 checkpoint
# ─────────────────────────────────────────────────────────────────
key, subkey = jax.random.split(key)
model = Genie(CFG, frame_shape=(H, W, C), key=subkey)

ckpt = ARTIFACTS / "genie_phase2.eqx"
if ckpt.exists():
    model = eqx.tree_deserialise_leaves(ckpt, model)
    print(f"✅  Loaded Phase-2 checkpoint from {ckpt}")
else:
    print("⚠️   No Phase-2 checkpoint found; using random init.")

print(f"GCM params: {count_trainable_params(model.gcm):,}\n")

# ─────────────────────────────────────────────────────────────────
# Phase-3 Forward  (GCM imitation of IDM)
# ─────────────────────────────────────────────────────────────────
def p3_forward(model, video):
    """
    video : (T, H, W, C)
    GCM is trained to predict u_t = IDM(z_t, z_{t+1}) using only
    (z_t, û_{t-1}) as input, in a teacher-forcing regime.

    Returns
    -------
    gcm_loss : scalar — L2 between GCM-predicted and IDM-extracted actions
    u_gcm    : (T-1, d_u)  — GCM action predictions
    u_idm    : (T-1, d_u)  — IDM ground-truth actions
    """
    # Encode all frames (frozen)
    z_seq = jax.vmap(lambda f: jax.lax.stop_gradient(model.encode(f)))(video)

    # IDM target actions (frozen)
    u_idm = jax.lax.stop_gradient(
        jax.vmap(model.extract_action)(z_seq[:-1], z_seq[1:]))   # (T-1, d_u)

    # GCM rollout (teacher-forcing: feed u_idm as previous action)
    gcm_init = model.gcm.initial_state()
    u_prev0  = jnp.zeros(model.d_u)

    def scan_fn(carry, inputs):
        gcm_h, u_prev = carry
        z_t, u_t_target = inputs
        gcm_h_new, u_t_pred = model.predict_action_gcm(gcm_h, z_t, u_prev)
        # Teacher forcing: next u_prev is the IDM target (not the prediction)
        u_prev_new = jax.lax.stop_gradient(u_t_target)
        loss_t = jnp.mean((u_t_pred - u_t_target) ** 2)
        return (gcm_h_new, u_prev_new), (u_t_pred, loss_t)

    _, (u_gcm, step_losses) = jax.lax.scan(
        scan_fn,
        (gcm_init, u_prev0),
        (z_seq[:-1], u_idm))

    gcm_loss = jnp.mean(step_losses)
    return gcm_loss, u_gcm, u_idm


# ─────────────────────────────────────────────────────────────────
# Optimiser — only GCM trainable
# ─────────────────────────────────────────────────────────────────
p3_cfg = CFG["phase_3"]
schedule = optax.cosine_decay_schedule(
    p3_cfg["learning_rate"],
    decay_steps=p3_cfg["nb_epochs"] * len(train_loader))
optimiser = optax.adam(schedule)

all_false   = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda m: m.gcm,
    all_false,
    replace=jax.tree_util.tree_map(eqx.is_inexact_array, model.gcm))

frozen = eqx.filter(model, jax.tree_util.tree_map(lambda x: not x, filter_spec))
opt_state = optimiser.init(eqx.filter(model, filter_spec))


@eqx.filter_jit
def train_step(model, batch):
    def batch_loss(trainable):
        m = eqx.combine(trainable, frozen)
        losses, _, _ = jax.vmap(p3_forward, in_axes=(None, 0))(m, batch)
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(batch_loss)(
        eqx.filter(model, filter_spec))
    updates, new_opt = optimiser.update(
        grads, opt_state, eqx.filter(model, filter_spec))
    new_trainable = optax.apply_updates(
        eqx.filter(model, filter_spec), updates)
    return eqx.combine(new_trainable, frozen), new_opt, loss


# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 3: GCM Training")
print("=" * 60)
train_losses, val_losses = [], []

for epoch in range(1, p3_cfg["nb_epochs"] + 1):
    t0 = time.time()
    bl = []
    for batch in train_loader:
        batch = jnp.array(batch)
        model, opt_state, loss = train_step(model, batch)
        bl.append(float(loss))

    vl = []
    for batch in test_loader:
        losses, _, _ = jax.vmap(p3_forward, in_axes=(None, 0))(
            model, jnp.array(batch))
        vl.append(float(jnp.mean(losses)))

    train_losses.append(np.mean(bl))
    val_losses.append(np.mean(vl))

    if epoch % p3_cfg["print_every"] == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{p3_cfg['nb_epochs']} | "
              f"GCM train={train_losses[-1]:.6f}  "
              f"val={val_losses[-1]:.6f}  "
              f"[{time.time()-t0:.1f}s]", flush=True)

# Save
eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase3.eqx", model)
print(f"\n✅  Saved Phase-3 model to {ARTIFACTS}/genie_phase3.eqx")

# Loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_losses, label="Train GCM", linewidth=2)
ax.plot(val_losses,   label="Val GCM",   linewidth=2, linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Action MSE")
ax.set_title("Phase 3 — GCM Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/p3_gcm_loss.png", dpi=120)
plt.close()


# ─────────────────────────────────────────────────────────────────
# ── INFERENCE FUNCTIONS ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def rollout_context(model, video, context_steps):
    """
    Context-conditioned rollout.
    First `context_steps` transitions are IDM-driven;
    remaining T-1-context_steps are GCM-driven.

    Returns predicted frames: (T-1, H, W, C)
    """
    T_loc = video.shape[0]
    z0    = model.encode(video[0])

    # Encode all reference frames for context phase
    z_all = jax.vmap(model.encode)(video)   # (T, d_z)
    u_all = jax.vmap(model.extract_action)(z_all[:-1], z_all[1:])  # (T-1, d_u)

    gcm_h  = model.gcm.initial_state()
    u_prev = jnp.zeros(model.d_u)

    pred_z   = []
    pred_frames = []
    z_t = z0

    for t in range(T_loc - 1):
        if t < context_steps:
            u_t = jax.lax.stop_gradient(u_all[t])
            # Update GCM state with ground-truth action (teacher forcing)
            gcm_h, _ = model.predict_action_gcm(gcm_h, z_t, u_prev)
            u_prev = u_t
        else:
            gcm_h, u_t = model.predict_action_gcm(gcm_h, z_t, u_prev)
            u_prev = u_t

        # Predict next latent via Transformer
        z_hist = jnp.stack(pred_z + [z_t])       # (t+1, d_z)
        u_hist = jnp.stack([u_all[i] if i < context_steps
                             else u_prev
                             for i in range(t + 1)])    # (t+1, d_u)
        # Use the last Transformer prediction
        z_preds = model.predict_next_latent(z_hist, u_hist)
        z_next  = z_preds[-1]

        pred_z.append(z_t)
        frame = model.decode(z_next, video.shape[1:])
        pred_frames.append(frame)
        z_t = z_next

    return jnp.stack(pred_frames)   # (T-1, H, W, C)


def rollout_autonomous(model, z0, T_gen):
    """
    Fully autonomous rollout starting from z0.
    No ground-truth frames used after the first.
    Returns predicted frames: (T_gen, H, W, C)
    """
    gcm_h  = model.gcm.initial_state()
    u_prev = jnp.zeros(model.d_u)
    z_t    = z0
    frames = []
    z_hist = [z0]
    u_hist = []

    for t in range(T_gen):
        gcm_h, u_t = model.predict_action_gcm(gcm_h, z_t, u_prev)
        u_prev = u_t
        u_hist.append(u_t)

        z_seq_arr = jnp.stack(z_hist)
        u_seq_arr = jnp.stack(u_hist)
        z_preds   = model.predict_next_latent(z_seq_arr, u_seq_arr)
        z_next    = z_preds[-1]

        frame = model.decode(z_next, (64, 64, 1))   # adjust if not MNIST
        frames.append(frame)
        z_hist.append(z_next)
        z_t = z_next

    return jnp.stack(frames)


# ─────────────────────────────────────────────────────────────────
# ── STATE-SWAP EXPERIMENT ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def state_swap_experiment(model, video_A, video_B, swap_step):
    """
    Hypothesis: "If you swap the state mid-generation, the model collapses."

    Setup
    -----
    - Run the first `swap_step` steps using video A's states.
    - At t = swap_step, replace z_t with Encoder(video_B[swap_step]).
    - Continue rolling out with the GCM actions (no ground-truth).

    Returns
    -------
    frames_normal : (T-1, H, W, C)  — baseline rollout from video A only
    frames_swapped: (T-1, H, W, C)  — rollout where state is swapped at t=swap_step
    frames_B      : (T-1, H, W, C)  — baseline rollout from video B only (oracle)
    swap_mse      : float            — reconstruction MSE after swap (vs video A gt)
    normal_mse    : float            — baseline MSE (vs video A gt)
    """
    T_loc = video_A.shape[0]
    z_A   = jax.vmap(model.encode)(video_A)   # (T, d_z)
    z_B   = jax.vmap(model.encode)(video_B)
    u_A   = jax.vmap(model.extract_action)(z_A[:-1], z_A[1:])

    # ── Normal rollout (video A) ─────────────────────────────────
    gcm_h  = model.gcm.initial_state()
    u_prev = jnp.zeros(model.d_u)
    z_t    = z_A[0]
    frames_normal = []
    z_hist_n, u_hist_n = [z_t], []

    for t in range(T_loc - 1):
        gcm_h, u_t = model.predict_action_gcm(gcm_h, z_t, u_prev)
        u_prev = u_t
        u_hist_n.append(u_t)
        z_seq_arr = jnp.stack(z_hist_n)
        u_seq_arr = jnp.stack(u_hist_n)
        z_next    = model.predict_next_latent(z_seq_arr, u_seq_arr)[-1]
        frames_normal.append(model.decode(z_next, video_A.shape[1:]))
        z_hist_n.append(z_next)
        z_t = z_next

    # ── Swapped rollout ─────────────────────────────────────────
    gcm_h  = model.gcm.initial_state()
    u_prev = jnp.zeros(model.d_u)
    z_t    = z_A[0]
    frames_swapped = []
    z_hist_s, u_hist_s = [z_t], []
    swap_done = False

    for t in range(T_loc - 1):
        if t == swap_step and not swap_done:
            # *** SWAP: replace current state with video B's encoded frame ***
            z_t = z_B[swap_step]
            swap_done = True

        gcm_h, u_t = model.predict_action_gcm(gcm_h, z_t, u_prev)
        u_prev = u_t
        u_hist_s.append(u_t)
        z_seq_arr = jnp.stack(z_hist_s)
        u_seq_arr = jnp.stack(u_hist_s)
        z_next    = model.predict_next_latent(z_seq_arr, u_seq_arr)[-1]
        frames_swapped.append(model.decode(z_next, video_A.shape[1:]))
        z_hist_s.append(z_next)
        z_t = z_next

    # ── Video-B oracle rollout ────────────────────────────────────
    gcm_h  = model.gcm.initial_state()
    u_prev = jnp.zeros(model.d_u)
    z_t    = z_B[0]
    frames_B = []
    z_hist_b, u_hist_b = [z_t], []
    for t in range(T_loc - 1):
        gcm_h, u_t = model.predict_action_gcm(gcm_h, z_t, u_prev)
        u_prev = u_t
        u_hist_b.append(u_t)
        z_seq_arr = jnp.stack(z_hist_b)
        u_seq_arr = jnp.stack(u_hist_b)
        z_next    = model.predict_next_latent(z_seq_arr, u_seq_arr)[-1]
        frames_B.append(model.decode(z_next, video_B.shape[1:]))
        z_hist_b.append(z_next)
        z_t = z_next

    fn = jnp.stack(frames_normal)
    fs = jnp.stack(frames_swapped)
    fb = jnp.stack(frames_B)

    gt_A = video_A[1:]
    normal_mse = float(jnp.mean((fn - gt_A) ** 2))
    swap_mse   = float(jnp.mean((fs - gt_A) ** 2))

    return fn, fs, fb, normal_mse, swap_mse


# ─────────────────────────────────────────────────────────────────
# ── VISUALISATION HELPERS ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────

def _row_strip(frames, H, W, C, n_frames=None):
    """Build a horizontal strip of frames for a single row."""
    if n_frames is not None:
        frames = frames[:n_frames]
    arr = np.array(frames)                         # (T, H, W, C)
    strip = arr.reshape(arr.shape[0] * W, H, C).transpose(1, 0, 2)
    return strip[..., 0] if C == 1 else strip      # squeeze channel if grey


def visualise_rollout(model, test_loader, context_steps, n_seqs, fname_prefix="plots/p3"):
    """
    Visualise context-conditioned rollouts using plot_videos from utils.
    """
    batch   = jnp.array(next(iter(test_loader)))
    p3_cfg  = CFG["phase_3"]
    c_steps = p3_cfg["context_steps"]
    n_seqs  = min(n_seqs, batch.shape[0])

    for i in range(n_seqs):
        video = batch[i]                              # (T, H, W, C)
        pred  = rollout_context(model, video, c_steps)

        gt_vis   = np.array(video[1:])    # (T-1, H, W, C)
        pred_vis = np.array(pred)         # (T-1, H, W, C)

        plot_videos(
            pred_vis,
            ref_video=gt_vis,
            plot_ref=True,
            forecast_start=c_steps + 1,
            show_titles=True,
            save_name=f"{fname_prefix}_rollout_seq{i:02d}.png",
            save_video=True)
    print(f"  Saved {n_seqs} rollout plots to {fname_prefix}_rollout_*.png")


def visualise_state_swap(model, test_loader, swap_step,
                         n_pairs=3, fname="plots/p3_state_swap.png"):
    """
    Visualise the state-swap experiment for multiple pairs.
    Shows 4 rows per pair:
        Row 0: GT video A
        Row 1: Normal rollout (video A)
        Row 2: Swapped rollout (video A states → video B state at t=swap)
        Row 3: GT video B
    """
    batch    = jnp.array(next(iter(test_loader)))
    n_pairs  = min(n_pairs, batch.shape[0] // 2)
    T_frames = T - 1

    mse_normal_list = []
    mse_swapped_list = []
    row_labels = []

    row_data = []
    for idx in range(n_pairs):
        vA = batch[idx]
        vB = batch[idx + n_pairs]

        fn, fs, fb, mse_n, mse_s = state_swap_experiment(
            model, vA, vB, swap_step)

        mse_normal_list.append(mse_n)
        mse_swapped_list.append(mse_s)
        row_data.append((vA, fn, fs, vB))
        print(f"  Pair {idx+1}: normal MSE={mse_n:.4f}  "
              f"swap MSE={mse_s:.4f}  "
              f"(ratio={mse_s/max(mse_n,1e-8):.2f}×)", flush=True)

    # Plot grid
    n_rows = n_pairs * 4
    fig, axes = plt.subplots(n_rows, T_frames,
                              figsize=(1.8 * T_frames, 1.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    row_names = ["GT A", "Normal rollout A", "Swapped rollout", "GT B"]
    colours   = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]

    for p, (vA, fn, fs, vB) in enumerate(row_data):
        base = p * 4
        seqs = [np.array(vA[1:]), np.array(fn), np.array(fs), np.array(vB[1:])]
        for r, (seq, name, col) in enumerate(zip(seqs, row_names, colours)):
            for t in range(T_frames):
                ax = axes[base + r, t]
                frame = seq[t]
                im_kwargs = {"vmin": 0, "vmax": 1}
                if frame.shape[-1] == 1:
                    ax.imshow(frame[..., 0], cmap="gray", **im_kwargs)
                else:
                    ax.imshow(np.clip(frame, 0, 1))
                ax.axis("off")
                if t == 0:
                    ax.set_ylabel(name, rotation=0, labelpad=60,
                                  ha="right", va="center",
                                  color=col, fontsize=8, fontweight="bold")
                if base + r == 0:
                    ax.set_title(
                        f"t={t+1}" + (" ⬅ SWAP" if t == swap_step else ""),
                        fontsize=7,
                        color="#F44336" if t == swap_step else "black")

    mse_avg_n = np.mean(mse_normal_list)
    mse_avg_s = np.mean(mse_swapped_list)
    fig.suptitle(
        f"State-Swap Experiment  (swap at t={swap_step+1})\n"
        f"Avg MSE — Normal: {mse_avg_n:.4f}   Swapped: {mse_avg_s:.4f}   "
        f"Ratio: {mse_avg_s/max(mse_avg_n,1e-8):.2f}×",
        fontsize=11, y=1.01)

    plt.tight_layout()
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved state-swap visualisation → {fname}")
    return mse_normal_list, mse_swapped_list


def visualise_action_space(model, test_loader,
                            fname="plots/p3_action_space.png"):
    """2-D PCA projection of IDM-extracted actions."""
    from sklearn.decomposition import PCA   # soft dep

    all_u = []
    for batch in test_loader:
        batch = jnp.array(batch)
        z_seq = jax.vmap(jax.vmap(model.encode))(batch)   # (B, T, d_z)
        u_seq = jax.vmap(
            lambda zs: jax.vmap(model.extract_action)(zs[:-1], zs[1:])
        )(z_seq)                                            # (B, T-1, d_u)
        all_u.append(np.array(u_seq.reshape(-1, model.d_u)))

    all_u = np.concatenate(all_u, axis=0)

    if all_u.shape[1] >= 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(all_u)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(emb[:, 0], emb[:, 1],
                        c=np.arange(len(emb)), cmap="viridis",
                        alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax, label="sample index")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title("IDM Latent Action Space (PCA)")
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved {fname}")
    else:
        print("  Skipping action-space PCA (d_u < 2)")


# ─────────────────────────────────────────────────────────────────
# Run Inference
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3 — Inference & Experiments")
print("=" * 60)

p3 = CFG["phase_3"]
n_vis = p3.get("num_vis_sequences", 5)
ctx   = p3.get("context_steps", 10)
swap  = ctx // 2    # swap happens halfway through the context phase

# 1. Context-conditioned rollouts
print(f"\n[1] Context-conditioned rollouts (context={ctx})")
visualise_rollout(model, test_loader, ctx, n_vis, fname_prefix="plots/p3")

# 2. State-swap experiment
print(f"\n[2] State-swap experiment (swap at step {swap})")
mse_n, mse_s = visualise_state_swap(
    model, test_loader, swap_step=swap, n_pairs=3,
    fname="plots/p3_state_swap.png")

# 3. Action space visualisation
print("\n[3] Action-space PCA")
try:
    visualise_action_space(model, test_loader)
except ImportError:
    print("  sklearn not available, skipping PCA plot.")

# 4. Summary bar chart: normal vs swapped MSE
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(mse_n))
w = 0.35
ax.bar(x - w/2, mse_n, w, label="Normal rollout",  color="#4CAF50", alpha=0.8)
ax.bar(x + w/2, mse_s, w, label="Swapped rollout", color="#F44336", alpha=0.8)
ax.axhline(np.mean(mse_n), color="#4CAF50", linestyle="--", linewidth=1.5,
           label=f"Mean normal  ({np.mean(mse_n):.4f})")
ax.axhline(np.mean(mse_s), color="#F44336", linestyle="--", linewidth=1.5,
           label=f"Mean swapped ({np.mean(mse_s):.4f})")
ax.set_xlabel("Sequence pair")
ax.set_ylabel("MSE (pixel space)")
ax.set_title("State-Swap Experiment — Reconstruction Error")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/p3_swap_mse_bar.png", dpi=120)
plt.close()
print("  Saved plots/p3_swap_mse_bar.png")

print("\n✅  Phase 3 complete.")