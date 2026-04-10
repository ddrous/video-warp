#%%
"""
phase3.py — Genie Phase 3: GCM Training + Inference + State-Swap
=================================================================

Trains the Generative Control Model (GCM), a GRU that autoregressively
predicts the next *discrete* action token from the current frame token
and the previous action token.  All other components are frozen.

GCM loss:
    L_gcm = (1/T-1) Σ_t  CrossEntropy( GCM(z_t, a_{t-1}),  a_t )

After training the GCM, three inference modes are available:
  1. Context-conditioned rollout (ρ·T steps IDM, rest GCM)
  2. Fully autonomous rollout (ρ = 0)
  3. State-swap experiment (latent z injected from a different video
     mid-rollout to probe state-action entanglement)

State-swap hypothesis
---------------------
In Genie-Discrete the Dynamics Transformer sees the full context of
(z_idx, a_idx) pairs.  When z_{t*} is replaced with a token from a
different video, the Transformer's attention context becomes incoherent:
it was trained on smooth trajectories, and the sudden token change
breaks its implicit temporal model.  We expect a measurable spike in
reconstruction MSE at and after the swap point.

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
train_loader, test_loader = get_dataloaders(CFG, phase="phase_3")
sample = next(iter(train_loader))
B, T, H, W, C = sample.shape
print(f"Phase-3 batch: B={B} T={T} H={H} W={W} C={C}")

#%%
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
    print("⚠️   No Phase-2 checkpoint; using random init")

print(f"GCM params: {count_trainable_params(model.gcm):,}\n")

#%%
# ─────────────────────────────────────────────────────────────────
# Phase-3 Forward — GCM imitation of IDM (teacher forcing)
# ─────────────────────────────────────────────────────────────────
def p3_forward(model, video):
    """
    video : (T, H, W, C)
    GCM is trained to predict a_t = IDM(z_t, z_{t+1}) from (z_t, a_{t-1}).
    Returns (gcm_loss, gcm_action_logits (T-1, action_K), idm_action_idx (T-1,))
    """
    T_loc = video.shape[0]

    # Encode all frames (frozen)
    def enc_one(frame):
        z_idx, z_q_st, _, _ = model.encode_frame(frame)
        return jax.lax.stop_gradient(z_idx), jax.lax.stop_gradient(z_q_st)

    z_idx_all, z_q_all = jax.vmap(enc_one)(video)   # (T,), (T, d_vq)

    # IDM target actions (frozen)
    def idm_one(z_q_t, z_q_tp1):
        a_idx, _, _ = jax.lax.stop_gradient(
            model.extract_action(z_q_t, z_q_tp1))
        return a_idx
    a_idx_target = jax.lax.stop_gradient(
        jax.vmap(idm_one)(z_q_all[:-1], z_q_all[1:]))  # (T-1,)

    # GCM rollout with teacher forcing
    h0     = model.gcm.initial_state()
    a_prev = jnp.array(0, dtype=jnp.int32)   # start-of-sequence action = 0

    def scan_fn(carry, inputs):
        h, a_prev_idx = carry
        z_idx_t, a_idx_t = inputs
        h_new, logits = model.gcm.step(h, z_idx_t, a_prev_idx)
        step_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[None], a_idx_t[None]).squeeze()
        # Teacher forcing: next a_prev is the IDM target
        return (h_new, a_idx_t), (logits, step_loss)

    _, (all_logits, step_losses) = jax.lax.scan(
        scan_fn, (h0, a_prev),
        (z_idx_all[:-1], a_idx_target))

    gcm_loss = jnp.mean(step_losses)
    return gcm_loss, all_logits, a_idx_target

#%%
# ─────────────────────────────────────────────────────────────────
# Optimiser — GCM only
# ─────────────────────────────────────────────────────────────────
p3_cfg = CFG["phase_3"]
total_steps = p3_cfg["nb_epochs"] * len(train_loader)
schedule    = optax.cosine_decay_schedule(p3_cfg["learning_rate"], total_steps)
optimiser   = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

all_false   = jax.tree_util.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda m: m.gcm, all_false,
    replace=jax.tree_util.tree_map(eqx.is_inexact_array, model.gcm))

frozen    = eqx.filter(model, jax.tree_util.tree_map(lambda x: not x, filter_spec))
opt_state = optimiser.init(eqx.filter(model, filter_spec))

@eqx.filter_jit
def train_step(model, batch):
    def loss_fn(trainable):
        m = eqx.combine(trainable, frozen)
        losses, _, _ = jax.vmap(p3_forward, in_axes=(None, 0))(m, batch)
        return jnp.mean(losses)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(eqx.filter(model, filter_spec))
    upd, new_opt = optimiser.update(grads, opt_state, eqx.filter(model, filter_spec))
    return eqx.combine(
        optax.apply_updates(eqx.filter(model, filter_spec), upd), frozen), \
        new_opt, loss

#%%
# ─────────────────────────────────────────────────────────────────
# Inference: context-conditioned + autonomous rollout
# ─────────────────────────────────────────────────────────────────
def rollout(model, video, context_steps: int):
    """
    Run a context-conditioned rollout.
    For t < context_steps: action from IDM (oracle).
    For t >= context_steps: action from GCM (autonomous).

    Returns predicted frames (T-1, H, W, C).
    """
    T_loc = video.shape[0]

    # Encode all frames for context
    z_idx_gt = jax.vmap(lambda f: model.encode_frame(f)[0])(video)   # (T,)
    z_q_gt   = jax.vmap(lambda f: model.encode_frame(f)[1])(video)   # (T, d_vq)
    a_idx_gt = jax.vmap(model.extract_action)(
        z_q_gt[:-1], z_q_gt[1:])[0]  # (T-1,) IDM actions

    h_gcm  = model.gcm.initial_state()
    a_prev = jnp.array(0, dtype=jnp.int32)
    z_idx_cur = z_idx_gt[0]

    # Warm-up GCM on context steps
    for t in range(min(context_steps, T_loc - 1)):
        h_gcm, _ = model.gcm.step(h_gcm, z_idx_gt[t], a_prev)
        a_prev   = a_idx_gt[t]
        z_idx_cur = z_idx_gt[t + 1]

    # Autoregressive rollout
    z_history = list(z_idx_gt[:context_steps + 1])
    a_history = list(a_idx_gt[:context_steps])
    pred_frames = []

    for t in range(context_steps, T_loc - 1):
        # GCM predicts next action
        h_gcm, logits = model.gcm.step(h_gcm, z_idx_cur, a_prev)
        a_t = jnp.argmax(logits).astype(jnp.int32)
        a_prev = a_t
        a_history.append(a_t)

        # Dynamics Transformer predicts next token
        z_seq_arr = jnp.array(z_history)                  # (t+1,)
        a_seq_arr = jnp.array(a_history[:len(z_history)]) # (t+1,)
        dyn_logits = model.dynamics(z_seq_arr, a_seq_arr)  # (t+1, frame_K)
        z_next = jnp.argmax(dyn_logits[-1]).astype(jnp.int32)

        frame = model.decode_from_index(z_next)
        pred_frames.append(frame)
        z_history.append(z_next)
        z_idx_cur = z_next

    return jnp.stack(pred_frames)   # (rollout_steps, H, W, C)


#%%
# ─────────────────────────────────────────────────────────────────
# State-Swap Experiment
# ─────────────────────────────────────────────────────────────────
def state_swap_experiment(model, video_A, video_B, swap_step: int):
    """
    Hypothesis: swapping z_t mid-rollout breaks Genie-Discrete because
    the Transformer's full causal context becomes incoherent.

    Protocol
    --------
    At t = swap_step, replace z_idx_cur with Encoder(video_B[swap_step]).
    Continue with GCM actions and Dynamics Transformer prediction.
    Measure reconstruction MSE vs video_A ground-truth.

    Returns
    -------
    frames_normal  : (T-1, H, W, C)  — baseline from video A
    frames_swapped : (T-1, H, W, C)  — state replaced at swap_step
    frames_B       : (T-1, H, W, C)  — oracle from video B
    mse_normal     : float
    mse_swapped    : float
    per_step_mse_normal  : (T-1,)
    per_step_mse_swapped : (T-1,)
    """
    T_loc = video_A.shape[0]
    ctx_steps = min(swap_step, p3_cfg["context_steps"])

    def _single_rollout(video_ctx, video_inject, inject_step):
        """Generic rollout with optional state injection."""
        z_idx_ctx = jax.vmap(lambda f: model.encode_frame(f)[0])(video_ctx)
        z_q_ctx   = jax.vmap(lambda f: model.encode_frame(f)[1])(video_ctx)
        a_idx_ctx = jax.vmap(model.extract_action)(
            z_q_ctx[:-1], z_q_ctx[1:])[0]

        h = model.gcm.initial_state()
        a_prev = jnp.array(0, dtype=jnp.int32)
        z_idx_cur = z_idx_ctx[0]

        # Context warm-up (IDM)
        for t in range(ctx_steps):
            h, _ = model.gcm.step(h, z_idx_ctx[t], a_prev)
            a_prev    = a_idx_ctx[t]
            z_idx_cur = z_idx_ctx[t + 1]

        z_hist, a_hist = list(z_idx_ctx[:ctx_steps+1]), list(a_idx_ctx[:ctx_steps])
        pred_frames = []

        for t in range(ctx_steps, T_loc - 1):
            if t == inject_step and video_inject is not None:
                # *** STATE SWAP ***
                z_idx_cur = jax.lax.stop_gradient(
                    model.encode_frame(video_inject[t])[0])

            h, logits = model.gcm.step(h, z_idx_cur, a_prev)
            a_t = jnp.argmax(logits).astype(jnp.int32)
            a_prev = a_t
            a_hist.append(a_t)

            z_arr = jnp.array(z_hist)
            a_arr = jnp.array(a_hist[:len(z_hist)])
            dyn_logits = model.dynamics(z_arr, a_arr)
            z_next = jnp.argmax(dyn_logits[-1]).astype(jnp.int32)

            pred_frames.append(model.decode_from_index(z_next))
            z_hist.append(z_next)
            z_idx_cur = z_next

        return jnp.stack(pred_frames)

    frames_N = _single_rollout(video_A, None,      swap_step)   # normal
    frames_S = _single_rollout(video_A, video_B,   swap_step)   # swapped
    frames_B = _single_rollout(video_B, None,      swap_step)   # oracle B

    gt_A = video_A[ctx_steps + 1 : ctx_steps + 1 + frames_N.shape[0]]
    ps_normal  = jnp.array([float(jnp.mean((frames_N[i] - gt_A[i])**2))
                             for i in range(len(frames_N))])
    ps_swapped = jnp.array([float(jnp.mean((frames_S[i] - gt_A[i])**2))
                             for i in range(len(frames_S))])

    return (frames_N, frames_S, frames_B,
            float(jnp.mean(ps_normal)), float(jnp.mean(ps_swapped)),
            np.array(ps_normal), np.array(ps_swapped))


#%%
# ─────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────
def _prep(f):
    f = np.clip(np.array(f), 0, 1)
    return ("gray", f[..., 0]) if f.shape[-1] == 1 else (None, f)

def vis_gcm_loss(xs, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(xs, losses, lw=2, color="#9C27B0")
    ax1.set_xlabel("Train step"); ax1.set_ylabel("CE Loss")
    ax1.set_title("Phase 3 — GCM Loss per step"); ax1.grid(alpha=0.3)
    ax2.semilogy(xs, losses, lw=2, color="#9C27B0")
    ax2.set_xlabel("Train step"); ax2.set_ylabel("CE Loss (log)")
    ax2.set_title("Phase 3 — GCM Loss (log)"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/p3_gcm_loss.png", dpi=120, bbox_inches="tight"); plt.close(fig)

def vis_rollout_strip(gt_video, pred_frames, step, ctx_steps,
                      seq_idx=0, fname=None):
    """Two-row strip: GT (top) vs Predicted (bottom)."""
    T_show = min(T - 1, len(pred_frames))
    fig, axes = plt.subplots(2, T_show, figsize=(1.9 * T_show, 4))
    for t in range(T_show):
        for row, arr in enumerate([gt_video[t+1], pred_frames[t]]):
            cmap, img = _prep(arr)
            axes[row, t].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, t].axis("off")
            if t == 0:
                axes[row, t].set_ylabel(["GT","Pred"][row], rotation=0,
                                         labelpad=25, fontsize=11, fontweight="bold")
        border_col = "#F44336" if t + 1 == ctx_steps else "white"
        for r in range(2):
            for spine in axes[r, t].spines.values():
                spine.set_edgecolor(border_col); spine.set_linewidth(2.5)
                spine.set_visible(True)
        axes[0, t].set_title(f"t={t+2}"
            + (" ⬅ctx" if t + 1 == ctx_steps else ""), fontsize=8)
    fig.suptitle(f"Phase 3 Rollout  seq={seq_idx}  step={step:,}  "
                 f"(red = context boundary)", fontsize=11)
    plt.tight_layout()
    fname = fname or f"plots/p3_rollout_seq{seq_idx}_step{step:06d}.png"
    plt.savefig(fname, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  [vis] {fname}", flush=True)

def vis_state_swap(video_A, video_B, frames_N, frames_S, frames_B,
                   swap_step, ps_n, ps_s, pair_idx, step):
    """Four-row grid: GT A, Normal, Swapped, GT B."""
    n_show = min(T - 1, len(frames_N))
    fig, axes = plt.subplots(4, n_show, figsize=(1.8 * n_show, 7.5))
    rows = [
        ("GT A",    video_A[1:n_show+1], "#2196F3"),
        ("Normal",  frames_N[:n_show],   "#4CAF50"),
        ("Swapped", frames_S[:n_show],   "#F44336"),
        ("GT B",    video_B[1:n_show+1], "#FF9800"),
    ]
    for r, (label, frames, col) in enumerate(rows):
        for t in range(n_show):
            cmap, img = _prep(frames[t])
            axes[r, t].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[r, t].axis("off")
            if t == 0:
                axes[r, t].set_ylabel(label, rotation=0, labelpad=28,
                                       color=col, fontsize=10, fontweight="bold")
            if t == swap_step - p3_cfg.get("context_steps", 10):
                for sp in axes[r, t].spines.values():
                    sp.set_edgecolor("#F44336"); sp.set_linewidth(3); sp.set_visible(True)
    for t in range(n_show):
        axes[0, t].set_title(
            f"t={t+2}" + (" ⬅SWAP" if t == swap_step - p3_cfg.get("context_steps",10) else ""),
            fontsize=7,
            color="#F44336" if t == swap_step - p3_cfg.get("context_steps",10) else "k")

    fig.suptitle(f"State-Swap Experiment  pair={pair_idx}  swap@t={swap_step+1}\n"
                 f"MSE — Normal: {ps_n:.4f}   Swapped: {ps_s:.4f}   "
                 f"Ratio: {ps_s/max(ps_n,1e-8):.2f}×",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    fname = f"plots/p3_swap_pair{pair_idx}_step{step:06d}.png"
    plt.savefig(fname, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  [vis] {fname}", flush=True)

def vis_per_step_mse(ps_normal_all, ps_swapped_all, swap_step, step):
    """Mean ± std of per-step MSE for normal vs swapped rollouts."""
    ps_n = np.array(ps_normal_all)    # (n_pairs, rollout_len)
    ps_s = np.array(ps_swapped_all)
    n_steps = ps_n.shape[1]
    ts = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, ps_n.mean(0), label="Normal",  lw=2, color="#4CAF50")
    ax.fill_between(ts, ps_n.mean(0) - ps_n.std(0),
                        ps_n.mean(0) + ps_n.std(0),
                    alpha=0.2, color="#4CAF50")
    ax.plot(ts, ps_s.mean(0), label="Swapped", lw=2, color="#F44336")
    ax.fill_between(ts, ps_s.mean(0) - ps_s.std(0),
                        ps_s.mean(0) + ps_s.std(0),
                    alpha=0.2, color="#F44336")
    swap_rel = swap_step - p3_cfg.get("context_steps", 10)
    if 0 <= swap_rel < n_steps:
        ax.axvline(swap_rel, color="#F44336", linestyle="--", linewidth=2,
                   label=f"Swap at step {swap_rel}")
    ax.set_xlabel("Rollout step (post-context)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"Per-step MSE — Normal vs Swapped (step {step:,})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/p3_swap_per_step_mse_step{step:06d}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

def vis_swap_summary(mse_ns, mse_ss, step):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(mse_ns))
    w = 0.35
    ax.bar(x - w/2, mse_ns, w, label="Normal",  color="#4CAF50", alpha=0.8)
    ax.bar(x + w/2, mse_ss, w, label="Swapped", color="#F44336", alpha=0.8)
    ax.axhline(np.mean(mse_ns), color="#4CAF50", linestyle="--", lw=1.5)
    ax.axhline(np.mean(mse_ss), color="#F44336", linestyle="--", lw=1.5)
    ax.set_xlabel("Sequence pair"); ax.set_ylabel("Mean MSE")
    ax.set_title("State-Swap — Reconstruction MSE per pair")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/p3_swap_summary_step{step:06d}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Swap summary  →  "
          f"Normal MSE={np.mean(mse_ns):.4f}  "
          f"Swapped MSE={np.mean(mse_ss):.4f}  "
          f"Ratio={np.mean(mse_ss)/max(np.mean(mse_ns),1e-8):.2f}×")

#%%
# ─────────────────────────────────────────────────────────────────
# Training loop — per step
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 3: GCM Training")
print(f"  GCM predicts discrete action tokens a_t ∈ {{0..{model.action_K-1}}}")
print(f"  Epochs={p3_cfg['nb_epochs']}  Steps/epoch={len(train_loader)}")
print("=" * 65)

xs_log, losses_log = [], []
gstep = 0
vis_batch = jnp.array(next(iter(test_loader))[:p3_cfg.get("num_vis_sequences", 5)])

for epoch in range(1, p3_cfg["nb_epochs"] + 1):
    for batch in train_loader:
        batch = jnp.array(batch)
        model, opt_state, loss = train_step(model, batch)
        xs_log.append(gstep); losses_log.append(float(loss)); gstep += 1

        if gstep % p3_cfg["print_every"] == 0:
            print(f"  step {gstep:6d} | ep {epoch:3d} | "
                  f"GCM_CE={float(loss):.4f}", flush=True)

        if gstep % p3_cfg["vis_every"] == 0:
            vis_gcm_loss(xs_log, losses_log)
            ctx = p3_cfg["context_steps"]
            for si in range(min(3, vis_batch.shape[0])):
                pred = rollout(model, vis_batch[si], ctx)
                vis_rollout_strip(np.array(vis_batch[si]), np.array(pred),
                                   gstep, ctx, si)

#%%
# ─────────────────────────────────────────────────────────────────
# Final Inference
# ─────────────────────────────────────────────────────────────────
vis_gcm_loss(xs_log, losses_log)
eqx.tree_serialise_leaves(ARTIFACTS / "genie_phase3.eqx", model)
print(f"✅  Saved → {ARTIFACTS}/genie_phase3.eqx")

print("\n" + "=" * 65)
print("Phase 3 — Inference & State-Swap Experiment")
print("=" * 65)

ctx       = p3_cfg["context_steps"]
swap_step = p3_cfg.get("swap_step", ctx // 2)
n_pairs   = p3_cfg.get("num_swap_pairs", 4)

# ── Context-conditioned rollouts ─────────────────────────────────
print(f"\n[1] Context-conditioned rollouts (context={ctx} steps)")
test_batch = jnp.array(next(iter(test_loader)))
for si in range(min(p3_cfg.get("num_vis_sequences", 5), test_batch.shape[0])):
    pred = rollout(model, test_batch[si], ctx)
    vis_rollout_strip(np.array(test_batch[si]), np.array(pred),
                       gstep, ctx, si, fname=f"plots/p3_final_rollout_seq{si}.png")
    # Also use plot_videos for publication-quality strip
    gt_strip   = np.array(test_batch[si][1:])
    pred_strip = np.array(pred)
    plot_videos(pred_strip, ref_video=gt_strip, plot_ref=True,
                forecast_start=ctx + 1,
                cmap="gray" if C == 1 else "viridis",
                save_name=f"plots/p3_plotvideos_seq{si}.png",
                save_video=True)

# ── State-swap experiment ─────────────────────────────────────────
print(f"\n[2] State-swap experiment  (swap at t={swap_step+1})")
mse_ns, mse_ss = [], []
ps_ns_all, ps_ss_all = [], []

test_iter = iter(test_loader)
batch_A = jnp.array(next(test_iter))
batch_B = jnp.array(next(test_iter))
n_pairs = min(n_pairs, batch_A.shape[0], batch_B.shape[0])

for pi in range(n_pairs):
    vA, vB = batch_A[pi], batch_B[pi]
    fN, fS, fB, m_n, m_s, ps_n, ps_s = state_swap_experiment(
        model, vA, vB, swap_step)
    mse_ns.append(m_n); mse_ss.append(m_s)
    ps_ns_all.append(ps_n); ps_ss_all.append(ps_s)
    vis_state_swap(np.array(vA), np.array(vB),
                   np.array(fN), np.array(fS), np.array(fB),
                   swap_step, m_n, m_s, pi, gstep)
    print(f"  Pair {pi+1}: normal MSE={m_n:.4f}  swapped MSE={m_s:.4f}  "
          f"ratio={m_s/max(m_n,1e-8):.2f}×", flush=True)

vis_swap_summary(mse_ns, mse_ss, gstep)
vis_per_step_mse(ps_ns_all, ps_ss_all, swap_step, gstep)

print("\n✅  Phase 3 complete.")