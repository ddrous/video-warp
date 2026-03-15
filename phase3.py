#%% Imports and Setup
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from utils import setup_run_dir, get_coords_grid, plot_videos, count_trainable_params
from loaders import get_dataloaders
from models import VWARP

try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    raise Exception(f"Error: Could not load config.yaml. ({e})")

TRAIN = True
DEBUG = CONFIG.get("debug", False)

#%% Phase 3 Forward Definition
def phase3_forward(model, ref_video):
    """ Forward pass for Phase 3: Freezes IDM/FDM and trains GCM to mimic IDM. """
    T = ref_video.shape[0]
    init_frame = ref_video[0]

    # Stop gradients for the frozen parts
    z_init = jax.lax.stop_gradient(model.encoder(jnp.transpose(init_frame, (2, 0, 1))))
    m_init = model.action_model.reset_gcm(T)
    a_init = jnp.zeros((model.lam_dim,))

    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t, a_tm1 = carry
        o_tp1, step_idx = scan_inputs

        z_tp1_enc = jax.lax.stop_gradient(model.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
        
        # Ground truth action from IDM (frozen)
        raw_a_target, a_target = jax.lax.stop_gradient(model.action_model.inverse_dynamics(z_t, z_tp1_enc))

        # Predicted action from GCM
        raw_a_pred, quant_a_pred = model.action_model.decode_gcm(m_t, step_idx, jax.lax.stop_gradient(z_t))

        if getattr(model, "use_action_residuals", False):
            raw_a_target = raw_a_target + a_tm1
            a_target = a_target + a_tm1
            raw_a_pred = raw_a_pred + a_tm1
            quant_a_pred = quant_a_pred + a_tm1

        # Update memory buffer using target action (Teacher Forcing)
        m_tp1 = model.action_model.encode_gcm(m_t, step_idx, jax.lax.stop_gradient(z_t), a_target)

        # Step dynamics (frozen)
        z_tp1 = jax.lax.stop_gradient(model.transition_model(z_t, a_target))

        return (z_tp1, m_tp1, a_target), (raw_a_pred, a_target)

    scan_inputs = (ref_video[1:], jnp.arange(1, T))
    _, (actions_preds, actions_targets) = jax.lax.scan(scan_step, (z_init, m_init, a_init), scan_inputs)

    return actions_preds, actions_targets

#%% Model & Optimiser Instantiation
key = jax.random.PRNGKey(CONFIG["seed"])
run_dir = setup_run_dir("phase_3", CONFIG, train=TRAIN)

train_loader, test_loader = get_dataloaders(CONFIG, phase="phase_3")
sample_batch = next(iter(train_loader))
B, T, H, W, C = sample_batch.shape
coords_grid = get_coords_grid(H, W)

key, subkey = jax.random.split(key)
model = VWARP(CONFIG, frame_shape=(H, W, C), key=subkey, init_gcm=True)

try:
    dummy = VWARP(CONFIG, frame_shape=(H, W, C), key=subkey, init_gcm=False)
    dummy = eqx.tree_deserialise_leaves("vwarp_phase2.eqx", dummy)
    
    model = eqx.tree_at(lambda m: m.encoder, model, dummy.encoder)
    model = eqx.tree_at(lambda m: m.transition_model, model, dummy.transition_model)
    model = eqx.tree_at(lambda m: m.action_model.idm, model, dummy.action_model.idm)
    print("✅ Transplanted Dynamics & Encoder weights from Phase 2")
except Exception as e:
    print(f"⚠️ Failed to load vwarp_phase2.eqx: {e}")

if not TRAIN:
    print("⏭️  Skipping Phase 3 training...")
    model = eqx.tree_deserialise_leaves("vwarp_phase3.eqx", model)

# Freeze everything EXCEPT GCM
filter_spec = jax.tree_util.tree_map(lambda _: False, model)
gcm_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model.action_model.gcm)
filter_spec = eqx.tree_at(lambda m: m.action_model.gcm, filter_spec, gcm_mask)

diff_model, static_model = eqx.partition(model, filter_spec)

optimizer = optax.chain(
    optax.adam(CONFIG["phase_3"]["learning_rate"]),
    optax.contrib.reduce_on_plateau(
        patience=CONFIG["phase_3"]["lr_patience"], cooldown=CONFIG["phase_3"]["lr_cooldown"],
        factor=CONFIG["phase_3"]["lr_factor"], rtol=CONFIG["phase_3"]["lr_rtol"],
        accumulation_size=CONFIG["phase_3"]["lr_accum_size"], min_scale=CONFIG["phase_3"]["lr_min_scale"]
    )
)
opt_state = optimizer.init(diff_model)

@eqx.filter_jit
def train_step(diff_m, static_m, opt_state, ref_videos):
    def loss_fn(d_model):
        m = eqx.combine(d_model, static_m)
        batched_fn = jax.vmap(phase3_forward, in_axes=(None, 0))
        a_preds, a_targets = batched_fn(m, ref_videos)
        
        if CONFIG["phase_3"]["loss_type"] == "L1":
            return jnp.mean(jnp.abs(a_preds - a_targets))
        else: # Default L2
            return jnp.mean((a_preds - a_targets)**2)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(diff_m)
    updates, opt_state = optimizer.update(grads, opt_state, diff_m, value=loss_val)
    diff_m = eqx.apply_updates(diff_m, updates)
    return diff_m, opt_state, loss_val

@eqx.filter_jit
def eval_step(m, ref_videos, context_ratio=0.0):
    # batched_fn = jax.vmap(phase3_forward, in_axes=(None, 0))
    # a_preds, a_targets = batched_fn(m, ref_videos)
    # return a_preds, a_targets

    _, _, pred_videos = m(ref_videos, coords_grid, context_ratio)
    return pred_videos

#%% Training Loop
if TRAIN:
    print(f"\n🚀 Starting Phase 3: GCM Action Matching -> Saving to {run_dir}")
    print(f"Trainable Params (GCM only): {count_trainable_params(diff_model)}")

    start_time = time.time()
    for epoch in range(CONFIG["phase_3"]["nb_epochs"]):
        epoch_losses = []
        lr_scales = []
        for batch_videos in train_loader:
            diff_model, opt_state, loss = train_step(diff_model, static_model, opt_state, batch_videos)
            epoch_losses.append(loss)

            lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(lr_scale)

        if (epoch+1) % CONFIG["phase_3"]["print_every"] == 0:
            print(f"Phase 3 - Epoch {epoch+1}/{CONFIG["phase_3"]['nb_epochs']} - Avg Loss: {np.mean(epoch_losses):.6f}", f"- LR Scale: {lr_scale:.4f}", flush=True)

        ## Plot video every 10ths of epochs
        if (epoch+1) % max(1, CONFIG["phase_3"]["nb_epochs"] // 10) == 0:
            pred_videos = eval_step(eqx.combine(diff_model, static_model), sample_batch, context_ratio=0.0)
            plot_videos(
                video=pred_videos[0], 
                ref_video=sample_batch[0], 
                plot_ref=True, 
                show_titles=True,
                save_name=run_dir / "plots" / f"p3_epoch{epoch+1}.png",
                save_video=True)

    print("\nPhase 3 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    final_model = eqx.combine(diff_model, static_model)
    eqx.tree_serialise_leaves(run_dir / "vwarp_phase3.eqx", final_model)
    eqx.tree_serialise_leaves("vwarp_phase3.eqx", final_model)
    print("✅ Saved Phase 3 Model")

    ## Plot and save loss curve as p3_loss.png
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label="Phase 3 Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Phase 3 Training Loss')
    plt.yscale('log')
    plt.legend()
    plt.draw()
    plt.savefig(run_dir / "plots" / "p3_loss.png")

else:
    final_model = model

#%% Generative Evaluation & Rollout
print("\n Generative Evaluation Rollout (context_ratio=0.0)...")
sample_vis = next(iter(test_loader))[:3]

# _, _, pred_videos = final_model(sample_vis, coords_grid, context_ratio=0.0)
_, _, pred_videos = eval_step(final_model, sample_vis, context_ratio=0.0)

for i in range(pred_videos.shape[0]):
    plot_videos(
        video=pred_videos[i], 
        ref_video=sample_vis[i], 
        plot_ref=True, 
        show_titles=True,
        save_name=run_dir / "plots" / f"p3_autoreg_vis_{i}.png",
        save_video=True
    )



# %% Copy nohup.log to run_dir for record keeping
sys.shutil.copy("nohup.log", run_dir / "nohup_p3.log")