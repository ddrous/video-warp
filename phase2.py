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
from pathlib import Path
import sys

from utils import setup_run_dir, get_coords_grid, plot_videos, count_trainable_params, ssim
from loaders import get_dataloaders
from models import VWARP


try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    raise Exception(f"Error: Could not load config.yaml. ({e})")

TRAIN = True
DEBUG = CONFIG.get("debug", False)

#%% Phase 2 Forward Definition
def phase2_forward(model, ref_video, coords_grid, render):
    """ Forward pass for Phase 2: Focuses on IDM and FDM. """
    T = ref_video.shape[0]
    init_frame = ref_video[0]

    z_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    z_init = jax.lax.stop_gradient(z_init)
    a_init = jnp.zeros((model.lam_dim,))

    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, a_tm1 = carry
        o_tp1, step_idx = scan_inputs

        if render or CONFIG["phase_2"]["loss_type"] == "pixel":
            time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            pred_out = model.render_frame(z_t, coords_grid_t)
        else:
            pred_out = None

        z_tp1_enc = jax.lax.stop_gradient(model.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
        a_t_raw, a_t_quant = model.action_model.inverse_dynamics(z_t, z_tp1_enc)

        if getattr(model, "use_action_residuals", False):
            a_t_raw = a_t_raw + a_tm1
            a_t_quant = a_t_quant + a_tm1

        # Gradient STE Estimator
        a_t = a_t_raw + jax.lax.stop_gradient(a_t_quant - a_t_raw)
        z_tp1 = model.transition_model(z_t, a_t)

        return (z_tp1, a_t), ((a_t_raw, a_t_quant), (z_tp1, z_tp1_enc), pred_out)

    scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
    _, (actions, latents, pred_video) = jax.lax.scan(scan_step, (z_init, a_init), scan_inputs)

    return actions, latents, pred_video

def apply_augmentations(ref_videos, key):
    k_aug, k_init = jax.random.split(key, 2)
    
    if CONFIG["phase_2"]["reverse_video_aug"]:
        do_reverse = jax.random.bernoulli(k_aug, 0.5, shape=(ref_videos.shape[0],))
        ref_videos = jax.vmap(lambda rev, vid: jax.lax.cond(rev, lambda v: jnp.flip(v, axis=0), lambda v: v, vid))(do_reverse, ref_videos)

    if CONFIG["phase_2"]["static_video_aug"]:
        add_to_front = jax.random.bernoulli(k_init, 0.5, shape=(ref_videos.shape[0],))
        nb_frames = ref_videos.shape[1]
        repeat_frames = nb_frames // 4
        ref_videos = jax.vmap(lambda add_front, vid: jax.lax.cond(
            add_front, 
            lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
            lambda v_in: jnp.concatenate([v_in[:nb_frames-repeat_frames], jnp.repeat(v_in[nb_frames-repeat_frames:nb_frames-repeat_frames+1], repeats=repeat_frames, axis=0)], axis=0),
            vid
        ))(add_to_front, ref_videos)
    return ref_videos

#%% Model & Optimiser Instantiation
key = jax.random.PRNGKey(CONFIG["seed"])
run_dir = setup_run_dir("phase_2", CONFIG, train=TRAIN)

train_loader, test_loader = get_dataloaders(CONFIG, phase="phase_2")
sample_batch = next(iter(train_loader))
B, T, H, W, C = sample_batch.shape
coords_grid = get_coords_grid(H, W)

key, subkey = jax.random.split(key)
model = VWARP(CONFIG, frame_shape=(H, W, C), key=subkey, init_gcm=False)

try:
    dummy_encoder = eqx.tree_deserialise_leaves(f"vwarp_enc.eqx", model.encoder)
    model = eqx.tree_at(lambda m: m.encoder, model, dummy_encoder)
    print(f"✅ Loaded Pretrained WeightCNN from ./")
except Exception as e:
    raise Exception(f"Error: Could not load pretrained encoder from ./vwarp_enc.eqx. ({e})")

if not TRAIN:
    print("⏭️  Skipping Phase 2 training, loading...")
    model = eqx.tree_deserialise_leaves("vwarp_phase2.eqx", model)

# FIX: Partition the model into trainable and frozen parts
filter_spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
filter_spec = eqx.tree_at(lambda m: m.encoder, filter_spec, replace=jax.tree_util.tree_map(lambda _: False, model.encoder))

diff_model, static_model = eqx.partition(model, filter_spec)

optimizer = optax.chain(
    optax.adam(CONFIG["phase_2"]["learning_rate"]),
    optax.contrib.reduce_on_plateau(
        patience=CONFIG["phase_2"]["lr_patience"], cooldown=CONFIG["phase_2"]["lr_cooldown"],
        factor=CONFIG["phase_2"]["lr_factor"], rtol=CONFIG["phase_2"]["lr_rtol"],
        accumulation_size=CONFIG["phase_2"]["lr_accum_size"], min_scale=CONFIG["phase_2"]["lr_min_scale"]
    )
)
# Initialize ONLY with the trainable parts
opt_state = optimizer.init(diff_model)


@eqx.filter_jit
def train_step(diff_m, static_m, opt_state, batch_keys, in_videos, coords_grid):
    def loss_fn(d_model):
        # Recombine the trainable and frozen parts before doing the forward pass
        m = eqx.combine(d_model, static_m)
        
        ref_videos = apply_augmentations(in_videos, batch_keys[0])
        batched_fn = jax.vmap(phase2_forward, in_axes=(None, 0, None, None))
        
        needs_render = (CONFIG["phase_2"]["loss_type"] == "pixel")
        (raw_actions, quant_actions), (pred_lats, gt_lats), pred_videos = batched_fn(m, ref_videos, coords_grid, needs_render)

        if CONFIG["phase_2"]["loss_type"] == "latent":
            loss = jnp.mean((pred_lats[:, :-1] - gt_lats[:, :-1])**2) 
        else:
            target_videos = jnp.concatenate([ref_videos[:, 1:], jnp.zeros_like(ref_videos[:, :1])], axis=1)
            mse_loss = jnp.mean((pred_videos[:, :-1] - target_videos[:, :-1])**2)
            ssim_loss = 1.0 - jnp.mean(jax.vmap(jax.vmap(ssim))(pred_videos[:, :-1], target_videos[:, :-1]))
            loss = ssim_loss + CONFIG["phase_2"]["mse_weight"] * mse_loss

        if CONFIG["discrete_actions"]:
            book_loss = jnp.mean((jax.lax.stop_gradient(raw_actions) - quant_actions)**2)
            commit_loss = jnp.mean((raw_actions - jax.lax.stop_gradient(quant_actions))**2)
            loss += book_loss + commit_loss

        return loss

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(diff_m)
    updates, opt_state = optimizer.update(grads, opt_state, diff_m, value=loss_val)
    diff_m = eqx.apply_updates(diff_m, updates)
    return diff_m, opt_state, loss_val


@eqx.filter_jit
def eval_step(model, batch_videos, coords_grid):
    actions, latents, pred_videos = jax.vmap(phase2_forward, in_axes=(None, 0, None, None))(model, batch_videos, coords_grid, True)

    # FIX: Return all three values so the unpacking at the bottom works!
    return actions, latents, pred_videos

#%% Training Loop
if TRAIN:
    print(f"\n🚀 Starting Phase 2: Dynamics Fitting -> Saving to {run_dir}")
    print(f"Trainable Params: {count_trainable_params(diff_model)}")

    start_time = time.time()
    for epoch in range(CONFIG["phase_2"]["nb_epochs"]):
        epoch_losses = []
        lr_scales = []
        for batch_videos in train_loader:
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch_videos.shape[0])
            
            # Pass partitioned models
            diff_model, opt_state, loss = train_step(diff_model, static_model, opt_state, batch_keys, batch_videos, coords_grid)
            epoch_losses.append(loss)

            lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(lr_scale)

        if (epoch+1) % CONFIG["phase_2"]["print_every"] == 0:
            print(f"Phase 2 - Epoch {epoch+1}/{CONFIG['phase_2']['nb_epochs']} - Avg Loss: {np.mean(epoch_losses):.6f} - LR Scale: {np.mean(lr_scales):.4f}")

        ## Visualise reconstructions every nb_epochs/10 epochs
        if (epoch+1) % max(1, (CONFIG["phase_2"]["nb_epochs"] // 10)) == 0:
            # Recombine to evaluate
            current_model = eqx.combine(diff_model, static_model)
            _, _, pred_videos = eval_step(current_model, sample_batch, coords_grid)
            plot_videos(
                video=pred_videos[0], 
                ref_video=sample_batch[0], 
                plot_ref=True, 
                save_name=run_dir / "plots" / f"p2_epoch{epoch+1}.png"
            )

    print("\nPhase 2 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # Recombine to save the full model
    model = eqx.combine(diff_model, static_model)
    eqx.tree_serialise_leaves(run_dir / "vwarp_phase2.eqx", model)
    eqx.tree_serialise_leaves("vwarp_phase2.eqx", model)
    print("✅ Saved Phase 2 Model")

    ## Plot and save the loss as p2_loss.png
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label="Phase 2 Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Phase 2 Training Loss')
    plt.legend()
    plt.draw()
    plt.savefig(run_dir / "plots" / "p2_loss.png")

else:
    # Just in case TRAIN is false, make sure model is unified
    model = eqx.combine(diff_model, static_model)

#%% Visualisations
print("\n Generating Visualizations...")
sample_vis = next(iter(test_loader))[:3]

(raw_acts, quant_acts), _, pred_videos = eval_step(model, sample_vis, coords_grid)

targets = np.concatenate([sample_vis[:, 1:], np.zeros_like(sample_vis[:, :1])], axis=1)

for i in range(pred_videos.shape[0]):
    plot_videos(
        video=pred_videos[i, :-1], 
        ref_video=targets[i, :-1], 
        plot_ref=True, 
        save_name=run_dir / "plots" / f"p2_vis_{i}.png",
        save_video=True
    )

if CONFIG["discrete_actions"]:
    print("Plotting Discrete Latent Action Heatmap for Sequence 0")
    plt.figure(figsize=(12, 6))
    sns.heatmap(quant_acts[0].T, cmap="coolwarm", center=0, annot=True, fmt=".4f", annot_kws={'size': 8})
    plt.xlabel("Time Step (t)")
    plt.ylabel("Latent Dimension")
    plt.title("Discrete Latent Action Heatmap (Phase 2)")
    plt.tight_layout()
    plt.savefig(run_dir / "plots" / "p2_action_heatmap.png")
    plt.show()