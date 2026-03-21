#%% Imports and Setup
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

from utils import setup_run_dir, get_coords_grid, plot_videos, count_trainable_params, torch
from loaders import get_dataloaders
from models import VWARP

try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    raise Exception(f"Error: Could not load config.yaml. ({e})")

TRAIN = True
DEBUG = CONFIG.get("debug", False)

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

#%% Phase 3 Forward Definition
def phase3_forward(model, ref_video):
    """ Forward pass for Phase 3: Freezes IDM/FDM and trains GCM to mimic IDM. """
    T = ref_video.shape[0]
    init_frame = ref_video[0]

    # Stop gradients for the frozen parts
    z_init = jax.lax.stop_gradient(model.encoder(jnp.transpose(init_frame, (2, 0, 1))))
    m_init = model.action_model.reset_gcm(T)
    z_A_init = jax.lax.stop_gradient(model.transition_model.mlp_A(z_init)) if model.split_forward else None

    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t, z_tA = carry
        o_tp1, step_idx = scan_inputs

        z_tp1_enc = jax.lax.stop_gradient(model.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
        
        # Ground truth action from IDM (frozen)
        raw_a_target, a_target = jax.lax.stop_gradient(model.action_model.decode_idm(z_t, z_tp1_enc))

        # Predicted action from GCM
        raw_a_pred, quant_a_pred = model.action_model.decode_gcm(m_t, step_idx, jax.lax.stop_gradient(z_t))

        if getattr(model.action_model, "translate_actions", False):
            raw_a_pred = model.action_model.action_bridge(jnp.concatenate([raw_a_pred, z_t], axis=-1))
            quant_a_pred = model.action_model.action_bridge(jnp.concatenate([quant_a_pred, z_t], axis=-1))

        # a_t = a_target
        a_t = quant_a_pred
        # a_t = jnp.where(step_idx%2==1, quant_a_pred, a_target) # Alternate between TF and free-running 

        # Update memory buffer using target action (Teacher Forcing)
        m_tp1 = model.action_model.encode_gcm(m_t, step_idx, jax.lax.stop_gradient(z_t), a_t)

        # Step dynamics (frozen)
        (z_tp1A, z_tp1B), z_tp1 = jax.lax.stop_gradient(model.transition_model(z_t, a_t))

        return (z_tp1, m_tp1, z_tp1A), (raw_a_pred, quant_a_pred, a_target)

    scan_inputs = (ref_video[1:], jnp.arange(1, T))
    _, (actions_preds, actions_pred_q, actions_targets) = jax.lax.scan(scan_step, (z_init, m_init, z_A_init), scan_inputs)

    return actions_preds, actions_pred_q, actions_targets

#%% Model & Optimiser Instantiation
key = jax.random.PRNGKey(CONFIG["seed"])
run_dir = setup_run_dir("phase_3", CONFIG, train=TRAIN)

train_loader, test_loader = get_dataloaders(CONFIG, phase="phase_3")
sample_batch = next(iter(train_loader))
B, T, H, W, C = sample_batch.shape
coords_grid = get_coords_grid(H, W)

key, subkey = jax.random.split(key)
model = VWARP(CONFIG, frame_shape=(H, W, C), key=subkey, init_gcm=True)
## Print parameter counts (in each submodule)
print(f"Total model parameters: {count_trainable_params(model)}")
print(f"    - Encoder: {count_trainable_params(model.encoder)}")
print(f"    - FDM: {count_trainable_params(model.transition_model)}")
print(f"    - IDM: {count_trainable_params(model.action_model.idm)}")
print(f"    - IDM Embeddings: {count_trainable_params(model.action_model.idm_embeddings)}")
print(f"    - GCM : {count_trainable_params(model.action_model.gcm)}")
print(f"    - GCM Embeddings: {count_trainable_params(model.action_model.gcm_embeddings)}")
print(f"    - Action Bridge: {count_trainable_params(model.action_model.action_bridge)}", flush=True)


try:
    dummy = VWARP(CONFIG, frame_shape=(H, W, C), key=subkey, init_gcm=False)
    dummy = eqx.tree_deserialise_leaves(run_dir / "artefacts" / "vwarp_phase2.eqx", dummy)
    
    model = eqx.tree_at(lambda m: m.encoder, model, dummy.encoder)
    model = eqx.tree_at(lambda m: m.transition_model, model, dummy.transition_model)
    model = eqx.tree_at(lambda m: m.action_model.idm, model, dummy.action_model.idm)
    if CONFIG["discrete_actions"]:
        model = eqx.tree_at(lambda m: m.action_model.idm_embeddings, model, dummy.action_model.idm_embeddings)

    print("✅ Transplanted Dynamics & Encoder weights from Phase 2")
except Exception as e:
    print(f"⚠️ Failed to load vwarp_phase2.eqx: {e}")

if not TRAIN:
    print("⏭️  Skipping Phase 3 training...")
    model = eqx.tree_deserialise_leaves(run_dir / "artefacts" / "vwarp_phase3.eqx", model)

# Freeze everything EXCEPT GCM, GCM Embeddings, and Action Bridge (if it exists)
filter_spec = jax.tree_util.tree_map(lambda _: False, model)
gcm_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model.action_model.gcm)
if model.action_model.gcm_embeddings is not None:
    gcm_embeddings_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model.action_model.gcm_embeddings)
if model.action_model.action_bridge is not None:
    action_bridge_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model.action_model.action_bridge)

filter_spec = eqx.tree_at(lambda m: m.action_model.gcm, filter_spec, gcm_mask)
if model.action_model.gcm_embeddings is not None:
    filter_spec = eqx.tree_at(lambda m: m.action_model.gcm_embeddings, filter_spec, gcm_embeddings_mask)
if model.action_model.action_bridge is not None:
    filter_spec = eqx.tree_at(lambda m: m.action_model.action_bridge, filter_spec, action_bridge_mask)

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
        a_preds, a_quant_preds, a_targets = batched_fn(m, ref_videos)
        
        if CONFIG["phase_3"]["loss_type"] == "L1":
            matching_loss = jnp.mean(jnp.abs(a_preds - a_targets))
        else: # Default L2
            matching_loss = jnp.mean((a_preds - a_targets)**2)

        ## Add codebook and commitment losses if using discrete actions
        if CONFIG["discrete_actions"]:
            codebook_loss = jnp.mean((a_quant_preds - jax.lax.stop_gradient(a_preds))**2)
            commitment_loss = jnp.mean((a_preds - jax.lax.stop_gradient(a_quant_preds))**2)
            total_loss = matching_loss + codebook_loss + commitment_loss
            return total_loss
        else:
            return matching_loss

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
    print(f"Trainable Params (GCM and co.): {count_trainable_params(diff_model)}", flush=True)

    start_time = time.time()
    all_losses = []
    for epoch in range(CONFIG["phase_3"]["nb_epochs"]):
        epoch_losses = []
        lr_scales = []
        for batch_videos in train_loader:
            diff_model, opt_state, loss = train_step(diff_model, static_model, opt_state, batch_videos)
            epoch_losses.append(loss)

            lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(lr_scale)

        all_losses.extend(epoch_losses)

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
                show_borders=True,
                save_name=run_dir / "plots" / f"p3_epoch{epoch+1}.png",
                save_video=False)

    print("\nPhase 3 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    final_model = eqx.combine(diff_model, static_model)
    eqx.tree_serialise_leaves(run_dir / "artefacts" / "vwarp_phase3.eqx", final_model)
    print("✅ Saved Phase 3 Model")

    ## Save the array as well
    np.save(run_dir / "artefacts" / "p3_loss.npy", np.array(all_losses))
    np.save(run_dir / "artefacts" / "p3_lr_scales.npy", np.array(lr_scales))

else:
    final_model = model

    try:
        all_losses = np.load(run_dir / "artefacts" / "p3_loss.npy")
        lr_scales = np.load(run_dir / "artefacts" / "p3_lr_scales.npy")
        print("✅ Loaded Phase 3 loss and LR scale history")
    except Exception as e:
        print(f"⚠️ Could not load Phase 3 loss history: {e}")

## Plot and save loss curve as p3_loss.png
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(all_losses, label='Loss')
ax.set_xlabel('Train Step')
num_steps = len(all_losses)
step_ticks = np.linspace(0, num_steps, 5)
ax.set_xticks(step_ticks)
ax.set_xticklabels([f"{int(tick/1000)}k" for tick in step_ticks])
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.set_title('Phase 3 Training Loss')
plt.draw()
plt.savefig(run_dir / "plots" / "p3_loss.png")


#%% Generative Evaluation & Rollout
print("\n Generative Evaluation Rollout (context_ratio=0.0)...")
sample_vis = next(iter(test_loader))[:3]

# _, _, pred_videos = final_model(sample_vis, coords_grid, context_ratio=0.0)
pred_videos = eval_step(final_model, sample_vis, context_ratio=0.0)

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
# shutil.copy("nohup.log", run_dir / "nohup_p3.log")
os.system(f"cp nohup.log {run_dir / 'nohup_p3.log'}")












#%% Custom Generative Evaluation & Rollout

@eqx.filter_jit
def custom_rollout(model, video, context_ratio=0.0):
    T = video.shape[0]
    init_frame = video[0]
    
    z_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    m_init = model.action_model.reset_gcm(T)
    z_A_init = model.transition_model.mlp_A(z_init) if model.split_forward else None

    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t, z_tA = carry
        o_tp1, step_idx = scan_inputs

        time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
        coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
        pred_out = model.render_frame(z_t, coords_grid_t)

        is_context = (step_idx / T) <= context_ratio

        def true_fn():
            return model.action_model.decode_idm(z_t, model.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
        def false_fn():
            raw_a, quant_a = model.action_model.decode_gcm(m_t, step_idx, z_t)

            ## Let's force the quantised to be id 0
            # quant_a = model.action_model.gcm_embeddings(0)
            # quant_a = jnp.zeros_like(quant_a)

            if model.action_model.translate_actions:
                raw_a = model.action_model.action_bridge(jnp.concatenate([raw_a, z_t], axis=-1))
                quant_a = model.action_model.action_bridge(jnp.concatenate([quant_a, z_t], axis=-1))
            return raw_a, quant_a

        _, a_t = jax.lax.cond(
            is_context,
            lambda: true_fn(),
            lambda: false_fn()
        )

        m_tp1 = model.action_model.encode_gcm(m_t, step_idx, z_t, a_t)
        (z_tp1A, _), z_tp1 = model.transition_model(z_t, a_t)

        return (z_tp1, m_tp1, z_tp1A), (a_t, z_t, pred_out)

    scan_inputs = (jnp.concatenate([video[1:], jnp.zeros_like(video[:1])], axis=0), jnp.arange(1, T+1))
    _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, z_A_init), scan_inputs)
    
    return actions, pred_latents, pred_video

sample_vis = next(iter(test_loader))
test_seq_id = np.random.randint(0, sample_vis.shape[0])
# test_seq_id = 102

print(f"\n Custom Rollout on test sequence {test_seq_id} with custom context_ratio")
_, _, pred_video = custom_rollout(final_model, sample_vis[test_seq_id], context_ratio=0.25)

plot_videos(
    video=pred_video, 
    ref_video=sample_vis[test_seq_id], 
    plot_ref=True, 
    show_titles=True,
    show_borders=True,
    save_name=run_dir / "plots" / f"p3_custom_ar_{test_seq_id}.png",
    save_video=True
)
