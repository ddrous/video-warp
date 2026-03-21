#%% Imports and Setup
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from utils import setup_run_dir, get_coords_grid, plot_videos, count_trainable_params
from loaders import get_dataloaders, torch
from models import WeightCNN, RootMLP, fourier_encode
from jax.flatten_util import ravel_pytree
import dm_pix as pix

try:
    cfg_name = sys.argv[1]
    with open(cfg_name, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"Warning: Could not load specified config file from command line. ({e}) Falling back to config.yaml")
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

#%% Initialization

# data_dir = CONFIG["data_path"]
# CONFIG["data_path"] = "../../" + str(data_dir)

run_dir = setup_run_dir("phase_1", CONFIG, train=TRAIN)
print(f"Runing and saving Phase 1 outputs to: {run_dir}", flush=True)

train_loader, test_loader = get_dataloaders(CONFIG, phase="phase_1")

vis_batch = next(iter(train_loader))
print(f"Sample batch shape: {vis_batch.shape}", flush=True)
if hasattr(train_loader.dataset, "max_val"):
    emp_max_val  = train_loader.dataset.max_val
    print(f"Empirical max val in the training set: {emp_max_val:.4f}", flush=True)
else:
    print("WARNING. Max val not provided. Assuming data is normealised RGB pixel vals, so empirical data range is 1.0. Needed for SSIM")
    emp_max_val = 1.0

print(f"Max val sanity check in the sample batch: {vis_batch.max():.4f}\n", flush=True)

if vis_batch.ndim == 4:
    B, H, W, C = vis_batch.shape
    T = 1
elif vis_batch.ndim == 5:
    B, T, H, W, C = vis_batch.shape
coords_grid = get_coords_grid(H, W)

key, k_root, k_enc = jax.random.split(key, 3)

coord_dim = 2 + 2 * 2 * CONFIG["num_fourier_freqs"]
add_time = 1 if CONFIG.get("use_time_in_root", False) else 0

template_root = RootMLP(
    coord_dim+add_time, C, CONFIG["root_width"], 
    CONFIG["root_depth"], CONFIG["root_activation"], k_root
)

flat_params, unravel_fn = ravel_pytree(template_root)
d_theta = flat_params.shape[0]

encoder = WeightCNN(
    in_channels=C, out_dim=d_theta, spatial_shape=(H, W), 
    theta_base=flat_params, key=k_enc, 
    hidden_width=CONFIG["cnn_hidden_width"], depth=CONFIG["cnn_depth"]
)

## Print parameter counts (in the layers, and in the theta base)
print(f"Total encoder parameters: {count_trainable_params(encoder)}")
print(f"    - of which theta_base has {flat_params.shape[0]} parameters")
print(f"    - and the layers of the CNN has {count_trainable_params(encoder.layers)} parameters", flush=True)

def render_frame(enc, offset, coords_grid):
    flat_coords = coords_grid.reshape(-1, 3)
    theta = offset + enc.theta_base

    def render_pt(th, coord):
        root = unravel_fn(th)
        # Use the exact same encoding function as the rest of the model!
        encoded_spatial = fourier_encode(coord[1:], CONFIG["num_fourier_freqs"])

        # Conditionally add time if configured
        if CONFIG.get("use_time_in_root", False):
            encoded_coord = jnp.concatenate([coord[:1], encoded_spatial], axis=-1)
        else:
            encoded_coord = encoded_spatial
            
        return root(encoded_coord)
        
    pred_flat = jax.vmap(render_pt, in_axes=(None, 0))(theta, flat_coords)
    return pred_flat.reshape(H, W, -1)



#%% Setup Optimiser
optimizer = optax.chain(
    optax.adam(CONFIG["phase_1"]["learning_rate"]),
    optax.contrib.reduce_on_plateau(
        patience=CONFIG["phase_1"]["lr_patience"], cooldown=CONFIG["phase_1"]["lr_cooldown"],
        factor=CONFIG["phase_1"]["lr_factor"], rtol=CONFIG["phase_1"]["lr_rtol"],
        accumulation_size=CONFIG["phase_1"]["lr_accum_size"], min_scale=CONFIG["phase_1"]["lr_min_scale"]
    )
)
opt_state = optimizer.init(encoder)

@eqx.filter_jit
def train_step(enc, opt_state, batch_frames, coords_grid):

    ## If the batch frames haave a time dimension, we must flatten along that channel
    if batch_frames.ndim == 5:  # (B, T, H, W, C) -> (B*T, H, W, C)
        B, T, H, W, C = batch_frames.shape
        batch_frames = batch_frames.reshape(B*T, H, W, C)

    def loss_fn(e):
        frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
        offsets = jax.vmap(e)(frames_enc)

        coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
        batched_render = jax.vmap(lambda offset: render_frame(e, offset, coords_grid_t0))
        reconstructed = batched_render(offsets)

        mse_loss = jnp.mean((reconstructed - batch_frames)**2)

        # ssim_loss = 1.0 - jnp.mean(jax.vmap(ssim)(reconstructed, batch_frames))

        def compute_ssim(pred, target):
            return pix.ssim(pred, target, max_val=emp_max_val)

        ssim_loss = 1.0 - jnp.mean(jax.vmap(compute_ssim)(reconstructed, batch_frames))

        mse_w = CONFIG["phase_1"]["mse_weight"]

        return  mse_w * mse_loss + (1 - mse_w) * ssim_loss

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(enc)
    updates, opt_state = optimizer.update(grads, opt_state, enc, value=loss_val)
    enc = eqx.apply_updates(enc, updates)
    return enc, opt_state, loss_val

@eqx.filter_jit
def eval_step(enc, batch_frames, coords_grid):

    ## If the batch frames haave a time dimension, we must flatten along that channel
    orig_dim = batch_frames.ndim
    if orig_dim == 5:  # (B, T, H, W, C) -> (B*T, H, W, C)
        B, T, H, W, C = batch_frames.shape
        batch_frames = batch_frames.reshape(B*T, H, W, C)

    """ Simply compute the reconstructted frames for a given batch, to be used in visualisation. Compute two losses and return them """
    frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
    offsets = jax.vmap(enc)(frames_enc)

    coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
    batched_render = jax.vmap(lambda offset: render_frame(enc, offset, coords_grid_t0))
    reconstructed = batched_render(offsets)

    mse_loss = jnp.mean((reconstructed - batch_frames)**2)

    def compute_ssim(pred, target):
        return pix.ssim(pred, target, max_val=emp_max_val)
    ssim_loss = 1.0 - jnp.mean(jax.vmap(compute_ssim)(reconstructed, batch_frames))
    
    ## Reshape the reconstructed frames back to (B, T, H, W, C) if needed
    if orig_dim == 5:
        reconstructed = reconstructed.reshape(B, T, H, W, C)

    return reconstructed, mse_loss, ssim_loss

#%% Training Loop

## if f"vwarp_enc_{CONFIG["dataset"].lower()}.eqx" exists, load it instead of training
if os.path.exists(f"artefacts/vwarp_enc_{CONFIG['dataset'].lower()}.eqx"):
    print(f"Found existing pretrained encoder at ./artefacts/vwarp_enc_{CONFIG['dataset'].lower()}.eqx. Skipping training...")
    TRAIN = False

if TRAIN:
    print(f"\n🚀 Starting Phase 1: Encoder Pretraining -> Saving to {run_dir}")
    start_time = time.time()
    epoch_losses_all = []
    lr_scales = []

    for epoch in range(CONFIG["phase_1"]["nb_epochs"]):
        epoch_losses = []
        for batch_frames in train_loader:
            encoder, opt_state, loss = train_step(encoder, opt_state, batch_frames, coords_grid)
            epoch_losses.append(loss)
            epoch_losses_all.append(loss)

            lr_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(lr_scale)

        avg_loss = np.mean(epoch_losses)
        if (epoch+1) % (CONFIG["phase_1"]["print_every"]) == 0:
            print(f"Epoch {epoch+1}/{CONFIG["phase_1"]['nb_epochs']} - Avg Loss: {avg_loss:.6f} - LR Scale: {lr_scale:.4f}", flush=True)

        ## Visualize reconstructions every nb_epocsh/10 epochs
        if (epoch+1) % max(1, CONFIG["phase_1"]["nb_epochs"] // 10) == 0:
            recon, mse_loss, ssim_loss = eval_step(encoder, vis_batch, coords_grid)
            # print(f"Sample Recon - MSE Loss: {mse_loss:.6f}, SSIM Loss: {ssim_loss:.6f}", flush=True)
            plot_videos(
                np.expand_dims(recon, axis=1)[0] if recon.ndim == 4 else recon[0],
                np.expand_dims(vis_batch, axis=1)[0] if vis_batch.ndim == 4 else vis_batch[0],
                show_borders=True,
                corner_radius=5,
                no_rescale=True,
                cmap="grey" if CONFIG["dataset"].lower() == "movingmnist" else "viridis",
                plot_ref=True, show_titles=True, save_name=run_dir / "plots" / f"p1_epoch{epoch+1}.png"
            )

    print("\nWall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    eqx.tree_serialise_leaves(run_dir / "artefacts" / "vwarp_enc.eqx", encoder)
    eqx.tree_serialise_leaves(f"artefacts/vwarp_enc_{CONFIG["dataset"].lower()}.eqx", encoder)
    print("✅ Saved isolated WeightCNN (with theta_base)")

    ## Save the array as well
    np.save(run_dir / "artefacts" / "p1_loss.npy", np.array(epoch_losses_all))
    np.save(run_dir / "artefacts" / "p1_lr_scales.npy", np.array(lr_scales))

else:
    ## Print Warning
    print("⚠️ TRAIN is set to False. Attempting to load encoder and visualizing reconstructions only.")
    try:
        encoder = eqx.tree_deserialise_leaves(f"artefacts/vwarp_enc_{CONFIG["dataset"].lower()}.eqx", encoder)
        eqx.tree_serialise_leaves(run_dir / "artefacts" / "vwarp_enc.eqx", encoder)
    except FileNotFoundError as e:
        try:
            encoder = eqx.tree_deserialise_leaves(run_dir / "artefacts" / "vwarp_enc.eqx", encoder)
        except Exception as e:
            print(f"⚠️ Could not find pretrained encoder. Continuing with untrained. ({e})")

    ## Load the loss and lr scale arrays if they exist, for visualisation
    try:
        epoch_losses_all = np.load(run_dir / "artefacts" / "p1_loss.npy")
        lr_scales = np.load(run_dir / "artefacts" / "p1_lr_scales.npy")
    except Exception as e:
        print(f"⚠️ Could not load loss and lr scale arrays. ({e}) Initializing empty arrays for visualisation.")
        epoch_losses_all = np.array([])
        lr_scales = np.array([])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_losses_all, label='Loss')
ax.set_xlabel('Train Step')
num_steps = len(epoch_losses_all)
step_ticks = np.linspace(0, num_steps, 5)
ax.set_xticks(step_ticks)
ax.set_xticklabels([f"{int(tick/1000)}k" for tick in step_ticks])
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.set_title('Phase 1 Training Loss')
plt.draw()
plt.savefig(run_dir / "plots" / "p1_loss.png")

#%% Visualise Reconstructions

print("\nVisualizing GT vs Reconstruction...")
# test_batch = next(iter(train_loader))[:1]
test_batch = vis_batch[:5]
reconstructed, mse_loss, ssim_loss = eval_step(encoder, test_batch, coords_grid)
test_id = np.random.randint(0, test_batch.shape[0])
print(f"    - Sample {test_id} from test set...")

print(reconstructed.shape, test_batch.shape)

plot_videos(
    np.expand_dims(reconstructed, axis=1)[test_id] if reconstructed.ndim == 4 else reconstructed[test_id],
    np.expand_dims(test_batch, axis=1)[test_id] if test_batch.ndim == 4 else test_batch[test_id],
    show_borders=True,
    corner_radius=5,
    no_rescale=True,
    cmap="grey" if CONFIG["dataset"].lower() == "movingmnist" else "viridis",
    plot_ref=True, show_titles=True, save_name=run_dir / "plots" / "phase1_recons.png"
)


# %% Copy nohup.log to run_dir for record keeping
# shutil.copy("nohup.log", run_dir / "nohup_p1.log")
os.system(f"cp nohup.log {run_dir / 'nohup_p1.log'}")


## if dim==4, then expand the time dimension for consistent indexing in the rest of the code
if reconstructed.ndim == 4:
    reconstructed = np.expand_dims(reconstructed, axis=1)
if test_batch.ndim == 4:
    test_batch = np.expand_dims(test_batch, axis=1)

print(f"Actual values of rec vs gt (for the sample frame visualised above): \nRecon: {reconstructed[test_id, 0, :5, :5, 0]} \nGT: {test_batch[test_id, 0, :5, :5, 0]}", flush=True)

## print more central pixels for the same frame, to check if the model is at least getting the general structure right
center_h, center_w = reconstructed.shape[2] // 2, reconstructed.shape[3] // 2
print(f"Central 5x5 patch values: \nRecon: {reconstructed[test_id, 0, center_h-2:center_h+3, center_w-2:center_w+3, 0]} \nGT: {test_batch[test_id, 0, center_h-2:center_h+3, center_w-2:center_w+3, 0]}", flush=True)


## Imshow the two side by side for a quick visual check
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(reconstructed[test_id, 0], cmap='viridis', vmin=-1, vmax=1)
plt.title("Reconstructed")
plt.axis('off') 

plt.subplot(1, 2, 2)
plt.imshow(test_batch[test_id, 0], cmap='viridis', vmin=-1, vmax=1)
plt.title("Ground Truth")
plt.axis('off')
plt.suptitle("Sample Reconstruction vs Ground Truth")
