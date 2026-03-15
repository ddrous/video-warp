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
import shutil

from utils import setup_run_dir, get_coords_grid, ssim, plot_videos, count_trainable_params
from loaders import get_dataloaders, torch
from models import WeightCNN, RootMLP, fourier_encode
from jax.flatten_util import ravel_pytree

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
run_dir = setup_run_dir("phase_1", CONFIG, train=TRAIN)

train_loader, test_loader = get_dataloaders(CONFIG, phase="phase_1")
vis_batch = next(iter(train_loader))[:2]
print(f"Sample batch shape: {vis_batch.shape}")
B, H, W, C = vis_batch.shape
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
print(f"    - and the rest of the CNN has {count_trainable_params(encoder) - flat_params.shape[0]} parameters")

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
    def loss_fn(e):
        frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
        offsets = jax.vmap(e)(frames_enc)

        coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
        batched_render = jax.vmap(lambda offset: render_frame(e, offset, coords_grid_t0))
        reconstructed = batched_render(offsets)

        mse_loss = jnp.mean((reconstructed - batch_frames)**2)
        ssim_loss = 1.0 - jnp.mean(jax.vmap(ssim)(reconstructed, batch_frames))
        
        mse_w = CONFIG["phase_1"]["mse_weight"]

        return  mse_w * mse_loss + (1 - mse_w) * ssim_loss

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(enc)
    updates, opt_state = optimizer.update(grads, opt_state, enc, value=loss_val)
    enc = eqx.apply_updates(enc, updates)
    return enc, opt_state, loss_val

@eqx.filter_jit
def eval_step(enc, batch_frames, coords_grid):
    """ Simply compute the reconstructted frames for a given batch, to be used in visualisation. Compute two losses and return them """
    frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
    offsets = jax.vmap(enc)(frames_enc)

    coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
    batched_render = jax.vmap(lambda offset: render_frame(enc, offset, coords_grid_t0))
    reconstructed = batched_render(offsets)

    mse_loss = jnp.mean((reconstructed - batch_frames)**2)
    ssim_loss = 1.0 - jnp.mean(jax.vmap(ssim)(reconstructed, batch_frames))
    
    return reconstructed, mse_loss, ssim_loss

#%% Training Loop

if TRAIN:
    print(f"\n🚀 Starting Phase 1: Encoder Pretraining -> Saving to {run_dir}")
    start_time = time.time()
    epoch_losses_all = []

    for epoch in range(CONFIG["phase_1"]["nb_epochs"]):
        epoch_losses = []
        for batch_frames in train_loader:
            encoder, opt_state, loss = train_step(encoder, opt_state, batch_frames, coords_grid)
            epoch_losses.append(loss)
            epoch_losses_all.append(loss)

        avg_loss = np.mean(epoch_losses)
        if (epoch+1) % (CONFIG["phase_1"]["print_every"]) == 0:
            print(f"Epoch {epoch+1}/{CONFIG["phase_1"]['nb_epochs']} - Avg Loss: {avg_loss:.6f}", flush=True)

        ## Visualize reconstructions every nb_epocsh/10 epochs
        if (epoch+1) % (CONFIG["phase_1"]["nb_epochs"] // 10) == 0:
            recon, mse_loss, ssim_loss = eval_step(encoder, vis_batch, coords_grid)
            # print(f"Sample Recon - MSE Loss: {mse_loss:.6f}, SSIM Loss: {ssim_loss:.6f}", flush=True)
            plot_videos(
                np.expand_dims(recon, axis=1)[0], 
                np.expand_dims(vis_batch, axis=1)[0], 
                plot_ref=True, show_titles=True, save_name=run_dir / "plots" / f"p1_epoch{epoch+1}.png"
            )

    print("\nWall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    eqx.tree_serialise_leaves(run_dir / "vwarp_enc.eqx", encoder)
    eqx.tree_serialise_leaves("vwarp_enc.eqx", encoder)
    print("✅ Saved isolated WeightCNN (with theta_base)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_losses_all, label='Loss')
    ax.set_xlabel('Train Steps')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Phase 1 Training Loss')
    plt.draw()
    plt.savefig(run_dir / "plots" / "p1_loss.png")

else:
    ## Print Warning
    print("⚠️ TRAIN is set to False. Loading encoder and visualizing reconstructions only.")
    encoder = eqx.tree_deserialise_leaves("vwarp_enc.eqx", encoder)
    eqx.tree_serialise_leaves(run_dir / "vwarp_enc.eqx", encoder)

#%% Visualise Reconstructions

print("\nVisualizing GT vs Reconstruction...")
test_batch = next(iter(test_loader))[:5]
reconstructed, mse_loss, ssim_loss = eval_step(encoder, test_batch, coords_grid)
test_id = np.random.randint(0, test_batch.shape[0])
print(f"    - Sample {test_id} from test set...")

plot_videos(
    np.expand_dims(reconstructed, axis=1)[test_id], 
    np.expand_dims(test_batch, axis=1)[test_id], 
    plot_ref=True, show_titles=False, save_name=run_dir / "plots" / "phase1_recons.png"
)


# %% Copy nohup.log to run_dir for record keeping
shutil.copy("nohup.log", run_dir / "nohup_p1.log")
