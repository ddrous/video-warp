#%% Cell 1: Imports, Utilities, and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
from jax.flatten_util import ravel_pytree
from typing import Optional

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

from matplotlib import patches
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

print("\n=== WARP: Weight-Space Adaptive Recurrent Prediction ===\n")

def count_trainable_params(model):
    def count_params(x):
        if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float32, jnp.float64]:
            return x.size
        return 0
    param_counts = jax.tree_util.tree_map(count_params, model)
    return sum(jax.tree_util.tree_leaves(param_counts))

# --- Configuration ---
TRAIN_PHASE_0 = False  # Optional pretraining of Encoder
TRAIN_PHASE_1 = True  # Train WARP Model directly
TRAIN_PHASE_2 = False  # Not needed for WARP, kept False
RUN_DIR = "./" if (not TRAIN_PHASE_1) else None

SINGLE_BATCH = False
USE_NLL_LOSS = False # Set to True for NLL loss (mean/std), False for MSE (preferred)

CONFIG = {
    "seed": 42,
    
    # Training Params
    # "p1_nb_epochs": 2500,        
    "p1_nb_epochs": 1000,        
    "p1_learning_rate": 1e-6 if USE_NLL_LOSS else 1e-4,
    "reverse_video_aug": False,
    "static_video_aug": False,
    
    # Phase 2 Params (Unused in WARP but kept for compatibility)
    "p2_nb_epochs": 1500,
    "p2_learning_rate": 1e-4,

    "print_every": 10,
    "batch_size": 2 if SINGLE_BATCH else 128*2,
    "use_nll_loss": USE_NLL_LOSS,

    # --- Architecture Params ---
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_time_in_root": False,
    "pretrain_encoder": TRAIN_PHASE_0, 

    # --- Plateau Scheduler Config ---
    "lr_patience": 400,      
    "lr_cooldown": 100,       
    "lr_factor": 0.5,        
    "lr_rtol": 1e-3,         
    "lr_accum_size": 5,     
    "lr_min_scale": 1e-0     
}

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

def setup_run_dir(base_dir="runs"):
    if TRAIN_PHASE_1:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)

        current_script = Path(__file__)
        if current_script.exists():
            shutil.copy(current_script, run_path / "main.py")
        
        (run_path / "artefacts").mkdir(exist_ok=True)
        (run_path / "plots").mkdir(exist_ok=True)
        return run_path
    else:
        return Path(RUN_DIR) if RUN_DIR else Path("./")

run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"
os.makedirs(artefacts_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

P1_LOAD_PATH = artefacts_path / "model_phase1_final.eqx" 

#%% Cell 2: PyTorch Data Loading & Plotting Helpers
def numpy_collate(batch):
    if isinstance(batch[0], tuple):
        videos = torch.stack([b[0] for b in batch]).numpy()
    else:
        videos = torch.stack(batch).numpy()
    
    if videos.ndim == 4:
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 5 and videos.shape[2] == 1:
        videos = np.transpose(videos, (0, 1, 3, 4, 2))

    videos = videos.astype(np.float32)
    if videos.max() > 2.0:
        videos = videos / 255.0
    return videos

print("Loading Moving MNIST Dataset...")
try:
    data_path = '../data' if (TRAIN_PHASE_1) else '../../../data'

    ## Manually load train and test splits to have more control over batching and shuffling
    mov_mnist_arrays = np.load(data_path + "/MovingMNIST/mnist_test_seq.npy")
    print(f"Original loaded MovingMNIST shape: {mov_mnist_arrays.shape} (T, N, H, W)")
    
    ## Split, 8000 train, 2000 test
    train_arrays = mov_mnist_arrays[:, :8000]
    test_arrays = mov_mnist_arrays[:, 8000:]

    class MovingMNISTDataset(torch.utils.data.Dataset):
        def __init__(self, data_array):
            self.data_array = data_array

        def __len__(self):
            return self.data_array.shape[1]

        def __getitem__(self, idx):
            video = self.data_array[:, idx]  # Shape (T, H, W)
            video = np.expand_dims(video, axis=-1)  # Add channel dimension -> (T, H, W, 1)
            return torch.from_numpy(video.astype(np.float32))

    dataset = MovingMNISTDataset(train_arrays)

    if SINGLE_BATCH:
        training_subset = Subset(dataset, range(CONFIG["batch_size"]))
        train_loader = DataLoader(
            training_subset, 
            batch_size=CONFIG["batch_size"]//1, 
            shuffle=False, 
            collate_fn=numpy_collate, 
            drop_last=True
        )
    else:
        train_loader = DataLoader(dataset, 
                                  batch_size=CONFIG["batch_size"], 
                                  shuffle=True, 
                                  collate_fn=numpy_collate, 
                                  drop_last=False)

    sample_batch = next(iter(train_loader))
    B, nb_frames, H, W, C = sample_batch.shape
    print(f"Batched Video shape: {sample_batch.shape}")
except Exception as e:
    print(f"Could not load MovingMNIST: {e}")
    raise e

y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1) 

def sbimshow(img, title="", ax=None):
    img = np.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if ax is None:
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

def plot_pred_ref_videos_rollout(video, ref_video, title="Render", save_name=None):
    nb_frames = video.shape[0]

    rescale = False
    if ref_video[..., :C].min() < 0.0:
        rescale = True
        ref_video = (ref_video + 1.0) / 2.0

    if video.shape[-1] == 1:
        fig, axes = plt.subplots(2, nb_frames, figsize=(2*nb_frames, 2*2))
        indices_to_plot = list(np.arange(0, nb_frames))

        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx] if not rescale else (video[idx] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"Pred t={idx}", ax=axes[0, i])
            ref_idx = min(idx, ref_video.shape[0]-1)
            sbimshow(ref_video[ref_idx], title=f"Ref t={ref_idx}", ax=axes[1, i])
    else:
        fig, axes = plt.subplots(3, nb_frames, figsize=(2*nb_frames, 3*2))
        indices_to_plot = list(np.arange(0, nb_frames))

        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx, ..., :C] if not rescale else (video[idx, ..., :C] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"Mean t={idx}", ax=axes[0, i])
            sbimshow(video[idx, ..., C:], title=f"Std t={idx}", ax=axes[1, i])
            ref_idx = min(idx, ref_video.shape[0]-1)
            sbimshow(ref_video[ref_idx], title=f"Ref t={ref_idx}", ax=axes[2, i])

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    plt.show()
    plt.close()

#%%

def plot_videos(video, ref_video=None, plot_ref=True, show_titles=True, show_labels=True, forecast_start=None, 
                                 vmin=None, vmax=None, save_name=None, 
                                 wspace=0.05, hspace=0.05, forecast_gap=0.2, 
                                 save_video=False, video_gap=5, cmap='grey'):
    """
    Plots a camera-ready rollout of ground truth and predicted video frames.
    
    Args:
        video (np.ndarray): Predicted video frames of shape (T, H, W, C).
        ref_video (np.ndarray, optional): Ground truth video frames of shape (T, H, W, C).
        plot_ref (bool): Whether to plot the ground truth reference row (Top).
        show_titles (bool): Whether to show time steps and side labels.
        forecast_start (int, optional): The 1-indexed time step where forecasting begins.
        vmin (float, optional): Min intensity scale (only applies if plot_ref=False).
        vmax (float, optional): Max intensity scale (only applies if plot_ref=False).
        save_name (str, optional): Path to save the figure.
        wspace (float): Horizontal spacing between frames.
        hspace (float): Vertical spacing between rows.
        forecast_gap (float): The width of the gap indicating the forecast start (relative to frame width).
        save_video (bool): If True, also exports a GIF.
        video_gap (int): Pixel height of the gap separating GT and Pred in the GIF.
    """
    # Modern, "techy" font setup (clean sans-serif commonly used in ML papers)
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        # 'font.sans-serif': ['DejaVu', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
        'font.size': 18,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }):
        
        nb_frames = video.shape[0]
        C = video.shape[-1]
        
        if plot_ref and ref_video is None:
            raise ValueError("ref_video must be provided if plot_ref is True.")

        # Handle normalization
        rescale = False
        if plot_ref and ref_video[..., :C].min() < -0.5:
            rescale = True
            ref_video = (ref_video + 1.0) / 2.0
        elif not plot_ref and video.min() < -0.5:
            rescale = True

        nrows = 2 if plot_ref else 1
        
        # Layout math: Create a spacer column if a forecast start is defined
        has_gap = forecast_start is not None and 1 < forecast_start <= nb_frames
        ncols = nb_frames + 1 if has_gap else nb_frames
        
        width_ratios = [1.0] * ncols
        spacer_col = -1
        if has_gap:
            spacer_col = forecast_start - 1
            width_ratios[spacer_col] = forecast_gap

        # Apply the parameterized forecast gap to the figure width
        fig_width = (nb_frames + (forecast_gap if has_gap else 0.0)) * 1.5
        fig = plt.figure(figsize=(fig_width, nrows * 1.55))
        
        # Apply the parameterized wspace and hspace
        gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios)
        
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, c])

        # Configure kwargs for pure predictions
        imshow_kwargs = {}
        if not plot_ref:
            if vmin is not None: imshow_kwargs['vmin'] = vmin
            if vmax is not None: imshow_kwargs['vmax'] = vmax

        frame_idx = 0
        for c in range(ncols):
            # 1. Handle Spacer Column
            if c == spacer_col:
                for r in range(nrows):
                    axes[r, c].axis('off')
                continue
                
            # 2. Prepare Frames
            pred_frame = video[frame_idx]
            if rescale: pred_frame = (pred_frame + 1.0) / 2.0
            
            # Only strictly clip if the user isn't supplying custom vmin/vmax overrides
            if plot_ref or (vmin is None and vmax is None):
                pred_frame = np.clip(pred_frame, 0.0, 1.0)
            
            if pred_frame.shape[-1] == 1: pred_frame = np.repeat(pred_frame, 3, axis=-1)

            if plot_ref:
                ref_idx = min(frame_idx, ref_video.shape[0] - 1)
                ref_frame = ref_video[ref_idx]
                if rescale: ref_frame = (ref_frame + 1.0) / 2.0
                ref_frame = np.clip(ref_frame, 0.0, 1.0)
                if ref_frame.shape[-1] == 1: ref_frame = np.repeat(ref_frame, 3, axis=-1)

            # 3. Render Images
            if plot_ref:
                axes[0, c].imshow(ref_frame)
                axes[1, c].imshow(pred_frame)
                target_axes = [axes[0, c], axes[1, c]]
            else:
                axes[0, c].imshow(pred_frame, **imshow_kwargs)
                target_axes = [axes[0, c]]

            # 4. Clean up axes (removes outlines)
            for ax in target_axes:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # 5. Handle Time Indices
            if show_titles:
                top_ax = axes[0, c]
                if frame_idx == 0 or (frame_idx + 1 == forecast_start):
                    title_str = f"$t={frame_idx + 1}$"
                else:
                    title_str = str(frame_idx + 1)
                
                font_weight = 'bold' if (has_gap and frame_idx + 1 == forecast_start) else 'normal'
                top_ax.set_title(title_str, pad=8, fontsize=18, fontweight=font_weight)

            frame_idx += 1

        # 6. Add Large Row Labels before the sequence
        if show_labels:
            if plot_ref:
                axes[0, 0].set_ylabel("GT", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
                # axes[0, 0].set_ylabel("Clean", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
                axes[1, 0].set_ylabel("Pred", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
                # axes[1, 0].set_ylabel("Edited", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
            else:
                axes[0, 0].set_ylabel("Pred", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')

        if save_name:
            plt.savefig(save_name, dpi=100, bbox_inches='tight', facecolor='white', transparent=False)
        
        plt.show()
        plt.close()

        # ---------------------------------------------------------
        # 7. GIF / Video Generation (With Top Labels)
        # ---------------------------------------------------------
        if save_video and save_name is not None:
            # Setup Font for Video Header
            try:
                # Try common system fonts for large text
                # font = ImageFont.truetype("arial.ttf", 24)
                font = ImageFont.truetype("Helvetica.ttc", 14)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()

            gif_frames = []
            for t in range(nb_frames):
                
                # Format Prediction Frame for GIF
                p_f = video[t]
                if rescale: p_f = (p_f + 1.0) / 2.0
                
                if not plot_ref and (vmin is not None or vmax is not None):
                    # Manually apply vmin/vmax normalization for the GIF
                    v_min = vmin if vmin is not None else p_f.min()
                    v_max = vmax if vmax is not None else p_f.max()
                    p_f = (p_f - v_min) / (v_max - v_min + 1e-8)
                
                p_f = np.clip(p_f, 0.0, 1.0)
                if p_f.shape[-1] == 1: p_f = np.repeat(p_f, 3, axis=-1)

                if plot_ref:
                    # Format Reference Frame for GIF
                    r_idx = min(t, ref_video.shape[0] - 1)
                    r_f = ref_video[r_idx]
                    if rescale: r_f = (r_f + 1.0) / 2.0
                    r_f = np.clip(r_f, 0.0, 1.0)
                    if r_f.shape[-1] == 1: r_f = np.repeat(r_f, 3, axis=-1)
                    
                    # Create a white spacer block (gap) between the GT and Pred horizontally
                    gap_block = np.ones((r_f.shape[0], video_gap, 3), dtype=r_f.dtype)
                    
                    # Stack them horizontally: GT (left), Gap, Prediction (right)
                    combined_frame = np.concatenate([r_f, gap_block, p_f], axis=1)
                else:
                    combined_frame = p_f

                # Convert float [0, 1] arrays to uint8 [0, 255] PIL Images
                frame_uint8 = (combined_frame * 255).astype(np.uint8)
                base_img = Image.fromarray(frame_uint8)

                # Add Header for text
                header_height = 15
                final_img = Image.new('RGB', (base_img.width, base_img.height + header_height), color='white')
                final_img.paste(base_img, (0, header_height))
                
                # Draw Labels centered above each video stream
                draw = ImageDraw.Draw(final_img)
                if plot_ref:
                    gt_w = r_f.shape[1]
                    pred_w = p_f.shape[1]
                    draw.text((gt_w // 2 - 9, 2), "GT", font=font, fill="black")
                    draw.text((gt_w + video_gap + pred_w // 2 - 15, 2), "Pred", font=font, fill="black")
                else:
                    draw.text((p_f.shape[1] // 2 - 15, 2), "Pred", font=font, fill="black")
                
                gif_frames.append(final_img)
            
            # Save the sequence to GIF matching the save_name
            gif_path = Path(save_name).with_suffix('.gif')
            
            # Export GIF (duration is in ms per frame, 150ms = ~6.6fps)
            gif_frames[0].save(
                gif_path, 
                save_all=True, 
                append_images=gif_frames[1:], 
                duration=150, 
                loop=0
            )
            print(f"Saved rollout animation to {gif_path}")

            # Display the generated GIF inline in the Jupyter Notebook
            try:
                from IPython.display import Image as IPyImage, display
                display(IPyImage(filename=str(gif_path)))
            except ImportError:
                print("IPython is not available to display the GIF inline.")




#%% Cell 3: Model Definition

def fourier_encode(x, num_freqs):
    freqs = 2.0 ** jnp.arange(num_freqs)
    angles = x[..., None] * freqs[None, None, :] * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)

class RootMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

class CNNEncoder(eqx.Module):
    layers: list

    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=8, depth=4):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        conv_layers = []
        current_in = in_channels
        current_out = hidden_width
        
        for i in range(depth):
            conv_layers.append(
                eqx.nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1, key=keys[i])
            )
            current_in = current_out
            current_out *= 2
            
        dummy_x = jnp.zeros((in_channels, H, W))
        for layer in conv_layers:
            dummy_x = layer(dummy_x)

        flat_dim = dummy_x.reshape(-1).shape[0]
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = x.reshape(-1)
        x = self.layers[-1](x)
        return x

class WARP(eqx.Module):
    """
    Weight-space Adaptive Recurrent Predictor
    Uses Linear Recurrence directly on the parameters of the root network.
    """
    encoder: CNNEncoder
    A: jax.Array
    B: jax.Array

    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, key):
        k_root, k_enc, k_A, k_B = jax.random.split(key, 4)
        self.frame_shape = frame_shape
        self.num_freqs = num_freqs
        H, W, C = frame_shape

        coord_dim = 2 + 2 * 2 * num_freqs 
        root_out_dim = C * 2 if CONFIG["use_nll_loss"] else C
        add_time = 1 if CONFIG["use_time_in_root"] else 0
        template_root = RootMLP(coord_dim+add_time, root_out_dim, root_width, root_depth, k_root)
        
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]

        # Encoder serves as the hypernetwork \phi to produce \theta_0 from x_0
        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=64, depth=4)

        # Linear recurrence matrices A & B
        self.A = jnp.eye(self.d_theta)
        self.B = jnp.zeros((self.d_theta, H * W * C))

    def render_pixels(self, theta, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            if CONFIG["use_time_in_root"]:
                encoded_coord = jnp.concatenate([coord[:1], fourier_encode(coord[1:], self.num_freqs)], axis=-1)
            else:
                encoded_coord = fourier_encode(coord[1:], self.num_freqs)
            
            out = root(encoded_coord)
            C = self.frame_shape[2]
            
            # Extract and clip mean if using NLL loss, else clip direct output
            if CONFIG["use_nll_loss"]:
                mean, std = out[:C], out[C:]
                mean = jnp.clip(mean, 0.0, 1.0)
                std = jax.nn.softplus(std) + 1e-4
                return jnp.concatenate([mean, std], axis=-1)
            else:
                return jnp.clip(out, 0.0, 1.0)

        return jax.vmap(render_pt, in_axes=(None, 0))(theta, coords)

    def render_frame(self, theta, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 3)
        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)
    def forward(self, ref_video, coords_grid, context_ratio=1.0, key=None):
        T = ref_video.shape[0]
        H, W, C = self.frame_shape
        
        # 1. Initialize from first frame \theta_0 = \phi(x_0)
        init_frame = ref_video[0]
        theta_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        
        time_coord_0 = jnp.array([1.0/(T-1)], dtype=theta_init.dtype)
        coords_grid_0 = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord_0), coords_grid], axis=-1)
        pred_out_0 = self.render_frame(theta_init, coords_grid_0)

        if key is None:
            key = jax.random.PRNGKey(0)

        # Prepare the initial prediction to act as 'x_next' for the first scan step
        if CONFIG.get("use_nll_loss", False):
            mean_0, std_0 = pred_out_0[..., :C], pred_out_0[..., C:]
            k_sample, key = jax.random.split(key)
            sampled_frame_0 = jnp.clip(mean_0 + std_0 * jax.random.normal(k_sample, mean_0.shape), 0.0, 1.0)
        else:
            sampled_frame_0 = pred_out_0

        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            # x_prev tracks the frame actually fed into the system in the previous step
            # x_pred_prev tracks the model's prediction made in the previous step
            theta_prev, x_prev, x_next, step_key = carry
            x_target, step_idx = scan_inputs
            
            k1, k2, k3 = jax.random.split(step_key, 3)

            if context_ratio is None:
                # Scheduled sampling with 0.5 probability during training
                use_target = jax.random.bernoulli(k1, 0.5)
                x_t = jax.lax.cond(use_target, lambda: x_target, lambda: x_next)
            else:
                # Autoregressive generation vs Teacher Forcing constraint during inference
                is_context = (step_idx / T) < context_ratio
                x_t = jax.lax.cond(is_context, lambda: x_target, lambda: x_next)
            
            # The WARP difference driver \Delta x_t
            delta_x = (x_t - x_prev).reshape(-1)
            
            # Core WARP Linear Recurrence \theta_t = A\theta_{t-1} + B\Delta x_t
            theta_t = jnp.matmul(self.A, theta_prev) + jnp.matmul(self.B, delta_x)

            # Clip the weights so that they don't explode
            theta_t = jnp.clip(theta_t, -0.5, 0.5)
            # theta_t = jnp.clip(theta_t, -1.0, 1.0)

            time_coord = jnp.array([(step_idx + 1)/(T-1)], dtype=theta_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            
            pred_out = self.render_frame(theta_t, coords_grid_t)
            
            if CONFIG.get("use_nll_loss", False):
                mean, std = pred_out[..., :C], pred_out[..., C:]
                noise = jax.random.normal(k2, mean.shape)
                sampled_frame = jnp.clip(mean + std * noise, 0.0, 1.0)
            else:
                sampled_frame = pred_out
                
            # Update carry with the newly selected frame (x_t) and the new prediction (sampled_frame)
            return (theta_t, x_t, sampled_frame, k3), (theta_t, pred_out)

        # Loop drives sequence modeling from 1 -> T-1 predicting x_2 -> x_T
        scan_targets = ref_video[1:-1]
        step_indices = jnp.arange(1, T - 1)
            
        _, (thetas, pred_videos) = jax.lax.scan(
            scan_step, 
            (theta_init, init_frame, sampled_frame_0, key), 
            (scan_targets, step_indices)
        )
        
        # Assemble [T] length predictions to map against full ref_video
        if CONFIG.get("use_nll_loss", False):
            init_frame_nll = jnp.concatenate([init_frame, jnp.zeros_like(init_frame)], axis=-1)
            full_preds = jnp.concatenate([init_frame_nll[None], pred_out_0[None], pred_videos], axis=0)
        else:
            full_preds = jnp.concatenate([init_frame[None], pred_out_0[None], pred_videos], axis=0)
            
        full_thetas = jnp.concatenate([theta_init[None], thetas], axis=0)
        
        return full_thetas, full_preds

#%% Cell 4: Training (WARP)
if TRAIN_PHASE_1:
    print(f"\n🚀 Starting Full WARP Training -> Saving to {run_path}")
    key, subkey = jax.random.split(key)

    model_p1 = WARP(
        root_width=CONFIG["root_width"], 
        root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], 
        frame_shape=(H, W, C), 
        key=subkey
    )
    
    print(f"Total Trainable Parameters in WARP: {count_trainable_params(model_p1)}")
    print(f"  - Number of parameters in dtheta (Root MLP): {model_p1.d_theta}", flush=True)

    optimizer_p1 = optax.chain(
        optax.adam(CONFIG["p1_learning_rate"]),
        optax.contrib.reduce_on_plateau(
            patience=CONFIG["lr_patience"], cooldown=CONFIG["lr_cooldown"],
            factor=CONFIG["lr_factor"], rtol=CONFIG["lr_rtol"],
            accumulation_size=CONFIG["lr_accum_size"], min_scale=CONFIG["lr_min_scale"]
        )
    )
    opt_state_p1 = optimizer_p1.init(eqx.filter(model_p1, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step_p1(model, opt_state, keys, in_videos, coords_grid):
        def loss_fn(m):
            k_aug, k_init = jax.random.split(keys[0], 2)
            ref_videos = in_videos

            if CONFIG["reverse_video_aug"]:
                do_reverse = jax.random.bernoulli(k_aug, 0.5, shape=(ref_videos.shape[0],))
                ref_videos = jax.vmap(lambda rev, vid: jax.lax.cond(rev, 
                                                                       lambda v: jnp.flip(v, axis=0), 
                                                                       lambda v: v, 
                                                                       vid))(do_reverse, ref_videos)

            if CONFIG["static_video_aug"]:
                add_to_front = jax.random.bernoulli(k_init, 0.5, shape=(ref_videos.shape[0],))
                nb_frames = ref_videos.shape[1]
                repeat_frames = nb_frames // 4
                ref_videos = jax.vmap(lambda add_front, vid: jax.lax.cond(add_front, 
                                                                            lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
                                                                            lambda v_in: jnp.concatenate([v_in[:nb_frames-repeat_frames], jnp.repeat(v_in[nb_frames-repeat_frames:nb_frames-repeat_frames+1], repeats=repeat_frames, axis=0)], axis=0),
                                                                                vid))(add_to_front, ref_videos)

            batch_keys = jax.random.split(keys[0], ref_videos.shape[0])
            # Fully Teacher-Forced (context_ratio = 1.0)
            _, pred_videos = jax.vmap(m.forward, in_axes=(0, None, None, 0))(ref_videos, coords_grid, 1.0, batch_keys)

            # Evaluate predictions starting from t=1 (discard the padded initial frame)
            targets = ref_videos[:, 1:]
            preds = pred_videos[:, 1:]

            if CONFIG["use_nll_loss"]:
                mean, std = preds[..., :C], preds[..., C:]
                total_loss = jnp.mean(0.5 * ((mean - targets) / std) ** 2 + jnp.log(std))
            else:
                total_loss = jnp.mean((preds - targets)**2)

            return total_loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer_p1.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    all_losses_p1 = []
    lr_scales_p1 = []
    start_time = time.time()

    sample_videos_vis = next(iter(train_loader))[:1]

    for epoch in range(CONFIG["p1_nb_epochs"]):
        epoch_losses = []
        for batch_idx, batch_videos in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch_videos.shape[0])
            model_p1, opt_state_p1, loss = train_step_p1(model_p1, opt_state_p1, batch_keys, batch_videos, coords_grid)
            epoch_losses.append(loss)
            lr_scales_p1.append(optax.tree_utils.tree_get(opt_state_p1, "scale"))

        all_losses_p1.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]):
            avg_loss = np.mean(epoch_losses)
            print(f"Training - Epoch {epoch+1}/{CONFIG['p1_nb_epochs']} - Avg Loss: {avg_loss:.6f}", flush=True)

        if (epoch+1) % (CONFIG["p1_nb_epochs"]//2) == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]:
            eqx.tree_serialise_leaves(artefacts_path / f"model_phase1_epoch{epoch+1}.eqx", model_p1)

        if (epoch+1) % (CONFIG["p1_nb_epochs"]//10) == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]:
            dummy_keys = jax.random.split(key, sample_videos_vis.shape[0])
            _, pred_videos = jax.vmap(model_p1.forward, in_axes=(0, None, None, 0))(sample_videos_vis, coords_grid, 1.0, dummy_keys)
            for i in range(pred_videos.shape[0]):
                # Plot the sequence excluding the forced t=0 frame aligning predictions
                plot_pred_ref_videos_rollout(pred_videos[i, 1:], sample_videos_vis[i, 1:], epoch+1, plots_path / f"p1_vis_epoch{epoch+1}_sample{i}.png")

    print("\nTraining Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    model_final = model_p1
    eqx.tree_serialise_leaves(artefacts_path / "model_phase1_final.eqx", model_p1)
    np.save(artefacts_path / "loss_history_p1.npy", np.array(all_losses_p1))
    np.save(artefacts_path / "lr_history_p1.npy", np.array(lr_scales_p1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(all_losses_p1, color='teal', alpha=0.8, label="Training Loss")
    ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color='teal')
    ax2 = ax1.twinx()  
    ax2.plot(lr_scales_p1, color='crimson', linewidth=2, label="LR Scale")
    plt.title("WARP Model Training Loss")
    fig.tight_layout()
    plt.savefig(plots_path / "p1_loss_history.png")
    plt.show()

#%% Cell 5: Phase 2 Training (GCM Matching)
print("\n=== Phase 2 is not needed for WARP as it utilizes a pure input-difference linear recurrence. ===")

#%% Cell 6: Evaluation & Plotting
print("\n=== Evaluating Model ===")

if not TRAIN_PHASE_1:
    print(f"📥 Loading completed WARP model from {artefacts_path}")
    key, subkey = jax.random.split(key)
    model_final = WARP(
        root_width=CONFIG["root_width"], 
        root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], 
        frame_shape=(H, W, C), 
        key=subkey
    )
    model_final = eqx.tree_deserialise_leaves(artefacts_path / "model_phase1_final.eqx", model_final)

    print(f"  - Number of paramters in root (dtheta): {model_final.d_theta}", flush=True)

@eqx.filter_jit
def evaluate(m, batch, coords, context_ratio, prng_key):
    batch_keys = jax.random.split(prng_key, batch.shape[0])
    batched_fn = jax.vmap(m.forward, in_axes=(0, None, None, 0))
    return batched_fn(batch, coords, context_ratio, batch_keys)

testing_subset = MovingMNISTDataset(test_arrays)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"]*1, shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

#%%
key, subkey = jax.random.split(key)
final_thetas, final_videos = evaluate(model_final, sample_batch, coords_grid, 0.5, subkey)

#%%
test_seq_id = np.random.randint(0, sample_batch.shape[0])
print(f"\nGenerating final forecast rollout visualization for test sequence ID: {test_seq_id}")

plot_videos(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id], 
    show_titles=True,
    show_labels=True,
    plot_ref=True,
    forecast_start=11,
    forecast_gap=0.08,
    hspace=0.02,
    wspace=0.02,
    save_name=plots_path / f"warp_inference_fs11_seq{test_seq_id}.pdf",
    save_video=False,
)

#%% Gradually decreasing the context ratio
real_id = 70
key, subkey = jax.random.split(key)
final_thetas, final_videos = evaluate(model_final, sample_batch[real_id:real_id+1], coords_grid, 21/20, subkey)
test_seq_id = 0

plot_videos(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id+real_id], 
    show_titles=True,
    show_labels=True,
    plot_ref=True,
    forecast_start=21,
    forecast_gap=0.2,
    hspace=0.02,
    wspace=0.02,
    save_name=plots_path / f"inference_gradual_fs21_seq{real_id}.pdf",
    save_video=True,
)

#%% Morphing digits from 8 to 9

corrupt_seq_id = 54
# corrupt_seq_id = 40
print(f"\nGenerating morphing visualization for corrupt test sequence ID: {corrupt_seq_id}")

@eqx.filter_jit
def get_latent_representation(model, video):
    init_frame = video[0]
    return model.encoder(jnp.transpose(init_frame, (2, 0, 1)))

latent_first = get_latent_representation(model_final, sample_batch[corrupt_seq_id])

print(f"Latent representation of the first frame in the first test sequence: {latent_first.shape}")

@eqx.filter_jit
def decode_latent(model, latent):
    time_coord = jnp.array([(1-1)/(20-1)])
    coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
    return model.render_frame(latent, coords_grid_t)

decoded_frame = decode_latent(model_final, latent_first)
if CONFIG["use_nll_loss"]: decoded_frame = decoded_frame[..., :C]
plt.imshow(np.clip(decoded_frame, 0.0, 1.0))
# plt.imshow(sample_batch[corrupt_seq_id][0])
plt.title("Decoded Frame from Latent Representation")
plt.axis('off')
plt.draw()
@eqx.filter_jit
def inference_rollout_morph(model, ref_video, coords_grid, context_ratio=0.0, corrupt=True, theta_corrupt=None, prng_key=None):
    T = ref_video.shape[0]
    H, W, C = model.frame_shape
    
    # 1. Initialize from first frame \theta_0 = \phi(x_0)
    init_frame = ref_video[0]
    theta_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    
    time_coord_0 = jnp.array([1.0/(T-1)], dtype=theta_init.dtype)
    coords_grid_0 = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord_0), coords_grid], axis=-1)
    pred_out_0 = model.render_frame(theta_init, coords_grid_0)

    if prng_key is None: 
        prng_key = jax.random.PRNGKey(0)

    # Prepare the initial prediction to act as 'x_next' for the first scan step
    if CONFIG.get("use_nll_loss", False):
        mean_0, std_0 = pred_out_0[..., :C], pred_out_0[..., C:]
        k_sample, prng_key = jax.random.split(prng_key)
        sampled_frame_0 = jnp.clip(mean_0 + std_0 * jax.random.normal(k_sample, mean_0.shape), 0.0, 1.0)
    else:
        sampled_frame_0 = pred_out_0

    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        # x_prev tracks the frame actually fed into the system in the previous step
        # x_pred_prev tracks the model's prediction made in the previous step
        theta_prev, x_prev, x_next, step_key = carry
        x_target, step_idx = scan_inputs

        is_context = (step_idx / T) < context_ratio
        x_t = jax.lax.cond(is_context, lambda: x_target, lambda: x_next)

        delta_x = (x_t - x_prev).reshape(-1)
        theta_t = jnp.matmul(model.A, theta_prev) + jnp.matmul(model.B, delta_x)

        # Clip the weights so that they don't explode
        theta_t = jnp.clip(theta_t, -0.5, 0.5)
        # theta_t = jnp.clip(theta_t, -1.0, 1.0)

        if corrupt and theta_corrupt is not None:
            theta_t = jax.lax.cond(step_idx == 5, lambda: theta_corrupt, lambda: theta_t)

        # time_coord = jnp.array([(step_idx + 1)/(T-1)], dtype=theta_t.dtype)
        time_coord = jnp.array([0/(T-1)], dtype=theta_t.dtype)
        coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)

        pred_out = model.render_frame(theta_t, coords_grid_t)

        ## Let's use decode latent instead
        # pred_out = decode_latent(model, theta_t)

        if CONFIG.get("use_nll_loss", False):
            mean, std = pred_out[..., :C], pred_out[..., C:]
            k1, k2 = jax.random.split(step_key)
            pred_frame = jnp.clip(mean + std * jax.random.normal(k1, mean.shape), 0.0, 1.0)
            next_key = k2
        else:
            pred_frame = pred_out
            next_key = step_key

        # Update carry with the newly selected frame (x_t) and the new prediction (pred_frame)
        return (theta_t, x_t, pred_frame, next_key), (theta_t, pred_out)

    scan_targets = ref_video[1:-1]
    step_indices = jnp.arange(1, T - 1)
    
    _, (thetas, pred_videos) = jax.lax.scan(
        scan_step, 
        (theta_init, init_frame, sampled_frame_0, prng_key), 
        (scan_targets, step_indices)
    )

    if CONFIG.get("use_nll_loss", False):
        init_frame_nll = jnp.concatenate([init_frame, jnp.zeros_like(init_frame)], axis=-1)
        full_preds = jnp.concatenate([init_frame_nll[None], pred_out_0[None], pred_videos], axis=0)
    else:
        # full_preds = jnp.concatenate([init_frame[None], pred_out_0[None], pred_videos], axis=0)
        full_preds = jnp.concatenate([pred_out_0[None], pred_videos], axis=0)

    return full_preds

test_seq_id = 57
print(f"\nGenerating morphing visualization for test sequence ID: {test_seq_id} (Digit 8 morphing into 9)")
key, k_corrupt, k_clean = jax.random.split(key, 3)

final_videos_corrupt = inference_rollout_morph(model_final, sample_batch[test_seq_id], coords_grid, 3/20, corrupt=True, theta_corrupt=latent_first, prng_key=k_corrupt)
final_videos_clean = inference_rollout_morph(model_final, sample_batch[test_seq_id], coords_grid, 3/20, corrupt=False, prng_key=k_clean)

plot_videos(
    final_videos_corrupt[:10], 
    final_videos_clean[:10], 
    # show_titles=False,
    # show_labels=False,
    plot_ref=True,
    forecast_start=1,
    forecast_gap=0.2,
    hspace=0.02,
    wspace=0.02,
    # vmin=0,
    # vmax=1,
    save_name=plots_path / f"inference_control_seq{test_seq_id}.pdf",
    save_video=True,
    cmap="gray",
)

np.savez(artefacts_path / f"warp_currupt.npz", 
         corrupt_pred_video=final_videos_corrupt,
         clean_pred_video=final_videos_clean,
         ref_video=sample_batch[test_seq_id],
         corrupt_latent=latent_first,
         corrupt_frame_pred=decoded_frame,
         corrupt_frame_ref=sample_batch[corrupt_seq_id, 0],
         )



## Plot first frame of gt sequence sample_batch[test_seq_id][0] and the decoded frame from the corrupted latent representation side by side for comparison. 
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)

# --- Add Purple Border to Context Frame ---
from PIL import Image, ImageDraw
import numpy as np

# Convert single-channel to 3-channel RGB for coloring the border
context_frame_rgb = np.repeat(sample_batch[test_seq_id][0], 3, axis=-1)
context_img = Image.fromarray((np.clip(context_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8))
border_size = 2
border_color_purple = (128, 0, 128)  # Purple
draw_context = ImageDraw.Draw(context_img)
for i in range(border_size):
    draw_context.rectangle([i, i, context_img.width - i - 1, context_img.height - i - 1], outline=border_color_purple)

# plt.imshow(context_img, cmap="gray") # cmap is ignored when RGB is passed
plt.imshow(context_img) 
plt.title("First Context Frame", fontsize=20)
plt.axis('off') 

plt.subplot(1, 2, 2)

# --- Add Red Border to Alien Frame ---
# Convert single-channel to 3-channel RGB for coloring the border
decoded_frame_rgb = np.repeat(sample_batch[corrupt_seq_id][0], 3, axis=-1) 
decoded_img = Image.fromarray((np.clip(decoded_frame_rgb, 0.0, 1.0) * 255).astype(np.uint8))
border_color_red = (255, 0, 0)  # Red
draw_decoded = ImageDraw.Draw(decoded_img)
for i in range(border_size):
    draw_decoded.rectangle([i, i, decoded_img.width - i - 1, decoded_img.height - i - 1], outline=border_color_red)

plt.imshow(decoded_img, cmap="gray") # cmap is ignored when RGB is passed
# plt.imshow(sample_batch[corrupt_seq_id][0], cmap="gray")

plt.title("Alien Frame", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.savefig(plots_path / f"decoded_corrupted_latent_seq{test_seq_id}.pdf")
plt.show()



















#%% 1. Weight Variance (Finding the Active Dimensions)
# Modifying to observe Weight (Theta) Variance rather than Actions, as Actions no longer exist in WARP.
all_thetas_flat = final_thetas.reshape(-1, model_final.d_theta)
theta_variances = np.var(all_thetas_flat, axis=0)

plt.figure(figsize=(10, 4))
plt.bar(range(model_final.d_theta), theta_variances, color='teal')
plt.xlabel("Theta Weight Dimension")
plt.ylabel("Variance across all data")
plt.title("Weight Dimension Importance (Variance)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "weight_dimension_variance_mnist.png")
plt.show()

top_dims = np.argsort(theta_variances)[-4:][::-1]
print(f"Top 4 most active weight dimensions: {top_dims}")

#%% 2. Continuous Weight Trajectories over Time
seq_thetas = final_thetas[test_seq_id] # Shape: (T, d_theta)
T_steps = seq_thetas.shape[0]

plt.figure(figsize=(12, 6))
colors = ['crimson', 'dodgerblue', 'forestgreen', 'darkorange']

for i, dim in enumerate(top_dims):
    plt.plot(range(T_steps), seq_thetas[:, dim], marker='o', linewidth=2, 
             color=colors[i], label=f"Dim {dim} (Rank {i+1})")

plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel("Time Step (t)")
plt.ylabel("Theta Space Values")
plt.title(f"Weight Evolution for Sequence {test_seq_id}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / f"theta_lines_seq{test_seq_id}.png")
plt.show()

# %% Save nohup
os.system(f"cp -r nohup.log {run_path}/nohup.log")

#%%
@eqx.filter_jit
def log_horizon_forecast(model, ref_video, coords_grid, context_ratio=2/20, prng_key=None):
    # Same autoregressive logic utilizing linear recurrence natively supported by model.forward
    _, pred_videos = model.forward(ref_video, coords_grid, context_ratio=context_ratio, key=prng_key)
    return pred_videos

# Select one sequence of 8 morphing into 9
test_seq_id = 54
total_length = 1000

print(f"\nForecasting a long horizon rollout for test sequence ID: {test_seq_id} with total length {total_length} frames")

input_video = sample_batch[test_seq_id]

## Pad sequence to total_length with zeros (or repeat last frame)
input_video = jnp.concatenate([input_video, jnp.zeros((total_length - input_video.shape[0], H, W, C))], axis=0)
key, subkey = jax.random.split(key)

output_video = log_horizon_forecast(model_final, input_video, coords_grid, 2/20, prng_key=subkey)

plot_videos(
    output_video[:50], 
    show_titles=False,
    show_labels=False,
    plot_ref=False,
    forecast_start=2,
    forecast_gap=0.2,
    hspace=0.02,
    wspace=0.02,
    save_video=False,
)

np.save(artefacts_path / f"warp_long_horizon_ID{test_seq_id}_T{total_length}.npy", output_video)

#%% Cell 7: Calculating Spatio-Temporal Metrics Across Test Set
import time
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

print("\n=== Calculating Spatio-Temporal Error Metrics over Full Test Set ===")

# --- Configuration & Setup ---
num_context_frames = 10  # Standard T_in for Moving MNIST
context_ratio_for_eval = float(num_context_frames) / 20.0

mse_list = []
mae_list = []
psnr_list = []
ssim_list = []

print(f"Evaluating {len(testing_subset)} test sequences with T_context = {num_context_frames}")
start_time = time.time()

# Loop through the entire test dataloader
for batch_idx, batch_videos in enumerate(tqdm(test_loader, desc="Testing set evaluation")):
    
    # 1. Run inference
    key, subkey = jax.random.split(key)
    _, final_videos = evaluate(model_final, batch_videos, coords_grid, context_ratio_for_eval, subkey)
    
    # Convert Jax arrays to Numpy
    pred_np = np.array(final_videos)
    target_np = np.array(batch_videos)
    
    # Iterate over every video in the current batch
    for b in range(target_np.shape[0]):
        y_true_future = target_np[b, num_context_frames:]*1
        y_pred_future = pred_np[b, num_context_frames:]*1
        
        squared_err = (y_true_future - y_pred_future) ** 2
        abs_err = np.abs(y_true_future - y_pred_future)
        
        seq_mse_per_frame = np.sum(squared_err, axis=(1, 2, 3))
        seq_mae_per_frame = np.sum(abs_err, axis=(1, 2, 3))
        
        mse_list.append(np.mean(seq_mse_per_frame))
        mae_list.append(np.mean(seq_mae_per_frame))
        
        seq_psnr = []
        seq_ssim = []
        
        for t in range(y_true_future.shape[0]):
            s = ssim(y_true_future[t], y_pred_future[t], data_range=1.0, channel_axis=-1)
            seq_ssim.append(s)
            
            mse_pixel = np.mean(squared_err[t])
            mse_pixel = max(mse_pixel, 1e-10) 
            p = 10 * np.log10(1.0 / mse_pixel)
            seq_psnr.append(p)
            
        psnr_list.append(np.mean(seq_psnr))
        ssim_list.append(np.mean(seq_ssim))

# --- Aggregate and Print Results ---
final_mse = np.mean(mse_list)
final_mae = np.mean(mae_list)
final_rmse = np.sqrt(final_mse)
final_psnr = np.mean(psnr_list)
final_ssim = np.mean(ssim_list)

print("\n" + "-"*54)
print("  Test Set Results (Averaged across all sequences)")
print(f"  Context Frames: {num_context_frames}")
print("-" * 54)
print(f"  Mean Squared Error (MSE):          {final_mse:.4f}")
print(f"  Mean Absolute Error (MAE):         {final_mae:.4f}")
print(f"  Root Mean Squared Error (RMSE):    {final_rmse:.4f}")
print(f"  Peak Signal-to-Noise Ratio (PSNR): {final_psnr:.4f}")
print(f"  Structural Similarity (SSIM):      {final_ssim:.4f}")
print("-" * 54)
print(f"Evaluation took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")