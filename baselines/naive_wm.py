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

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

## Massive warning: Phase 0 correspond to the usual Phase 1. Phase 1 to Phase 2, etc. 
print("\n=== WARNING: PHASE 0 IS THE USUAL ENCODER PRETRAINING, PHASE 1 IS THE USUAL IDM/FDM TRAINING, AND PHASE 2 IS THE USUAL GCM TRAINING! \n I'M SORRY FOR THE CONFUSION 😔 ===\n")


def count_trainable_params(model):
    def count_params(x):
        if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float32, jnp.float64]:
            return x.size
        return 0
    param_counts = jax.tree_util.tree_map(count_params, model)
    return sum(jax.tree_util.tree_leaves(param_counts))

# --- Configuration ---
TRAIN_PHASE_0 = False  # Optional pretraining of Encoder + Base Theta via autoencoding
TRAIN_PHASE_1 = True  # Train IDM, FDM, and maybe (Encoder, Base Theta)
TRAIN_PHASE_2 = True  # Train GCM (Memory Model) to match IDM
RUN_DIR = "./" if (not TRAIN_PHASE_1 or not TRAIN_PHASE_2) else None

SINGLE_BATCH = False
USE_NLL_LOSS = False

CONFIG = {
    "seed": 42,
    
    # Phase 1 Params
    "p1_nb_epochs": 1500,        
    "p1_learning_rate": 1e-4 if USE_NLL_LOSS else 1e-4,
    "reverse_video_aug": True,
    "static_video_aug": True,
    "action_l1_reg": 0.00,     # L1 regularisation on continuous actions to limit info capacity
    "mse_weight": 0.0,        # Option to blend MSE into SSIM
    "aux_encoder_loss": False,
    "aux_loss_weight": 1.0,
    "aux_loss_num_steps": 4,

    # Phase 2 Params
    # "p2_nb_epochs": 1500,
    "p2_nb_epochs": 1000,
    "p2_learning_rate": 1e-4,

    "print_every": 10,
    "batch_size": 2 if SINGLE_BATCH else 128*2,
    "inf_context_ratio": 0.5,
    "use_nll_loss": USE_NLL_LOSS,

    # --- Architecture Params ---
    "lam_space": 4,
    "mem_space": 256,
    "icl_decoding": True,
    "discrete_actions": False,
    "split_forward": True,
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
    if TRAIN_PHASE_1 or TRAIN_PHASE_2:
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
        return Path(RUN_DIR)

run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"
os.makedirs(artefacts_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

# Path to explicitly load Phase 1 weights if Phase 1 is skipped
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
    data_path = '../data' if (TRAIN_PHASE_1 or TRAIN_PHASE_2) else '../../data'

    # dataset = datasets.MovingMNIST(root=data_path, split=None, download=True)
    # train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=numpy_collate, drop_last=True)

    ## Manually load train and test splits to have more control over batching and shuffling
    mov_mnist_arrays = np.load(data_path + "/MovingMNIST/mnist_test_seq.npy")
    print(f"Original loaded MovingMNIST shape: {mov_mnist_arrays.shape} (T, N, H, W)")
    ## Split, 8000 train, 2000 test
    train_arrays = mov_mnist_arrays[:, :8000]
    test_arrays = mov_mnist_arrays[:, 8000:]

    ## Create PyTorch dataset
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
                                  drop_last=False,
                                  num_workers=8)

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

#%%
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
            # Handle offsets for plotting ref against predicted properly (sometimes diff by 1)
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



def plot_videos(video, ref_video=None, plot_ref=True, show_titles=True, show_labels=True, forecast_start=None, 
                                 vmin=None, vmax=None, save_name=None, 
                                 wspace=0.05, hspace=0.05, forecast_gap=0.2, 
                                 save_video=False, video_gap=5):
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
                axes[1, 0].set_ylabel("Pred", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
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




#%% Visualize Augmentations
print("\n=== Visualizing Video Augmentations ===")

# 1. Grab a single video and duplicate it to make a test batch of 4 identical videos
test_vid = sample_batch[0:1] # Shape: (1, T, H, W, C)
test_batch = jnp.repeat(test_vid, 4, axis=0)

# 2. Force deterministic boolean arrays to test every combination
# Row 0: Original
# Row 1: Reversed
# Row 2: Static Front
# Row 3: Static Back
do_reverse = jnp.array([False, True, False, False])
do_static  = jnp.array([False, False, True, True])
add_to_front = jnp.array([False, False, True, False]) # Only matters if do_static is True

# --- Apply Reverse Augmentation ---
aug_batch = jax.vmap(lambda rev, vid: jax.lax.cond(
    rev, 
    lambda v: jnp.flip(v, axis=0), 
    lambda v: v, 
    vid
))(do_reverse, test_batch)

# --- Apply Static Augmentation (With Bug Fix) ---
nb_frames = aug_batch.shape[1]
repeat_frames = nb_frames // 4

def static_aug(add_front, v):
    return jax.lax.cond(
        add_front, 
        # FIX: Take the remaining frames from the START of the sequence, not the end
        lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
        # False branch was already correct
        lambda v_in: jnp.concatenate([v_in[:nb_frames-repeat_frames], jnp.repeat(v_in[nb_frames-repeat_frames:nb_frames-repeat_frames+1], repeats=repeat_frames, axis=0)], axis=0),
        v
    )

aug_batch = jax.vmap(lambda apply_stat, add_front, vid: jax.lax.cond(
    apply_stat,
    lambda v: static_aug(add_front, v),
    lambda v: v,
    vid
))(do_static, add_to_front, aug_batch)

# --- Plot the Results ---
fig, axes = plt.subplots(4, nb_frames, figsize=(nb_frames * 1.5, 4 * 1.5))
row_titles = ["Original", "Reversed", "Static (Front)", "Static (Back)"]

for row in range(4):
    for t in range(nb_frames):
        ax = axes[row, t]
        img = aug_batch[row, t]
        
        # Handle 1-channel grayscale for RGB imshow
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
            
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row and Column labels
        if t == 0:
            ax.set_ylabel(row_titles[row], fontsize=14, rotation=0, labelpad=60, ha='center', va='center', fontweight='bold')
        if row == 0:
            ax.set_title(f"t={t}")

plt.suptitle("Effect of Temporal Video Augmentations", y=1.00, fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()


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

class CNNDecoder(eqx.Module):
    linear: eqx.nn.Linear
    layers: list
    pre_flat_shape: tuple
    out_channels: int = eqx.field(static=True)

    def __init__(self, in_dim, out_channels, spatial_shape, key, hidden_width=64, depth=4):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        self.out_channels = out_channels * 2 if CONFIG.get("use_nll_loss", False) else out_channels
        
        # 1. Forward trace exactly like the encoder to get the spatial shapes
        current_in = out_channels
        current_out = hidden_width
        dummy_x = jnp.zeros((current_in, H, W))
        
        channel_progression = []
        for i in range(depth):
            dummy_layer = eqx.nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1, key=keys[0])
            dummy_x = dummy_layer(dummy_x)
            channel_progression.append(current_out)
            current_in = current_out
            current_out *= 2
            
        self.pre_flat_shape = dummy_x.shape
        flat_dim = dummy_x.reshape(-1).shape[0]
        
        # 2. Build mapping from Latent -> Flattened Spatial
        self.linear = eqx.nn.Linear(in_dim, flat_dim, key=keys[0])
        
        # 3. Build Transposed Convolutions
        deconv_layers = []
        c_in = self.pre_flat_shape[0]
        
        for i in range(depth):
            # Reverse the channel sequence
            c_out = channel_progression[depth - 2 - i] if i < depth - 1 else self.out_channels
            deconv_layers.append(
                eqx.nn.ConvTranspose2d(
                    c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[i+1]
                )
            )
            c_in = c_out
        
        self.layers = deconv_layers

    def __call__(self, x):
        x = self.linear(x)
        x = jax.nn.relu(x)
        x = x.reshape(self.pre_flat_shape)
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
            
        # Final layer linear output (loss function or render_frame handles NLL / clipping)
        x = self.layers[-1](x)
        return x

class ForwardDynamics(eqx.Module):
    mlp_A: Optional[eqx.nn.MLP]
    mlp_B: Optional[eqx.nn.MLP]
    giant_mlp: Optional[eqx.nn.MLP]
    split_forward: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, lam_dim, split_forward, key):
        self.split_forward = split_forward
        k1, k2, k3 = jax.random.split(key, 3)
        if split_forward:
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k1)
            self.mlp_B = eqx.nn.MLP(lam_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k2)
            self.giant_mlp = None
        else:
            self.mlp_A = None
            self.mlp_B = None
            self.giant_mlp = eqx.nn.MLP(dyn_dim + lam_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k3)

    def __call__(self, z_prev, a):
        if self.split_forward:
            return self.mlp_A(z_prev) + self.mlp_B(a)
            # return self.mlp_A(z_prev)
            # return self.mlp_B(a)
        else:
            return self.giant_mlp(jnp.concatenate([z_prev, a], axis=-1))

class TransformerBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads, key):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k1
        )
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model * 4, depth=1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask):
        x_norm = jax.vmap(self.ln1)(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x

class InverseDynamics(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key, num_actions=None):
        if num_actions: ## Discrete case
            self.mlp = eqx.nn.MLP(dyn_dim * 2, num_actions, width_size=dyn_dim*1, depth=2, key=key)
        else:
            self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=2, key=key)
        
    def __call__(self, z_prev, z_target):
        return self.mlp(jnp.concatenate([z_prev, z_target], axis=-1))

class MemoryModuleAtt(eqx.Module):
    """
    Autoregressive Transformer Memory Module for Latent Actions (GCM).
    """
    d_model: int
    max_len: int
    pos_emb: jax.Array
    blocks: tuple
    proj_in: eqx.nn.Linear
    
    lam_dim: int = eqx.field(static=True)
    icl_decoding: bool = eqx.field(static=True)
    
    action_mlp: Optional[eqx.nn.MLP]
    output_proj: Optional[eqx.nn.Linear]

    def __init__(self, lam_dim, mem_dim, latent_dim, key, max_len=20, num_heads=4, num_blocks=4, num_actions=4):
        self.max_len = max_len
        self.icl_decoding = CONFIG["icl_decoding"]
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        
        self.proj_in = eqx.nn.Linear(latent_dim + lam_dim, self.d_model, key=k1)
        self.pos_emb = jax.random.normal(k2, (max_len, self.d_model)) * 0.02
        
        block_keys = jax.random.split(k3, num_blocks)
        self.blocks = tuple(TransformerBlock(self.d_model, num_heads, bk) for bk in block_keys)

        if self.icl_decoding:
            self.action_mlp = None
            if num_actions:
                self.output_proj = eqx.nn.Linear(self.d_model, num_actions, key=k6)
            else:
                self.output_proj = eqx.nn.Linear(self.d_model, lam_dim, key=k6)
        else:
            self.action_mlp = eqx.nn.MLP(self.d_model + latent_dim, lam_dim, width_size=self.d_model * 2, depth=3, key=k4)
            self.output_proj = None

    def reset(self, T):
        return jnp.zeros((T, self.d_model))

    def encode(self, buffer, step_idx, z, a):
        token = self.proj_in(jnp.concatenate([z, a], axis=-1))
        return buffer.at[step_idx - 1].set(token)

    def decode(self, buffer, step_idx, z_current):
        T = buffer.shape[0]
        if self.icl_decoding:
            zero_action = jnp.zeros((self.lam_dim,), dtype=z_current.dtype)
            query_token = self.proj_in(jnp.concatenate([z_current, zero_action], axis=-1))
            temp_buffer = buffer.at[step_idx - 1].set(query_token)
            
            x = temp_buffer + self.pos_emb[:T]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            
            for block in self.blocks:
                x = block(x, mask)
            context = x[step_idx - 1]
            return self.output_proj(context)
        else:
            def compute_context():
                x = buffer + self.pos_emb[:T]
                mask = jnp.tril(jnp.ones((T, T), dtype=bool))
                for block in self.blocks:
                    x = block(x, mask)
                return x[step_idx - 2]
                
            context = jax.lax.cond(step_idx > 1, compute_context, lambda: jnp.zeros(self.d_model))
            return self.action_mlp(jnp.concatenate([context, z_current], axis=-1))


class VanillaRNNCell(eqx.Module):
    """A standard Elman RNN cell for lightweight baselining."""
    weight_ih: eqx.nn.Linear
    weight_hh: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        # Bias is only needed once, so we include it in the input projection
        self.weight_ih = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=k1)
        self.weight_hh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k2)

    def __call__(self, input: jax.Array, hidden: jax.Array) -> jax.Array:
        # Standard RNN formulation: tanh(W_x * x + W_h * h + b)
        return jax.nn.tanh(self.weight_ih(input) + self.weight_hh(hidden))

class MemoryModule(eqx.Module):
    """
    Recurrent Memory Module for Latent Actions.
    Uses either a Vanilla RNN, GRU, or LSTM as the core memory architecture.
    Supports both continuous (lam_dim) and discrete (num_actions) output spaces.
    """
    d_model: int
    rnn_type: str = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    num_actions: Optional[int] = eqx.field(static=True)
    
    # Core recurrent cell
    rnn_cell: eqx.Module
    
    # Decoder component
    action_decoder: eqx.nn.MLP

    def __init__(self, lam_dim, mem_dim, latent_dim, key, rnn_type="GRU", num_actions=None, **kwargs):
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        self.rnn_type = rnn_type.upper()
        self.num_actions = num_actions
        
        k1, k2 = jax.random.split(key, 2)
        
        # 1. Recurrent Cell Initialization
        input_dim = latent_dim + lam_dim
        if self.rnn_type == "LSTM":
            self.rnn_cell = eqx.nn.LSTMCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "GRU":
            self.rnn_cell = eqx.nn.GRUCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "RNN":
            self.rnn_cell = VanillaRNNCell(input_dim, self.d_model, key=k1)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Must be 'LSTM', 'GRU', or 'RNN'.")

        # 2. Decoder Initialization
        decode_input_dim = self.d_model + latent_dim
        
        # Route output dimension: discrete (logits) vs continuous (vector)
        out_dim = num_actions if num_actions is not None else lam_dim
        
        self.action_decoder = eqx.nn.MLP(
            in_size=decode_input_dim, 
            out_size=out_dim, 
            width_size=self.d_model * 1, 
            depth=1, 
            key=k2
        )

    def reset(self, T):
        """
        Returns the initial hidden state(s) for the RNN.
        The 'T' argument is kept strictly to maintain the API.
        """
        if self.rnn_type == "LSTM":
            return (jnp.zeros((self.d_model,)), jnp.zeros((self.d_model,)))
        else:
            # Both GRU and Vanilla RNN use a single hidden state vector
            return jnp.zeros((self.d_model,))

    def encode(self, state, step_idx, z, a):
        """
        Takes the current RNN state, concatenates [z, a], and steps the RNN forward.
        """
        rnn_input = jnp.concatenate([z, a], axis=-1)
        new_state = self.rnn_cell(rnn_input, state)
        return new_state

    def decode(self, state, step_idx, z_current):
        """
        Extracts the hidden memory, concatenates it with the current observation, 
        and predicts the action via MLP.
        """
        if self.rnn_type == "LSTM":
            h = state[0]
        else:
            h = state
            
        decode_input = jnp.concatenate([h, z_current], axis=-1)
        return self.action_decoder(decode_input)


class LAM(eqx.Module):
    """ Action Model holding both the IDM (Phase 1) and GCM (Phase 2). """
    idm: InverseDynamics
    gcm: Optional[MemoryModuleAtt]
    discrete_actions: bool = eqx.field(static=True)
    action_embedding: Optional[eqx.nn.Embedding]

    def __init__(self, dyn_dim, lam_dim, mem_dim, max_len, num_heads, num_blocks, num_actions, key, phase=1):
        k1, k2 = jax.random.split(key)
        self.discrete_actions = num_actions is not None

        self.idm = InverseDynamics(dyn_dim, lam_dim, key=k1, num_actions=num_actions if self.discrete_actions else None)
        
        # Only instantiate GCM in Phase 2
        if phase == 2:
            # self.gcm = MemoryModuleAtt(lam_dim, mem_dim, dyn_dim, key=k2, num_heads=num_heads, num_blocks=num_blocks, max_len=max_len, num_actions=num_actions if self.discrete_actions else None)
            self.gcm = MemoryModule(lam_dim, mem_dim, dyn_dim, key=k2, rnn_type="GRU", num_actions=num_actions if self.discrete_actions else None)
        else:
            self.gcm = None

        if self.discrete_actions:
            self.action_embedding = eqx.nn.Embedding(num_actions, lam_dim, key=k2)
        else:
            self.action_embedding = None

    def discretise_action(self, logits):
        soft_probs = jax.nn.softmax(logits, axis=-1)
        hard_idx = jnp.argmax(logits, axis=-1)
        hard_probs = jax.nn.one_hot(hard_idx, num_classes=logits.shape[-1])
        ste_probs = soft_probs + jax.lax.stop_gradient(hard_probs - soft_probs)
        action = jnp.dot(ste_probs, self.action_embedding.weight)
        return action

    def inverse_dynamics(self, z_prev, z_target):
        if not self.discrete_actions:
            return self.idm(z_prev, z_target)
        else:
            logits = self.idm(z_prev, z_target)
            return self.discretise_action(logits)

    def decode_memory(self, buffer, step_idx, z_current):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        if not self.discrete_actions:
            return self.gcm.decode(buffer, step_idx, z_current)
        else:
            logits = self.gcm.decode(buffer, step_idx, z_current)
            return self.discretise_action(logits)

    def encode_memory(self, buffer, step_idx, z_current, a):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        return self.gcm.encode(buffer, step_idx, z_current, a)
    
    def reset_memory(self, T):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        return self.gcm.reset(T)

class WARP(eqx.Module):
    encoder: CNNEncoder
    forward_dyn: ForwardDynamics
    decoder: CNNDecoder  # Repurposed to hold the decoder PyTree!
    action_model: LAM

    d_theta: int = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    split_forward: bool = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    mem_dim: int = eqx.field(static=True)
    phase: int = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, lam_dim, mem_dim, split_forward, key, phase=1):
        k_root, k_enc, k_lam, k_fwd, k_mem = jax.random.split(key, 5)
        self.frame_shape = frame_shape
        self.num_freqs = num_freqs
        self.lam_dim = lam_dim
        self.split_forward = split_forward
        self.phase = phase
        H, W, C = frame_shape

        # Standard bottleneck size
        self.d_theta = 962 

        # The CNNDecoder lives inside `decoder`. 
        # This keeps the Phase 2 transplantation cells perfectly intact.
        self.decoder = CNNDecoder(in_dim=self.d_theta, out_channels=C, spatial_shape=(H, W), key=k_root, hidden_width=64, depth=4)

        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=64, depth=4)

        if CONFIG["pretrain_encoder"] and self.phase == 1:
            try:
                self.encoder, self.decoder = eqx.tree_deserialise_leaves("movingmnist_enc.eqx", (self.encoder, self.decoder))
            except:
                print("Warning: movingmnist_enc.eqx not found. Starting from scratch.")

        self.forward_dyn = ForwardDynamics(self.d_theta, lam_dim, split_forward, key=k_fwd)
        self.mem_dim = mem_dim

        num_actions = 4 if CONFIG["discrete_actions"] else None
        self.action_model = LAM(self.d_theta, lam_dim, mem_dim, max_len=20, num_heads=4, num_blocks=4, num_actions=num_actions, key=k_lam, phase=self.phase)

    def render_frame(self, z_t, coords_grid):
        # coords_grid is still accepted to prevent API breakage downstream, but we safely ignore it.
        H, W, C = self.frame_shape
        
        if not CONFIG["pretrain_encoder"]:
            decoder = self.decoder
        else:
            # Safely stop gradients across the entire PyTree
            decoder = jax.tree_util.tree_map(jax.lax.stop_gradient, self.decoder)

        out = decoder(z_t) # outputs (channels, H, W)
        out = jnp.transpose(out, (1, 2, 0)) # standardise back to (H, W, channels)

        # Handle NLL continuous distribution splitting if enabled
        if CONFIG.get("use_nll_loss", False):
            mean, std = out[..., :C], out[..., C:]
            std = jax.nn.softplus(std) + 1e-4
            return jnp.concatenate([mean, std], axis=-1)
            
        return out

    # -------------------------------------------------------------------------------------
    # PHASE 1 FORWARD: IDM Forcing (GCM is ignored/None)
    # -------------------------------------------------------------------------------------
    def phase1_forward(self, ref_video, coords_grid):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        if CONFIG["pretrain_encoder"]:
            z_init = jax.lax.stop_gradient(z_init)

        @eqx.filter_checkpoint
        def scan_step(z_t, scan_inputs):
            o_tp1, step_idx = scan_inputs

            time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            pred_out = self.render_frame(z_t, coords_grid_t)

            z_tp1_enc = self.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
            if CONFIG["pretrain_encoder"]:
                z_tp1_enc = jax.lax.stop_gradient(z_tp1_enc)

            a_t = self.action_model.inverse_dynamics(z_t, z_tp1_enc)
            z_tp1 = self.forward_dyn(z_t, a_t)

            return z_tp1, (a_t, z_t, pred_out)

        # scan_inputs = (ref_video[1:], jnp.arange(1, T))
        scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
        _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, z_init, scan_inputs)

        return actions, pred_latents, pred_video

    # -------------------------------------------------------------------------------------
    # PHASE 2 FORWARD: Action Matching (IDM/Base is frozen via stop_gradient)
    # -------------------------------------------------------------------------------------
    def phase2_forward(self, ref_video):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        # Explicitly freeze via stop_gradient
        z_init = jax.lax.stop_gradient(self.encoder(jnp.transpose(init_frame, (2, 0, 1))))
        m_init = self.action_model.reset_memory(T)

        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t = carry
            o_tp1, step_idx = scan_inputs

            z_tp1_enc = jax.lax.stop_gradient(self.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
            
            # Ground truth action from IDM (frozen)
            a_target = jax.lax.stop_gradient(self.action_model.inverse_dynamics(z_t, z_tp1_enc))

            # Predicted action from GCM
            a_pred = self.action_model.decode_memory(m_t, step_idx, jax.lax.stop_gradient(z_t))

            # Update memory buffer using target action (Teacher Forcing)
            m_tp1 = self.action_model.encode_memory(m_t, step_idx, jax.lax.stop_gradient(z_t), a_target)

            # Step dynamics (frozen)
            z_tp1 = jax.lax.stop_gradient(self.forward_dyn(z_t, a_target))

            return (z_tp1, m_tp1), (a_pred, a_target)

        scan_inputs = (ref_video[1:], jnp.arange(1, T))
        _, (a_preds, a_targets) = jax.lax.scan(scan_step, (z_init, m_init), scan_inputs)

        return a_preds, a_targets

    # -------------------------------------------------------------------------------------
    # INFERENCE ROLLOUT: Context-Conditioned Autoregressive Generation
    # -------------------------------------------------------------------------------------
    def inference_rollout(self, ref_video, coords_grid, context_ratio=0.0):
        T = ref_video.shape[0]
        init_frame = ref_video[0]
        
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        m_init = self.action_model.reset_memory(T)

        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t = carry
            o_tp1, step_idx = scan_inputs

            time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            pred_out = self.render_frame(z_t, coords_grid_t)

            # Determine if we are still in the context window
            is_context = (step_idx / T) < context_ratio

            # Conditionally choose action: IDM (Teacher Forcing) vs GCM (Autoregressive)
            a_t = jax.lax.cond(
                is_context,
                lambda: self.action_model.inverse_dynamics(
                    z_t, 
                    self.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
                ),
                lambda: self.action_model.decode_memory(m_t, step_idx, z_t)
            )

            m_tp1 = self.action_model.encode_memory(m_t, step_idx, z_t, a_t)
            z_tp1 = self.forward_dyn(z_t, a_t)

            return (z_tp1, m_tp1), (a_t, z_t, pred_out)

        # Pass the future ground truth frames into the scan so the IDM can use them
        scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
        _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init), scan_inputs)
        
        return actions, pred_latents, pred_video

#%% Cell 4: Phase 1 Training (Base Model & IDM)
if TRAIN_PHASE_1:
    print(f"\n🚀 [PHASE 1] Starting Base Training (IDM + FDM + maybe Enc) -> Saving to {run_path}")
    key, subkey = jax.random.split(key)

    model_p1 = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=1
    )
    
    print(f"Total Trainable Parameters in Phase 1 WARP: {count_trainable_params(model_p1)}")
    print(f"  - Number of paramters in dtheta (Root MLP): {model_p1.d_theta}", flush=True)
    print(f" - In the encoder: {count_trainable_params(model_p1.encoder)}")
    print(f" - In the decoder: {count_trainable_params(model_p1.decoder)}")
    print(f" - In the forward dynamics: {count_trainable_params(model_p1.forward_dyn)}")
    print(f" - In the IDM: {count_trainable_params(model_p1.action_model.idm)}")
    if model_p1.action_model.gcm is not None:
        print(f" - In the GCM: {count_trainable_params(model_p1.action_model.gcm)}")
    if model_p1.action_model.action_embedding is not None:
        print(f" - In the action embedding: {count_trainable_params(model_p1.action_model.action_embedding)}")

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

            # 1. Reverse Video Augmentation
            if CONFIG["reverse_video_aug"]:
                do_reverse = jax.random.bernoulli(k_aug, 0.5, shape=(ref_videos.shape[0],))
                ref_videos = jax.vmap(lambda rev, vid: jax.lax.cond(rev, 
                                                                       lambda v: jnp.flip(v, axis=0), 
                                                                       lambda v: v, 
                                                                       vid))(do_reverse, ref_videos)

            ## Repeat either the 0th or the mid frame at the end to ensure scan has T frames (since it looks at t+1)
            if CONFIG["static_video_aug"]:
                add_to_front = jax.random.bernoulli(k_init, 0.5, shape=(ref_videos.shape[0],))
                nb_frames = ref_videos.shape[1]
                repeat_frames = nb_frames // 4
                ref_videos = jax.vmap(lambda add_front, vid: jax.lax.cond(add_front, 
                                                                            # Add static frames at the front
                                                                            lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
                                                                            # Add static frames at the back
                                                                            lambda v_in: jnp.concatenate([v_in[:nb_frames-repeat_frames], jnp.repeat(v_in[nb_frames-repeat_frames:nb_frames-repeat_frames+1], repeats=repeat_frames, axis=0)], axis=0),
                                                                                vid))(add_to_front, ref_videos)

            actions, _, pred_videos = jax.vmap(m.phase1_forward, in_axes=(0, None))(ref_videos, coords_grid)

            # Note: pred_videos shape is [B, T-1, H, W, C], matching ref_videos[1:]
            # ref_videos = ref_videos[:, 1:]
            # ref_videos = ref_videos

            # # 2. SSIM + MSE Loss
            # def ssim(x, y, data_range=1.0):
            #     C1, C2 = (0.01 * data_range)**2, (0.03 * data_range)**2
            #     mu_x, mu_y = jnp.mean(x), jnp.mean(y)
            #     sigma_x, sigma_y = jnp.var(x), jnp.var(y)
            #     sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))
            #     return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
            # # ssim_loss = 1.0 - jnp.mean(jax.vmap(jax.vmap(ssim))(pred_videos, ref_videos))
            # ssim_loss = 1.0 - jnp.mean(jax.vmap(ssim)(pred_videos, ref_videos))

            mse_loss = jnp.mean((pred_videos - ref_videos)**2)

            # rec_loss = (1-CONFIG["mse_weight"]) * rec_loss_ssim + CONFIG["mse_weight"] * mse_loss
            rec_loss = mse_loss
            # rec_loss = ssim_loss

            # 3. L1 Continuous Action Regularisation
            action_l1_loss = 0.0
            if not CONFIG["discrete_actions"] and CONFIG["action_l1_reg"] > 0:
                action_l1_loss = CONFIG["action_l1_reg"] * jnp.mean(jnp.abs(actions))

            total_loss = rec_loss + action_l1_loss

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
            print(f"Phase 1 - Epoch {epoch+1}/{CONFIG['p1_nb_epochs']} - Avg Loss: {avg_loss:.6f}", flush=True)

        ## Save checkpoints and visualizations
        if (epoch+1) % (CONFIG["p1_nb_epochs"]//2) == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]:
            eqx.tree_serialise_leaves(artefacts_path / f"model_phase1_epoch{epoch+1}.eqx", model_p1)

        if (epoch+1) % (CONFIG["p1_nb_epochs"]//10) == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]:
            _, _, pred_videos = jax.vmap(model_p1.phase1_forward, in_axes=(0, None))(sample_videos_vis, coords_grid)
            for i in range(pred_videos.shape[0]):
                plot_pred_ref_videos_rollout(pred_videos[i], sample_videos_vis[i], epoch+1, plots_path / f"p1_vis_epoch{epoch+1}_sample{i}.png")

    print("\nPhase 1 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    # Save Phase 1 artifacts
    eqx.tree_serialise_leaves(artefacts_path / "model_phase1_final.eqx", model_p1)
    np.save(artefacts_path / "loss_history_p1.npy", np.array(all_losses_p1))
    np.save(artefacts_path / "lr_history_p1.npy", np.array(lr_scales_p1))

    # Phase 1 Dashboard
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(all_losses_p1, color='teal', alpha=0.8, label="Phase 1 Loss")
    ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color='teal')
    ax2 = ax1.twinx()  
    ax2.plot(lr_scales_p1, color='crimson', linewidth=2, label="LR Scale")
    plt.title("Phase 1: Base Model Training Loss")
    fig.tight_layout()
    plt.savefig(plots_path / "p1_loss_history.png")
    plt.show()

#%% Cell 5: Phase 2 Training (GCM Matching)
if TRAIN_PHASE_2:
    print(f"\n🚀 [PHASE 2] Starting GCM Training (Action Matching) -> Saving to {run_path}")
    key, subkey = jax.random.split(key)

    # 1. Initialize fresh Phase 2 model (with GCM enabled)
    model_p2 = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=2
    )

    # 2. Transplant weights from Phase 1
    print("📥 Loading Base weights from Phase 1...")
    dummy_p1 = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=1
    )
    # Load from the fresh save or the explicit P1 path
    load_path = artefacts_path / "model_phase1_final.eqx" if TRAIN_PHASE_1 else P1_LOAD_PATH
    dummy_p1 = eqx.tree_deserialise_leaves(load_path, dummy_p1)
    
    model_p2 = eqx.tree_at(lambda m: m.encoder, model_p2, dummy_p1.encoder)
    model_p2 = eqx.tree_at(lambda m: m.forward_dyn, model_p2, dummy_p1.forward_dyn)
    model_p2 = eqx.tree_at(lambda m: m.decoder, model_p2, dummy_p1.decoder)
    model_p2 = eqx.tree_at(lambda m: m.action_model.idm, model_p2, dummy_p1.action_model.idm)

    # 3. Partition parameters: Freeze everything except GCM
    # First, create a mask where absolutely everything is False
    filter_spec = jax.tree_util.tree_map(lambda _: False, model_p2)
    
    # Next, compute the proper gradient mask (True for float arrays) using the ACTUAL model
    gcm_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model_p2.action_model.gcm)
    
    # Graft the active GCM mask into our all-False filter_spec
    filter_spec = eqx.tree_at(lambda m: m.action_model.gcm, filter_spec, gcm_mask)
    
    # Partition the model using the corrected spec
    diff_model_p2, static_model_p2 = eqx.partition(model_p2, filter_spec)

    print(f"Trainable Parameters in Phase 2 (GCM only): {count_trainable_params(diff_model_p2)}")

    optimizer_p2 = optax.chain(
        optax.adam(CONFIG["p2_learning_rate"]),
        optax.contrib.reduce_on_plateau(
            patience=CONFIG["lr_patience"], cooldown=CONFIG["lr_cooldown"],
            factor=CONFIG["lr_factor"], rtol=CONFIG["lr_rtol"],
            accumulation_size=CONFIG["lr_accum_size"], min_scale=CONFIG["lr_min_scale"]
        )
    )
    opt_state_p2 = optimizer_p2.init(diff_model_p2)

    @eqx.filter_jit
    def train_step_p2(diff_m, static_m, opt_state, ref_videos):
        def loss_fn(d_model):
            # Recombine model for forward pass
            m = eqx.combine(d_model, static_m)
            
            # Action matching phase forward
            batched_fn = jax.vmap(m.phase2_forward, in_axes=(0,))
            a_preds, a_targets = batched_fn(ref_videos)
            
            # L1 Match Loss (GCM matching IDM)
            total_loss = jnp.mean(jnp.abs(a_preds - a_targets))
            return total_loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(diff_m)
        updates, opt_state = optimizer_p2.update(grads, opt_state, diff_m, value=loss_val)
        diff_m = eqx.apply_updates(diff_m, updates)
        return diff_m, opt_state, loss_val

    all_losses_p2 = []
    lr_scales_p2 = []
    start_time = time.time()

    for epoch in range(CONFIG["p2_nb_epochs"]):
        epoch_losses = []
        for batch_idx, batch_videos in enumerate(train_loader):
            diff_model_p2, opt_state_p2, loss = train_step_p2(diff_model_p2, static_model_p2, opt_state_p2, batch_videos)
            epoch_losses.append(loss)
            lr_scales_p2.append(optax.tree_utils.tree_get(opt_state_p2, "scale"))

        all_losses_p2.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["p2_nb_epochs"]):
            avg_loss = np.mean(epoch_losses)
            print(f"Phase 2 - Epoch {epoch+1}/{CONFIG['p2_nb_epochs']} - Avg Loss: {avg_loss:.6f}", flush=True)

        ## Visualize the the same phase 1 predictions
        if (epoch+1) % (CONFIG["p2_nb_epochs"]//10) == 0 or (epoch+1) == CONFIG["p2_nb_epochs"]:
            model_vis = eqx.combine(diff_model_p2, static_model_p2)
            _, _, pred_videos = jax.vmap(model_vis.phase1_forward, in_axes=(0, None))(sample_videos_vis, coords_grid)
            for i in range(pred_videos.shape[0]):
                plot_pred_ref_videos_rollout(pred_videos[i], sample_videos_vis[i], epoch+1, plots_path / f"p2_vis_epoch{epoch+1}_sample{i}.png")

    print("\nPhase 2 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    # Save Phase 2 artifacts
    model_final = eqx.combine(diff_model_p2, static_model_p2)
    eqx.tree_serialise_leaves(artefacts_path / "model_phase2_final.eqx", model_final)
    np.save(artefacts_path / "loss_history_p2.npy", np.array(all_losses_p2))
    np.save(artefacts_path / "lr_history_p2.npy", np.array(lr_scales_p2))

    # Phase 2 Dashboard
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(all_losses_p2, color='royalblue', alpha=0.8, label="Phase 2 Loss")
    ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color='royalblue')
    ax2 = ax1.twinx()  
    ax2.plot(lr_scales_p2, color='darkorange', linewidth=2, label="LR Scale")
    plt.title("Phase 2: GCM Action Matching Loss")
    fig.tight_layout()
    plt.savefig(plots_path / "p2_loss_history.png")
    plt.show()

#%% Cell 6: Evaluation & Plotting
print("\n=== Evaluating Phase 2 Model ===")

# If we skipped training, load the Phase 2 model
if not TRAIN_PHASE_2:
    print(f"📥 Loading completed Phase 2 WARP model from {artefacts_path}")
    key, subkey = jax.random.split(key)
    model_final = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=2
    )
    model_final = eqx.tree_deserialise_leaves(artefacts_path / "model_phase2_final.eqx", model_final)

    print(f"  - Number of paramters in root (dtheta): {model_final.d_theta}", flush=True)

@eqx.filter_jit
def evaluate(m, batch, coords, context_ratio):
    batched_fn = jax.vmap(m.inference_rollout, in_axes=(0, None, None))
    return batched_fn(batch, coords, context_ratio)

testing_subset = MovingMNISTDataset(test_arrays)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"]*1, shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

#%%
# Evaluation using context-maybe
# final_actions, _, final_videos = evaluate(model_final, sample_batch, coords_grid, CONFIG["inf_context_ratio"])
final_actions, _, final_videos = evaluate(model_final, sample_batch, coords_grid, 0.5)

#%%
test_seq_id = np.random.randint(0, sample_batch.shape[0])
print(f"\nGenerating final forecast rollout visualization for test sequence ID: {test_seq_id}")

# plot_pred_ref_videos_rollout(
#     final_videos[test_seq_id], 
#     sample_batch[test_seq_id, 1:], # Ground truth targets (shifted by 1)
#     title=f"Pred", 
#     save_name=plots_path / f"inference_forecast_rollout_seq{test_seq_id}.png"
# )

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
    save_name=plots_path / f"inference_fs11_seq{test_seq_id}.pdf",
    save_video=False,
)

#%% Gradually decreasing the context ratio
real_id = 1
final_actions, _, final_videos = evaluate(model_final, sample_batch[real_id:real_id+1], coords_grid, 21/20)
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
    # vmin=0,
    # vmax=1,
    save_name=plots_path / f"inference_gradual_fs21_seq{real_id}.pdf",
    save_video=True,
)

#%% Morphing digits from 8 to 9

## We need to rewrite the inference_rollout
@eqx.filter_jit
def inference_rollout_morph(model, ref_video, coords_grid, context_ratio=0.0):
    T = ref_video.shape[0]
    init_frame = ref_video[0]
    
    z_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    m_init = model.action_model.reset_memory(T)
    
    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t, a_tm1 = carry
        o_tp1, step_idx = scan_inputs

        time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
        coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
        pred_out = model.render_frame(z_t, coords_grid_t)

        # Determine if we are still in the context window
        is_context = (step_idx / T) < context_ratio

        # Conditionally choose action: IDM (Teacher Forcing) vs GCM (Autoregressive)
        a_t = jax.lax.cond(
            is_context,
            lambda: model.action_model.inverse_dynamics(
                z_t, 
                model.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
            ),
            lambda: model.action_model.decode_memory(m_t, step_idx, z_t)
        )

        z_89 = jnp.zeros_like(z_t) # 8 and 9
        # ## Interpolate between the original latent and the zero 89 latent based on the step index (morphing effect)
        # morph_ratio = jnp.clip((step_idx+1 - T//2) / (T//2), 0, 1) # Starts morphing at the halfway point
        # ## Start morphintg immediately but more gradually
        # # morph_ratio = jnp.clip((step_idx+1) / T, 0, 1) # Linear morphing from the start to the end
        # z_t = (1 - morph_ratio) * z_t + morph_ratio * z_89

        z_t = jnp.where(step_idx >= T//2, z_89, z_t) # Abrupt morphing at the halfway point

        ## The 4D action encodes the next location on the canvas, 2 for the first digit, 2 for the second.
        ## For the first 10 steps, we want the first to move, and the second to stay put. For the next 10, vice versa. Staying put means reusing the location from the previous step (a_tm1).
    
        a_t1 = jax.lax.cond(
            step_idx == 1, # At the first step, we have no previous action, so we just use the current IDM action for both digits
            lambda: a_t,
            lambda: jnp.concatenate([a_t[:2], a_tm1[2:]]), # Move the first digit (first 2 action values) and keep the second digit's location the same (last 2 action values)
        )
        a_t2 = jax.lax.cond(
            step_idx == T//2, # At the morphing point, we switch to moving the second digit, so we use the current IDM action for both digits
            lambda: a_tm1,
            lambda: jnp.concatenate([a_tm1[:2], a_t[2:]]), # Move the second digit (last 2 action values) and keep the first digit's location the same (first 2 action values)
        )

        # a_t1 = jnp.concatenate([a_t[:2], a_tm1[2:]])
        # a_t2 = jnp.concatenate([a_tm1[:2], a_t[2:]])
        # a_t = jnp.where(step_idx <= T//2, a_t1, a_t2)

        # a_t = -2*jnp.ones((model.lam_dim,), dtype=z_init.dtype) # Initialize previous action as zeros
        # a_rd = jax.random.uniform(jax.random.PRNGKey(step_idx), shape=a_t.shape, minval=-2, maxval=2) # Random actions for morphing
        # a_t = jnp.concatenate([a_rd[:2], a_t[2:]])

        a_zeros = jnp.zeros((model.lam_dim,), dtype=z_init.dtype)
        a_ones = 1*jnp.ones((model.lam_dim,), dtype=z_init.dtype)
        # a_t = jnp.concatenate([a_zeros[:2], a_ones[2:]])
        # a_t = a_tm1 + 0.01*(a_t - a_tm1) # Smoothly interpolate from the previous action to the current action to create a smoother morphing effect
        # alpha = 0.9985
        # a_t = alpha*a_tm1 + (1-alpha)*a_t # Exponential moving average to smooth the actions over time, creating a more gradual morphing effect

        # ratio = jnp.clip((step_idx+1) / T, 0, 1)
        # a_t = ratio * a_zeros + (1-ratio) * a_t

        m_tp1 = model.action_model.encode_memory(m_t, step_idx, z_t, a_t)
        z_tp1 = model.forward_dyn(z_t, a_t)

        return (z_tp1, m_tp1, a_t), (a_t, z_t, pred_out)

    # Pass the future ground truth frames into the scan so the IDM can use them
    scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
    a_init = jnp.zeros((model.lam_dim,), dtype=z_init.dtype)
    _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, a_init), scan_inputs) # Note: pred_video shape is [T, H, W, C]
    return pred_video

# Select one sequence of 8 morphing into 9
test_seq_id = np.random.randint(0, sample_batch.shape[0])
# test_seq_id = 233
print(f"\nGenerating morphing visualization for test sequence ID: {test_seq_id} (Digit 8 morphing into 9)")
final_videos = inference_rollout_morph(model_final, sample_batch[test_seq_id], coords_grid, 0/20)

# plot_videos(
#     final_videos, 
#     sample_batch[test_seq_id], 
#     show_titles=True,
#     show_labels=False,
#     plot_ref=False,
#     forecast_start=11,
#     forecast_gap=0.2,
#     hspace=0.02,
#     wspace=0.02,
#     # vmin=0,
#     # vmax=1,
#     save_name=plots_path / f"inference_control_seq{test_seq_id}.pdf",
#     save_video=True,
# )

plot_videos(
    final_videos, 
    sample_batch[test_seq_id], 
    show_titles=False,
    show_labels=False,
    plot_ref=False,
    forecast_start=1,
    forecast_gap=0.2,
    hspace=0.02,
    wspace=0.02,
    # vmin=0,
    # vmax=1,
    save_name=plots_path / f"inference_control_seq{test_seq_id}.pdf",
    save_video=True,
)




#%% Morphing digits from 8 to 9

## We need to rewrite the inference_rollout
@eqx.filter_jit
def inference_rollout_morph(model, ref_video, coords_grid, context_ratio=0.0):
    T = ref_video.shape[0]
    init_frame = ref_video[0]
    
    z_init = model.encoder(jnp.transpose(init_frame, (2, 0, 1)))
    m_init = model.action_model.reset_memory(T)
    
    @eqx.filter_checkpoint
    def scan_step(carry, scan_inputs):
        z_t, m_t, a_tm1 = carry
        o_tp1, step_idx = scan_inputs

        time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
        coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
        pred_out = model.render_frame(z_t, coords_grid_t)

        # Determine if we are still in the context window
        is_context = (step_idx / T) < context_ratio

        # Conditionally choose action: IDM (Teacher Forcing) vs GCM (Autoregressive)
        a_t = jax.lax.cond(
            is_context,
            lambda: model.action_model.inverse_dynamics(
                z_t, 
                model.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
            ),
            lambda: model.action_model.decode_memory(m_t, step_idx, z_t)
        )

        # z_t = jnp.zeros_like(z_t)
        # z_t = z_init

        a_zeros = jnp.zeros((model.lam_dim,), dtype=z_init.dtype)
        a_ones = 1*jnp.ones((model.lam_dim,), dtype=z_init.dtype)

        # ratio = jnp.clip((step_idx+1) / T, 0, 1)
        # a_t = ratio * a_zeros + (1-ratio) * a_t
        # a_t = a_ones

        a_zeroone = jnp.concatenate([a_zeros[:2], a_ones[2:]])

        ratio = jnp.clip((step_idx+1) / T, 0, 1)
        # a_t = ratio * a_zeros + (1-ratio) * a_t
        # a_t = ratio * a_zeroone + (1-ratio) * a_t

        m_tp1 = model.action_model.encode_memory(m_t, step_idx, z_t, a_t)
        z_tp1 = model.forward_dyn(z_t, a_t)

        return (z_tp1, m_tp1, a_t), (a_t, z_t, pred_out)

    # Pass the future ground truth frames into the scan so the IDM can use them
    scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
    a_init = jnp.zeros((model.lam_dim,), dtype=z_init.dtype)
    _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, a_init), scan_inputs) # Note: pred_video shape is [T, H, W, C]
    return pred_video

# Select one sequence of 8 morphing into 9
test_seq_id = np.random.randint(0, sample_batch.shape[0])
# test_seq_id = 199
print(f"\nGenerating morphing visualization for test sequence ID: {test_seq_id} (Digit 8 morphing into 9)")
input_video = sample_batch[test_seq_id]

# ## Darken the left side og the first frame.
# firt_frame = jnp.concatenate([input_video[0][:, :3*W//4] * 0.0, input_video[0][:, 3* W//4:]], axis=1)
# input_video = jnp.repeat(firt_frame[None], repeats=20, axis=0)

## Increase the lenght of th video to L = 100. Fill with zeros.
add_last_frames = jnp.repeat(input_video[-1:], repeats=150 - input_video.shape[0], axis=0)
# add_zeros_frames = jnp.zeros((350 - input_video.shape[0], H, W, C))
input_video = jnp.concatenate([input_video, add_last_frames], axis=0)


output_video = inference_rollout_morph(model_final, input_video, coords_grid, 0/20)

plot_videos(
    output_video, 
    input_video, 
    show_titles=False,
    show_labels=False,
    plot_ref=True,
    forecast_start=1,
    forecast_gap=0.2,
    hspace=0.02,
    wspace=0.02,
    # vmin=0,
    # vmax=1,
    save_name=plots_path / f"inference_long_seq{test_seq_id}.pdf",
    save_video=True,
)


#%% 1. Action Variance (Finding the Joystick Dimensions)
all_actions_flat = final_actions.reshape(-1, model_final.lam_dim)
action_variances = np.var(all_actions_flat, axis=0)

plt.figure(figsize=(10, 4))
plt.bar(range(model_final.lam_dim), action_variances, color='teal')
plt.xlabel("Latent Dimension")
plt.ylabel("Variance across all data")
plt.title("Latent Action Dimension Importance (Variance)")
plt.xticks(range(model_final.lam_dim))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "action_dimension_variance_mnist.png")
plt.show()

top_dims = np.argsort(action_variances)[-4:][::-1]
print(f"Top 4 most active latent dimensions: {top_dims}")

#%% 2. Continuous Action Trajectories over Time
seq_actions = final_actions[test_seq_id] # Shape: (T, lam_dim)
T_steps = seq_actions.shape[0]

plt.figure(figsize=(12, 6))
colors = ['crimson', 'dodgerblue', 'forestgreen', 'darkorange']

for i, dim in enumerate(top_dims):
    plt.plot(range(T_steps), seq_actions[:, dim], marker='o', linewidth=2, 
             color=colors[i], label=f"Dim {dim} (Rank {i+1})")

plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel("Time Step (t)")
plt.ylabel("Latent Action Value (Velocity Proxy)")
plt.title(f"Continuous Action Evolution for Sequence {test_seq_id}\n(Sudden changes indicate wall bounces)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / f"action_lines_seq{test_seq_id}.png")
plt.show()

# %% Save nohup
os.system(f"cp -r nohup.log {run_path}/nohup.log")


# %%
