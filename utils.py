import os
import shutil
import datetime
import yaml
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw

sns.set_theme(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

def count_trainable_params(model):
    """Counts the number of trainable float parameters in an Equinox module."""
    def count_params(x):
        if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float32, jnp.float64]:
            return x.size
        return 0
    param_counts = jax.tree_util.tree_map(count_params, model)
    return sum(jax.tree_util.tree_leaves(param_counts))

def setup_run_dir(phase_name, config, train=True, base_dir="runs"):
    """
    Sets up the directory for the current phase.
    If train=True, creates a timestamped folder, copies the calling script,
    dumps the config to yaml, and returns the path.
    """

    ## If phase 2 or 3, do nothing, return ./
    if not train or phase_name in ["phase_2", "phase_3"]:
        data_dir = Path(config["data_path"])
        # if data_dir.parent.parent.exists():
        #     config["data_path"] = str(data_dir.parent.parent)
        # else:
        #     print(f"Warning: Could not find parent directory for {data_dir}. Keeping original path.")

        # data_dir = "../../data"
        config["data_path"] = "../../" + str(data_dir.name)

        run_path = Path("./")

    # if train:
    if train and phase_name not in ["phase_2", "phase_3"]:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)

        (run_path / "artefacts").mkdir(exist_ok=True)
        (run_path / "plots").mkdir(exist_ok=True)
        
        with open(run_path / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # 1. Handle current_script but ignore ipykernel_launcher
        current_script = Path(sys.argv[0])
        files_to_copy = [
            "utils.py", "loaders.py", "models.py", "phase1.py",
            "phase2.py", "phase3.py"
        ]

        # if current_script.exists() and current_script.is_file() and "ipykernel_launcher" not in current_script.name:
        #     files_to_copy.append(current_script.name)

        # 2. Copy the files and use a set() to avoid trying to copy the same file twice
        for fname in set(files_to_copy):
            src_file = Path(fname)
            if src_file.exists() and src_file.is_file():
                shutil.copy(src_file, run_path / src_file.name)
            elif fname in ["phase_1.py", "phase1.py"]:
                # Optional: Print a warning so you know exactly why it's failing if it still doesn't copy
                pass 

    return run_path

def get_coords_grid(H, W):
    """Generates a normalised coordinate grid for the INR."""
    y_coords = jnp.linspace(-1, 1, H)
    x_coords = jnp.linspace(-1, 1, W)
    X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
    return jnp.stack([X_grid, Y_grid], axis=-1)

def ssim(x, y, data_range=1.0):
    """Computes Structural Similarity Index Measure between two batched images."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = jnp.mean(x)
    mu_y = jnp.mean(y)
    sigma_x = jnp.var(x)
    sigma_y = jnp.var(y)
    sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))

    ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return ssim_numerator / ssim_denominator


def plot_videos(video, ref_video=None, plot_ref=True, show_titles=True, show_labels=True, forecast_start=None, 
                vmin=None, vmax=None, save_name=None, 
                wspace=0.05, hspace=0.05, forecast_gap=0.2, 
                save_video=False, video_gap=5, show_borders=False, corner_radius=5):
    """
    Plots a camera-ready rollout of ground truth and predicted video frames.
    
    Args:
        show_borders (bool): If True, applies rounded borders to the frames.
        corner_radius (int): The radius of the rounded corners (if show_borders=True).
    """
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 18,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }):
        
        nb_frames = video.shape[0]
        C = video.shape[-1]
        
        if plot_ref and ref_video is None:
            raise ValueError("ref_video must be provided if plot_ref is True.")

        rescale = False
        if plot_ref and ref_video[..., :C].min() < -0.5:
            rescale = True
            ref_video = (ref_video + 1.0) / 2.0
        elif not plot_ref and video.min() < -0.5:
            rescale = True

        nrows = 2 if plot_ref else 1
        has_gap = forecast_start is not None and 1 < forecast_start <= nb_frames
        ncols = nb_frames + 1 if has_gap else nb_frames
        
        width_ratios = [1.0] * ncols
        spacer_col = -1
        if has_gap:
            spacer_col = forecast_start - 1
            width_ratios[spacer_col] = forecast_gap

        fig_width = (nb_frames + (forecast_gap if has_gap else 0.0)) * 1.5
        fig = plt.figure(figsize=(fig_width, nrows * 1.55))
        
        gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios)
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, c])

        imshow_kwargs = {}
        if not plot_ref:
            if vmin is not None: imshow_kwargs['vmin'] = vmin
            if vmax is not None: imshow_kwargs['vmax'] = vmax

        frame_idx = 0
        for c in range(ncols):
            if c == spacer_col:
                for r in range(nrows):
                    axes[r, c].axis('off')
                continue
                
            pred_frame = video[frame_idx]
            if rescale: pred_frame = (pred_frame + 1.0) / 2.0
            
            if plot_ref or (vmin is None and vmax is None):
                pred_frame = np.clip(pred_frame, 0.0, 1.0)
            
            if pred_frame.shape[-1] == 1: pred_frame = np.repeat(pred_frame, 3, axis=-1)

            if plot_ref:
                ref_idx = min(frame_idx, ref_video.shape[0] - 1)
                ref_frame = ref_video[ref_idx]
                if rescale: ref_frame = (ref_frame + 1.0) / 2.0
                ref_frame = np.clip(ref_frame, 0.0, 1.0)
                if ref_frame.shape[-1] == 1: ref_frame = np.repeat(ref_frame, 3, axis=-1)

            # Capture the imshow objects so we can apply clip_path
            if plot_ref:
                im_ref = axes[0, c].imshow(ref_frame)
                im_pred = axes[1, c].imshow(pred_frame)
                target_axes = [(axes[0, c], im_ref, ref_frame), (axes[1, c], im_pred, pred_frame)]
            else:
                im_pred = axes[0, c].imshow(pred_frame, **imshow_kwargs)
                target_axes = [(axes[0, c], im_pred, pred_frame)]

            for ax, im_obj, frame_data in target_axes:
                ax.set_xticks([])
                ax.set_yticks([])
                
                h, w = frame_data.shape[:2]
                
                # Turn off default square spines
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
                # Conditionally apply the rounded border and clipping
                if show_borders:
                    rect = patches.FancyBboxPatch(
                        (-0.5, -0.5), w, h,
                        boxstyle=f"round,pad=0,rounding_size={corner_radius}", 
                        linewidth=1.2, edgecolor='black', facecolor='none',
                        transform=ax.transData
                    )
                    ax.add_patch(rect)
                    im_obj.set_clip_path(rect)

            if show_titles:
                top_ax = axes[0, c]
                if frame_idx == 0 or (frame_idx + 1 == forecast_start):
                    title_str = f"$t={frame_idx + 1}$"
                else:
                    title_str = str(frame_idx + 1)
                
                font_weight = 'bold' if (has_gap and frame_idx + 1 == forecast_start) else 'normal'
                top_ax.set_title(title_str, pad=8, fontsize=18, fontweight=font_weight)

            frame_idx += 1

        if show_labels:
            if plot_ref:
                axes[0, 0].set_ylabel("GT", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
                axes[1, 0].set_ylabel("Pred", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')
            else:
                axes[0, 0].set_ylabel("Pred", rotation=0, labelpad=25, ha='right', va='center', fontsize=28, fontweight='bold')

        if save_name:
            plt.savefig(save_name, dpi=100, bbox_inches='tight', facecolor='white', transparent=False)
        else:
            plt.draw()

        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
            
        plt.close(fig)

        # ---------------------------------------------------------
        # GIF / Video Generation
        # ---------------------------------------------------------
        if save_video and save_name is not None:
            try:
                font = ImageFont.truetype("Helvetica.ttc", 14)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()

            def process_pil_image(img_array, radius=corner_radius, apply_frame=show_borders):
                """Helper function to conditionally apply rounded corners and a border."""
                h, w = img_array.shape[:2]
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                
                if not apply_frame:
                    return img
                
                # Create a rounded mask
                mask = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
                
                # Paste the frame using the mask onto a white background
                rounded_img = Image.new("RGB", (w, h), "white")
                rounded_img.paste(img, (0, 0), mask=mask)
                
                # Draw the bounding frame border
                draw_border = ImageDraw.Draw(rounded_img)
                draw_border.rounded_rectangle((0, 0, w-1, h-1), radius=radius, outline="black", width=1)
                
                return rounded_img

            gif_frames = []
            for t in range(nb_frames):
                p_f = video[t]
                if rescale: p_f = (p_f + 1.0) / 2.0
                
                if not plot_ref and (vmin is not None or vmax is not None):
                    v_min = vmin if vmin is not None else p_f.min()
                    v_max = vmax if vmax is not None else p_f.max()
                    p_f = (p_f - v_min) / (v_max - v_min + 1e-8)
                
                p_f = np.clip(p_f, 0.0, 1.0)
                if p_f.shape[-1] == 1: p_f = np.repeat(p_f, 3, axis=-1)

                if plot_ref:
                    r_idx = min(t, ref_video.shape[0] - 1)
                    r_f = ref_video[r_idx]
                    if rescale: r_f = (r_f + 1.0) / 2.0
                    r_f = np.clip(r_f, 0.0, 1.0)
                    if r_f.shape[-1] == 1: r_f = np.repeat(r_f, 3, axis=-1)
                    
                    # Apply styling to individual frames conditionally
                    img_ref = process_pil_image(r_f)
                    img_pred = process_pil_image(p_f)
                    
                    # Create canvas for both
                    combined_w = img_ref.width + video_gap + img_pred.width
                    combined_h = max(img_ref.height, img_pred.height)
                    combined_frame = Image.new('RGB', (combined_w, combined_h), 'white')
                    
                    # Paste them side-by-side
                    combined_frame.paste(img_ref, (0, 0))
                    combined_frame.paste(img_pred, (img_ref.width + video_gap, 0))
                else:
                    combined_frame = process_pil_image(p_f)

                header_height = 15
                final_img = Image.new('RGB', (combined_frame.width, combined_frame.height + header_height), color='white')
                final_img.paste(combined_frame, (0, header_height))
                
                draw = ImageDraw.Draw(final_img)
                if plot_ref:
                    gt_w = r_f.shape[1]
                    pred_w = p_f.shape[1]
                    draw.text((gt_w // 2 - 10, 2), "GT", font=font, fill="black")
                    draw.text((gt_w + video_gap + pred_w // 2 - 17, 2), "Pred", font=font, fill="black")
                else:
                    draw.text((p_f.shape[1] // 2 - 17, 2), "Pred", font=font, fill="black")
                
                gif_frames.append(final_img)
            
            gif_path = Path(save_name).with_suffix('.gif')
            gif_frames[0].save(
                gif_path, save_all=True, append_images=gif_frames[1:], duration=150, loop=0
            )
            print(f"Saved rollout animation to {gif_path}")

            try:
                from IPython.display import Image as IPyImage, display
                display(IPyImage(filename=str(gif_path)))
            except ImportError:
                pass