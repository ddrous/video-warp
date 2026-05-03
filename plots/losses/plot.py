#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Apply requested settings ---
warnings.filterwarnings("ignore")
# sns.set(palette="muted", color_codes=True, font_scale=4.2)
sns.set_theme(style="white", context="talk", font_scale=1.6)
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_losses(continuous_path, discrete_path):
    # 1. Load the data
    try:
        cont_data = np.load(continuous_path)
        disc_data = np.load(discrete_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Process Continuous Data (Shape: 601, 2)
    # We take the first 600 rows. Assuming col 0 is step/epoch and col 1 is the loss value.
    if cont_data.ndim > 1:
        loss_continuous = cont_data[:600, -1] 
    else:
        loss_continuous = cont_data[:600]

    # 3. Process Discrete Data (Shape: 2500000,)
    # 2,500,000 steps = 6000 epochs. Therefore, 600 epochs = 250,000 steps.
    target_epochs = 600
    total_epochs = 6000
    total_steps = len(disc_data)
    
    steps_for_target = int(total_steps * (target_epochs / total_epochs)) # 250,000
    disc_slice = disc_data[:steps_for_target]
    
    # Chunk the 250,000 steps into exactly 600 bins and take the mean of each chunk
    loss_discrete = np.array([np.mean(chunk) for chunk in np.array_split(disc_slice, target_epochs)])

    # 4. Create the beautiful plot
    epochs = np.arange(1, target_epochs + 1)
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines with a sleek linewidth
    sns.lineplot(x=epochs, y=loss_continuous, label="Continuous", linewidth=4.5, ax=ax)
    sns.lineplot(x=epochs, y=loss_discrete, label="Discrete", linewidth=4.5, alpha=0.9, ax=ax)

    # --- Set Log Scale ---
    ax.set_yscale("log")

    # Formatting labels and title
    # ax.set_title("Training Loss Comparison (First 600 Epochs)", pad=20, fontweight='bold')
    ax.set_xlabel("Epochs (strided)", labelpad=10, fontweight='medium')
    ax.set_ylabel("MSE", labelpad=10, fontweight='medium')

    # Clean up the axes by removing the top and right spines
    sns.despine()
    
    # Add subtle grid styling: both major and minor ticks for the Y log scale
    ax.grid(axis='y', which='both', linestyle='-', alpha=0.15)
    ax.grid(axis='x', linestyle='--', alpha=0.15)

    # Style the legend to be borderless and blend seamlessly into the white background
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.draw()

    ## Save the plot if needed
    plt.savefig("minigrid_loss_comparison.pdf", dpi=100, bbox_inches='tight')

# --- Execution ---
continuous_file = "continuous.npy"
discrete_file = "discrete.npy"
plot_losses(continuous_file, discrete_file)

