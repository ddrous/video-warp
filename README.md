# V-WARP (Video-Weight-space Adaptive Recurrent Prediction)

V-WARP unifies world modelling and video generation by actively modulating the weights of an Implicit Neural Representation (INR) to render future frames. Instead of operating purely in abstract decoding spaces, V-WARP's physics-driven approach ensures spatial consistency via its Coordinate-Based rendering structure.

## Folder Structure

* `config.yaml`: Shared base configurations for datasets, architectures
* `loaders.py`: Data loaders for MovingMNIST, MiniGrid, WeatherBench, and PhyWorld environments.
* `utils.py`: Utilities for metrics (SSIM), advanced visualization plotting, and run directory management.
* `models.py`: Core components and the main `VWARP` class housing the vectorized inference rollout functions.
* `phase_1.py`: **Autoencoder Pre-training.** Learns the base network and fits the `WeightCNN` to individual frames using SSIM/MSE.
* `phase_2.py`: **Dynamics Fitting.** Uses teacher forcing to train the Inverse Dynamics Model (IDM) and transition model (FDM: $A+B$).
* `phase_3.py`: **Generative Control.** Freezes the Dynamics & Encoder to train the Generative Control Model (GCM). The GCM learns to mimic the action sequences deduced by the IDM.

## Setup & Data

Please ensure the data paths are correct before running (configured in `config.yaml`):

* **MiniGrid**: `./data/MiniGrid/minigrid.npy`
* **MovingMNIST**: `./data/MovingMNIST/mnist_test_seq.npy`
* **WeatherBench**: `./data/WeatherBench/*.nc`
* **PhyWorld**: `./data/PhyWorld/*.hdf5`

To switch between Continuous and Discrete setups, modify the `dataset` and `discrete_actions` flags in the respective configurations. 

**Debugging:** You can toggle `single_batch: True` in `base_config.yaml` to run a tiny 2-sample batch for rapid pipeline testing.

## Execution Order

The modules run sequentially:

1. **Pre-train Encoder** (from the root directory)
   ```bash
   python phase_1.py
    ```
2. **Fit Dynamics**  (from the generated run directory, e.g., `runs/230101-123456/`)
   ```bash
   python phase_2.py
   ```
3. **Train Generative Control**  (from the same run directory)
   ```bash
   python phase_3.py
   ```

## Metrics & Visualization
* **SSIM**: Evaluates the structural similarity between predicted and ground truth frames.

