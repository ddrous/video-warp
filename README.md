# V-WARP (Video-Weight-space Adaptive Recurrent Prediction)

V-WARP unifies world modelling and video generation by actively modulating the weights of an Implicit Neural Representation (INR) to render future frames. Instead of operating purely in abstract latent spaces, V-WARP's physics-driven approach ensures spatial consistency via its Coordinate-Based rendering structure.

## Folder Structure

* `config.yaml`: Example base configurations for datasets, architectures, trainings, etc.
* `loaders.py`: Data loaders for MovingMNIST, MiniGrid, WeatherBench, and PhyWorld environments.
* `utils.py`: Utilities for metrics, advanced visualization plotting, and run directory management.
* `models.py`: Core components and the main `VWARP` class housing the vectorized inference rollout functions.
* `phase1.py`: **Encoder Pre-training.** Learns the base network and fits a CNN to individual frames using SSIM/MSE.
* `phase2.py`: **Dynamics Fitting.** Uses teacher forcing to train the Inverse Dynamics Model (IDM) and transition model (FDM: $A+B$).
* `phase3.py`: **Generative Control.** Freezes the Dynamics & Encoder to train the Generative Control Model (GCM). The GCM learns to mimic the action sequences deduced by the IDM.

## Setup & Data

Please ensure the data paths are correct before running (configured in `config.yaml`):

* **MiniGrid**: `./data/MiniGrid/minigrid.npy`
* **MovingMNIST**: `./data/MovingMNIST/mnist_test_seq.npy`
* **WeatherBench**: `./data/WeatherBench/*.nc`
* **PhyWorld**: `./data/PhyWorld/*.hdf5`

To switch between Continuous and Discrete setups, modify the `dataset` and `discrete_actions` flags in the respective configurations. 

**Debugging:** You can toggle `debug: True` in `config.yaml` to run a tiny 2-sample batch for rapid pipeline testing.

## Execution Order

The modules run sequentially:

1. **Pre-train Encoder** (from the root directory). To skip encoder training, set `pretrain_encoder: False` in the config, and toggle TRAIN=False in the phase1.py script.
   ```bash
   nohup python phase1.py cfgs/config.yaml > nohup.log
    ```
2. **Fit Dynamics**  (from the generated run directory, e.g., `runs/230101-123456/`)
   ```bash
   nohup python phase2.py > nohup.log
   ```
3. **Train Generative Control**  (from the same run directory)
   ```bash
   nohup python phase3.py > nohup.log
   ```

## Trained Models (TODO: Add links to trained models)
The following folders will be made available for each dataset, containing trained models, logs, and plots used in the paper:
* **MiniGrid**: `runs/260316-202801-PerfectMiniGrid*`
* **MovingMNIST**: `runs/260312-113749-PerfectMNIST*`
* **WeatherBench**: `runs/260321-013009-WeatherBench*`
* **PhyWorld**: `runs/260318-174150-PhyWorld*`
