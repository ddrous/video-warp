import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import glob
import os
import subprocess
import h5py
import tempfile
import cv2
import xarray as xr

def numpy_collate(batch):
    """Collates a list of numpy arrays into a batched numpy array for Jax."""
    if isinstance(batch[0], tuple):
        videos = torch.stack([b[0] for b in batch]).numpy()
    else:
        videos = torch.stack(batch).numpy()
    
    # FIX: Check if the 4D array is already RGB (MiniGrid frames) before expanding
    if videos.ndim == 4 and videos.shape[-1] not in [1, 3]: 
        # (B, T, H, W) -> (B, T, H, W, 1) for grayscale videos missing a channel
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 3: 
        # (B, H, W) -> (B, H, W, 1) for grayscale frames
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 5 and videos.shape[2] == 1:
        videos = np.transpose(videos, (0, 1, 3, 4, 2))

    videos = videos.astype(np.float32)
    # if videos.max() > 2.0:
    #     videos = videos / 255.0

    return videos

class MiniGridDataset(Dataset):
    def __init__(self, data_array):
        self.data_array = data_array
        if self.data_array.max() > 2.0:
            self.data_array = self.data_array / 255.0
        self.max_val = self.data_array.max()
    def __len__(self):
        return self.data_array.shape[0]
    def __getitem__(self, idx):
        video = self.data_array[idx]
        return torch.from_numpy(video.astype(np.float32))

class MovingMNISTDataset(Dataset):
    def __init__(self, data_array):
        self.data_array = data_array
        if self.data_array.max() > 2.0:
            self.data_array = self.data_array / 255.0
        self.max_val = self.data_array.max()
    def __len__(self):
        return self.data_array.shape[1]
    def __getitem__(self, idx):
        video = self.data_array[:, idx]
        video = np.expand_dims(video, axis=-1)
        return torch.from_numpy(video.astype(np.float32))

class FrameDataset(Dataset):
    def __init__(self, data_array):
        self.data_array = data_array
        if self.data_array.max() > 2.0:
            self.data_array = self.data_array / 255.0
        self.max_val = self.data_array.max()
    def __len__(self):
        return self.data_array.shape[0]
    def __getitem__(self, idx):
        frame = self.data_array[idx]
        return torch.from_numpy(frame.astype(np.float32))

class WeatherBenchTemperature(Dataset):
    def __init__(self, data_path="./data/WeatherBench", split="train", download=False, seq_len=24, mean=None, std=None):
        self.data_path = data_path
        self.split = split
        self.seq_len = seq_len
        
        if download:
            self._download_and_extract()
            
        if split == "train":
            years = [str(y) for y in range(1979, 2016)]
        elif split == "val":
            years = ['2016']
        elif split == "test":
            years = ['2017', '2018']
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
            
        file_patterns = [os.path.join(data_path, f"*{y}*.nc") for y in years]
        files_to_load = []
        for pat in file_patterns:
            files_to_load.extend(glob.glob(pat))
            
        if len(files_to_load) == 0:
            raise FileNotFoundError(f"No .nc files found. Try setting download=True.")
            
        dataset = xr.open_mfdataset(sorted(files_to_load), combine='by_coords')
        raw_data = dataset.get('t2m').values 
        
        if split == "train":
            self.mean = raw_data.mean() if mean is None else mean
            self.std = raw_data.std() if std is None else std
        else:
            self.mean = raw_data.mean() if mean is None else mean
            self.std = raw_data.std() if std is None else std
            
        norm_data = (raw_data - self.mean) / self.std
        self.data = np.expand_dims(norm_data, axis=1).astype(np.float32)

        self.max_val = self.data.max()

    def _download_and_extract(self):
        if len(glob.glob(os.path.join(self.data_path, "*.nc"))) > 0: return
        os.makedirs(self.data_path, exist_ok=True)
        zip_path = os.path.join(self.data_path, "2m_temperature.zip")
        url = "https://dataserv.ub.tum.de/public.php/dav/files/m1524895/5.625deg/2m_temperature/?accept=zip"
        subprocess.run(["wget", url, "-O", zip_path], check=True)
        subprocess.run(["unzip", "-q", zip_path, "-d", self.data_path], check=True)
        os.remove(zip_path)

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len]
        return torch.from_numpy(seq)

class PhyWorldDataset(Dataset):
    def __init__(self, data_path, seq_len=32):
        self.data_path = data_path
        self.seq_len = seq_len
        self.video_data = []
        
        with h5py.File(data_path, 'r') as f:
            video_group = f['video_streams']
            for key in video_group.keys():
                self.video_data.append(video_group[key][:])
        self.video_data = np.concatenate(self.video_data, axis=0)

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_bytes = self.video_data[idx]
        frames = []
        
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_vid:
            temp_vid.write(video_bytes)
            temp_vid.flush()
            cap = cv2.VideoCapture(temp_vid.name)
            
            while len(frames) < self.seq_len:
                ret, frame_bgr = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()
            
        frames = np.array(frames)
        if len(frames) < self.seq_len:
            pad = np.zeros((self.seq_len - len(frames), *frames.shape[1:]), dtype=frames.dtype)
            frames = np.concatenate([frames, pad], axis=0)
        else:
            frames = frames[:self.seq_len]

        ## Downsample the frames to 128 x 128 to reduce memory
        frames = frames[:, ::2, ::2, :]

        if frames.max() > 2.0:
            frames = frames / 255.0

        return torch.from_numpy(frames.astype(np.float32))

def get_dataloaders(config, phase="phase_1"):
    dataset_name = config["dataset"]
    data_path = config["data_path"]
    single_batch = config.get("debug", False)
    
    is_phase1 = (phase == "phase_1")
    batch_size = config[phase]["batch_size"]

    print(f"Initializing {dataset_name} DataLoaders...")
    
    if dataset_name.lower() == "minigrid":
        arrays = np.load(f"{data_path}/MiniGrid/minigrid.npy")
        arrays = arrays[:, :10]
        train_size = int(0.8 * arrays.shape[0])
        train_arrays = arrays[:train_size]
        test_arrays = arrays[train_size:]
        
        if is_phase1:
            if config.get("encoder_sees_test", False):
                train_data = np.concatenate([train_arrays, test_arrays], axis=0).reshape(-1, *train_arrays.shape[2:])
            else:
                train_data = train_arrays.reshape(-1, *train_arrays.shape[2:])
            test_data = test_arrays.reshape(-1, *test_arrays.shape[2:])
            dataset = FrameDataset(train_data)
            test_dataset = FrameDataset(test_data)
        else:
            dataset = MiniGridDataset(train_arrays)
            test_dataset = MiniGridDataset(test_arrays)

    elif dataset_name.lower() == "movingmnist":
        arrays = np.load(f"{data_path}/MovingMNIST/mnist_test_seq.npy")
        train_arrays = arrays[:, :8000]
        test_arrays = arrays[:, 8000:]
        
        if is_phase1:
            if config.get("encoder_sees_test", False):
                train_data = np.concatenate([train_arrays, test_arrays], axis=1)
                train_data = np.transpose(train_data, (1, 0, 2, 3)).reshape(-1, *train_arrays.shape[2:])
            else:
                train_data = np.transpose(train_arrays, (1, 0, 2, 3)).reshape(-1, *train_arrays.shape[2:])
            test_data = np.transpose(test_arrays, (1, 0, 2, 3)).reshape(-1, *test_arrays.shape[2:])
            dataset = FrameDataset(train_data)
            test_dataset = FrameDataset(test_data)
        else:
            dataset = MovingMNISTDataset(train_arrays)
            test_dataset = MovingMNISTDataset(test_arrays)
            
    elif dataset_name.lower() == "weatherbench":
        # print("final data folder is: ", f"{data_path}/WeatherBench/2m_temperature/")
        dataset = WeatherBenchTemperature(data_path=f"{data_path}/WeatherBench/2m_temperature/", split="train")
        test_dataset = WeatherBenchTemperature(data_path=f"{data_path}/WeatherBench/2m_temperature/", split="test", mean=dataset.mean, std=dataset.std)
        if is_phase1:
            # raise NotImplementedError("Phase 1 Frame Extraction for WeatherBench is custom.")
            print("⚠️ Phase 1 frame by frame extraction not implemented. Time dimension will be flattened during training. Could cause issues!")

    elif dataset_name.lower() == "phyworld":
        dataset = PhyWorldDataset(f"{data_path}/PhyWorld/collision_30K.hdf5", seq_len=32)
        test_dataset = PhyWorldDataset(f"{data_path}/PhyWorld/collision_eval.hdf5", seq_len=32)
        if is_phase1:
            # raise NotImplementedError("Phase 1 Frame Extraction for PhyWorld is custom.")
            print("⚠️ Phase 1 frame by frame extraction not implemented. Time dimension will be flattened during training. Could cause issues!")
            
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if single_batch:
        print("⚠️ Running in single_batch mode (Debug): Limiting to 2 sequences/frames.")
        dataset = Subset(dataset, range(min(2, len(dataset))))
        test_dataset = Subset(test_dataset, range(min(2, len(test_dataset))))
        batch_size = min(2, batch_size)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=False)
    
    return train_loader, test_loader

#%% Test to visualise some batches from the dataloader
if __name__ == "__main__":
    import yaml
    from utils import plot_videos

    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)

    dataset = CONFIG["dataset"]
    train_loader, test_loader = get_dataloaders(CONFIG, phase="phase_1")
    sample_batch = next(iter(train_loader))
    print(f"Sample batch shape from {dataset} train loader: {sample_batch.shape}")

    seq_idx = np.random.randint(sample_batch.shape[0])
    plot_videos(sample_batch[seq_idx],
                plot_ref=False,
                save_name=f"artefacts/test_vis.png",
                save_video=True,
    )
