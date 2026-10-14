import sys
sys.path.append("./PriSTI-TurboDTI")
from dataset_aqi36 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


with open("./PriSTI-TurboDTI/config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = 'AQI36'
config["seed"] = SEED


class AQIDataset(Dataset):
    def __init__(self, num_samples=329, device="cpu", data_type="DDPM", DDPM_ratio=0.5):
        self.num_samples = num_samples
        self.device = device
        
        # Set file paths based on data_type selection
        if data_type == "DDPM":
            self.paths = [
                f"./PriSTI-TurboDTI/sample_AQI_DDPM/sample_{i}.pt"
                #f"./PriSTI-TurboDTI/sample_AQI_CM/sample_{i}.pt"
                for i in range(num_samples)
            ]
        elif data_type == "CM":
            self.paths = [
                f"./PriSTI-TurboDTI/sample_AQI_CM/sample_{i}.pt"
                for i in range(num_samples)
            ]
        elif data_type == "hybrid":
            # Mix DDPM and CM samples based on DDPM_ratio
            DDPM_num_samples = int(num_samples * DDPM_ratio)
            DDPM_paths = [
                f"./PriSTI-TurboDTI/sample_AQI_DDPM/sample_{i}.pt"
                for i in range(DDPM_num_samples)
            ]
            CM_paths = [
                f"./PriSTI-TurboDTI/sample_AQI_CM/sample_{i}.pt"
                for i in range(DDPM_num_samples, num_samples)
            ]
            self.paths = DDPM_paths + CM_paths
        else:
            print("error data_type")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load and return sample data from the pre-saved file
        path = self.paths[idx]
        data = torch.load(path, map_location="cpu", weights_only=True)
        return {
            'observed_data': data['observed_data'],
            'observed_mask': data['observed_mask'],
            'observed_tp': data['observed_tp'],
            'cond_mask': data['cond_mask'],
            'coeffs': data['coeffs'],
            'teacher_sample': data['teacher_sample']
        }




class AQIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, device="cuda:0", num_workers=16, data_type="DDPM", DDPM_ratio=1.0, num_samples=329):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.num_samples = num_samples
        
        # Initialize the custom dataset for training
        self.train_dataset = AQIDataset(num_samples=num_samples, device="cpu", data_type=data_type, DDPM_ratio=DDPM_ratio)
        
        # Get DataLoader for training, validation, and testing, along with scalers
        self.train_loader_orig, self.valid_loader, self.test_loader, self.scaler, self.mean_scaler = get_dataloader(
            batch_size=batch_size,
            device=device,
            val_len=0.1,
            is_interpolate=True,
            num_workers=num_workers,
            target_strategy="hybrid",
            mask_sensor=[]
        )

    def train_dataloader(self):
        # Return DataLoader for training with batch_size=1 and shuffle enabled
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        # Return validation DataLoader
        return self.valid_loader

    def test_dataloader(self):
        # Return test DataLoader
        return self.test_loader


if __name__ == "__main__":
    pass