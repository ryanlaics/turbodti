import sys
sys.path.append("./PriSTI-TurboDTI")
from dataset_pems04 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


with open("./PriSTI-TurboDTI/config/traffic.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = 'pems-04'
config["seed"] = SEED

class Pems04Dataset(Dataset):
    def __init__(self, num_samples=742, device="cpu", data_type="DDPM", DDPM_ratio=0.5):
        self.num_samples = num_samples
        self.device = device
        DDPM_base = "./PriSTI-TurboDTI/sample_PeMS04_DDPM/sample_{}.pt"
        CM_base = "./PriSTI-TurboDTI/sample_PeMS04/CM/sample_{}.pt"

        # Initialize paths based on data_type and ratio
        if data_type == "DDPM":
            self.paths = [DDPM_base.format(i) for i in range(num_samples)]
        elif data_type == "CM":
            self.paths = [CM_base.format(i) for i in range(num_samples)]
        elif data_type == "hybrid":
            self.paths = []
            for i in range(num_samples):
                if random.random() < DDPM_ratio:
                    self.paths.append(DDPM_base.format(i))
                else:
                    self.paths.append(CM_base.format(i))
        else:
            print("error data_type")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load the data sample from the corresponding path
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

    



class PeMS04DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, device="cuda:0", num_workers=16, data_type="hybrid", DDPM_ratio=1.0, num_samples=742):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.num_samples = num_samples

        # Initialize the dataset with specified data_type and DDPM_ratio
        self.train_dataset = Pems04Dataset(num_samples=num_samples, device="cpu", data_type=data_type, DDPM_ratio=DDPM_ratio)

        # Get dataloaders for training, validation, and testing splits along with scalers
        self.train_loader_orig, self.valid_loader, self.test_loader, self.scaler, self.mean_scaler = get_dataloader(
            batch_size=batch_size,
            device=device,
            val_len=0.1,
            missing_pattern="point",
            is_interpolate=True,
            num_workers=num_workers,
            target_strategy="hybrid",
        )

        
        

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

if __name__ == "__main__":
    pass