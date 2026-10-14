import sys
sys.path.append("./CSDI-TurboDTI")
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



# Custom Dataset for PeMS04, supporting DDPM, CM, or hybrid data sources
class Pems04Dataset(Dataset):
    def __init__(self, num_samples=742, device="cpu", data_type="hybrid", DDPM_ratio=1.0):
        self.num_samples = num_samples
        self.device = device

        # Define file path patterns for DDPM and CM samples
        DDPM_base = "./CSDI-TurboDTI/sample_PeMS04_DDPM/sample_{}.pt"
        CM_base = "./CSDI-TurboDTI/sample_PeMS04_CM/sample_{}.pt"

        # Select sample paths based on the data_type
        if data_type == "DDPM":
            self.paths = [DDPM_base.format(i) for i in range(num_samples)]
        elif data_type == "CM":
            self.paths = [CM_base.format(i) for i in range(num_samples)]
        elif data_type == "hybrid":
            # Hybrid selection: each sample randomly chosen from DDPM or CM
            self.paths = []
            for i in range(num_samples):
                if random.random() < DDPM_ratio:
                    self.paths.append(DDPM_base.format(i))
                else:
                    self.paths.append(CM_base.format(i))
        else:
            print("error data_type")  # Invalid data_type input

    def __len__(self):
        # Return total number of samples
        return self.num_samples

    def __getitem__(self, idx):
        # Load sample from the selected path
        path = self.paths[idx]
        data = torch.load(path, map_location="cpu", weights_only=True)  # Load .pt file to CPU

        # Return a dictionary containing all required fields
        return {
            'observed_data': data['observed_data'],       # Original data
            'observed_mask': data['observed_mask'],       # Mask for observed data
            'observed_tp': data['observed_tp'],           # Time points
            'cond_mask': data['cond_mask'],               # Conditioning mask
            'coeffs': data['coeffs'],                     # Additional side information
            'teacher_sample': data['teacher_sample']      # Reference sample from teacher model
        }

    



# PyTorch Lightning DataModule for the PeMS04 dataset
class PeMS04DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, device="cuda:0", num_workers=16, data_type="hybrid", DDPM_ratio=1.0, num_samples=742):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.num_samples = num_samples

        # Custom dataset for DDPM / CM / hybrid samples
        self.train_dataset = Pems04Dataset(
            num_samples=num_samples,
            device="cpu",
            data_type=data_type,
            DDPM_ratio=DDPM_ratio
        )

        # Load original data loaders and scalers
        self.train_loader_orig, self.valid_loader, self.test_loader, self.scaler, self.mean_scaler = get_dataloader(
            batch_size=batch_size,
            device=device,
            val_len=0.1,                    # Use 10% of data for validation
            missing_pattern="point",        # Type of missingness
            is_interpolate=True,            # Use interpolation
            num_workers=num_workers,
            target_strategy="hybrid"        # Strategy for target selection
        )

    def train_dataloader(self):
        # Return DataLoader using the custom dataset, with batch_size=1
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        # Return validation DataLoader from get_dataloader
        return self.valid_loader

    def test_dataloader(self):
        # Return test DataLoader from get_dataloader
        return self.test_loader


if __name__ == "__main__":
    pass