import sys
sys.path.append("./CSDI-TurboDTI")
from dataset_pems08 import get_dataloader
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



class Pems08Dataset(Dataset):
    def __init__(self, num_samples=780, device="cpu", data_type="DDPM", DDPM_ratio=0.5):
        self.num_samples = num_samples
        self.device = device

        # Define base paths for DDPM and CM samples
        DDPM_base = "./CSDI-TurboDTI/sample_PeMS08_DDPM/sample_{}.pt"
        CM_base = "./CSDI-TurboDTI/sample_PeMS08_CM/sample_{}.pt"

        # Select data paths based on the specified data type
        if data_type == "DDPM":
            self.paths = [DDPM_base.format(i) for i in range(num_samples)]
        elif data_type == "CM":
            self.paths = [CM_base.format(i) for i in range(num_samples)]
        elif data_type == "hybrid":
            # Use a mix of DDPM and CM samples according to DDPM_ratio
            self.paths = []
            for i in range(num_samples):
                if random.random() < DDPM_ratio:
                    self.paths.append(DDPM_base.format(i))
                else:
                    self.paths.append(CM_base.format(i))
        else:
            print("error data_type")

    def __len__(self):
        # Return total number of samples
        return self.num_samples

    def __getitem__(self, idx):
        # Load the sample at the given index
        path = self.paths[idx]
        data = torch.load(path, map_location="cpu", weights_only=True)

        # Return relevant data fields as a dictionary
        return {
            'observed_data': data['observed_data'],
            'observed_mask': data['observed_mask'],
            'observed_tp': data['observed_tp'],
            'cond_mask': data['cond_mask'],
            'coeffs': data['coeffs'],
            'teacher_sample': data['teacher_sample']
        }

    



class PeMS08DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, device="cuda:0", num_workers=16, data_type="hybrid", DDPM_ratio=1.0, num_samples=780):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.num_samples = num_samples

        # Initialize custom training dataset (DDPM / CM / hybrid)
        self.train_dataset = Pems08Dataset(
            num_samples=num_samples,
            device="cpu",
            data_type=data_type,
            DDPM_ratio=DDPM_ratio
        )

        # Load validation/test data and scalers using helper function
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
        # Return training DataLoader (custom dataset, batch_size=1)
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