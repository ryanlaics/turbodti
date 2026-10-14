import sys
sys.path.append("./CSDI-TurboDTI")  # Append local project path

from dataset_aqi36 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Use default CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data loaders with interpolation enabled
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    batch_size=16,
    device=device,
    val_len=0.1,
    is_interpolate=True,
    num_workers=16,
    target_strategy="hybrid",
    mask_sensor=[]
)

class LimitedDataLoader:
    """
    A wrapper for limiting the number of batches returned by a DataLoader.
    Useful for debugging or evaluating on a smaller subset.
    """
    def __init__(self, dataloader, limit):
        self.dataloader = list(dataloader)[:limit]
        self.limit = limit

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return min(len(self.dataloader), self.limit)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.dataloader):
            raise IndexError(f"Index {index} out of range for LimitedDataLoader with length {len(self.dataloader)}")
        return self.dataloader[index]




class AQIDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        # Return a limited number of training batches for controlled training
        return LimitedDataLoader(train_loader, limit=500)
    
    def val_dataloader(self):
        return valid_loader
    
    def test_dataloader(self):
        return test_loader





  