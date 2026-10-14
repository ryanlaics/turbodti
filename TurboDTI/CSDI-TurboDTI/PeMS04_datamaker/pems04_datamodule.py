

import sys
sys.path.append("./CSDI-TurboDTI")

from dataset_pems04 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml

# from main_model import PriSTI_aqi36

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



# get dataloader
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=16, device="cuda:0", val_len=0.1, missing_pattern="point",
        is_interpolate=True, num_workers=16,
        target_strategy="hybrid",
    )

# A wrapper class to limit the number of batches returned by a dataloader
class LimitedDataLoader:
    def __init__(self, dataloader, limit=1000000):
        self.dataloader = dataloader  # Original dataloader
        self.limit = limit  # Maximum number of batches to yield

    def __iter__(self):
        batch_count = 0
        for batch in self.dataloader:
            if batch_count >= self.limit:
                break  # Stop when limit is reached
            yield batch  # Yield batch from original dataloader
            batch_count += 1

    def __len__(self):
        # Length is the smaller of the original dataloader or the set limit
        return min(len(self.dataloader), self.limit)


# PyTorch Lightning DataModule customized for PeMS04 dataset
class PeMS04DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return train_loader  # Use full training loader

    def val_dataloader(self):
        # Use validation loader with a batch limit of 20 
        # validation for evolutionary search
        return LimitedDataLoader(valid_loader, limit=20)

    def test_dataloader(self):
        # Use test loader with a batch limit of 100000
        return LimitedDataLoader(test_loader, limit=100000)

    


  