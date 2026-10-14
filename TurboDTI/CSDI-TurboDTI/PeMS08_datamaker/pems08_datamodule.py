

import sys
sys.path.append("./CSDI-TurboDTI")

from dataset_pems08 import get_dataloader
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
        self.dataloader = dataloader
        self.limit = limit

    def __iter__(self):
        batch_count = 0
        for batch in self.dataloader:
            if batch_count >= self.limit:
                break
            yield batch
            batch_count += 1

    def __len__(self):
        return min(len(self.dataloader), self.limit)

# PyTorch Lightning DataModule customized for PeMS08 dataset
class PeMS08DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return train_loader
    
    def val_dataloader(self):
        # Use validation loader with a batch limit of 20 
        # validation for evolutionary search
        return LimitedDataLoader(valid_loader,limit=20)
    
    def test_dataloader(self):
        #return test_loader
        return LimitedDataLoader(test_loader,limit=100000)
    



  