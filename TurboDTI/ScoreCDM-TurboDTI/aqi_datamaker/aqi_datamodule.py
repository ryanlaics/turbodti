import sys
sys.path.append("./")

from dataset_aqi36 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=16, device="cuda:0", val_len=0.1, 
        is_interpolate=True, num_workers=64,
        target_strategy="hybrid", mask_sensor=[]
    )

class LimitedDataLoader:
    def __init__(self, dataloader, limit):
        # Convert original DataLoader to list and truncate to limit
        self.dataloader = list(dataloader)[:limit]
        self.limit = limit

    def __iter__(self):
        # Return iterator over limited batches
        return iter(self.dataloader)

    def __len__(self):
        # Return the number of batches available (capped by limit)
        return min(len(self.dataloader), self.limit)

    def __getitem__(self, index):
        # Support indexing with range check
        if index < 0 or index >= len(self.dataloader):
            raise IndexError(f"Index {index} out of range for LimitedDataLoader with length {len(self.dataloader)}")
        return self.dataloader[index]





class AQI36DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return LimitedDataLoader(train_loader,limit=50000)
    
    def val_dataloader(self):
        return valid_loader
    
    def test_dataloader(self):
        return test_loader
        


if __name__ == "__main__":
    dm = AQI36DataModule()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        this = batch
        break
    print(this)
  