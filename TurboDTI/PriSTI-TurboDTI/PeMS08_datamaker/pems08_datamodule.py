import sys
sys.path.append("./PriSTI-TurboDTI")
from dataset_pems08 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml



SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open("./PriSTI-TurboDTI/config/traffic.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = 'pems-08'
config["seed"] = SEED


train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=16, device="cuda:0", val_len=0.1, missing_pattern="point",
        is_interpolate=True, num_workers=64,
        target_strategy="hybrid",
    )

class LimitedDataLoader:
    def __init__(self, dataloader, limit):
        self.dataloader = dataloader  # Original dataloader to wrap
        self.limit = limit            # Limit the number of batches to iterate

    def __iter__(self):
        batch_count = 0
        for batch in self.dataloader:
            if batch_count >= self.limit:  # Stop iteration when limit reached
                break
            yield batch
            batch_count += 1

    def __len__(self):
        # Return the limited length or the original length if smaller
        return min(len(self.dataloader), self.limit)



class PeMS08DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return train_loader  # Return full training data loader
    
    def val_dataloader(self):
        return LimitedDataLoader(valid_loader, limit=20)  # Limit validation batches to 20
    
    def test_dataloader(self):
        # return test_loader
        return LimitedDataLoader(test_loader, limit=100000)  # Limit test batches to 100,000 (practically unlimited)



if __name__ == "__main__":
    dm = PeMS08DataModule()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        this = batch
        break
    print(this)
  