import sys
sys.path.append("./PriSTI-TurboDTI")
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

with open("./PriSTI-TurboDTI/config/traffic.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = 'pems-04'
config["seed"] = SEED


train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    batch_size=16, device="cuda:0", val_len=0.1, missing_pattern="point",
    is_interpolate=True, num_workers=64,
    target_strategy="hybrid",
)

class LimitedDataLoader:
    def __init__(self, dataloader, limit):
        self.dataloader = dataloader
        self.limit = limit

    def __iter__(self):
        batch_count = 0
        # Yield batches until limit is reached
        for batch in self.dataloader:
            if batch_count >= self.limit:
                break
            yield batch
            batch_count += 1

    def __len__(self):
        # Return the smaller of original length or limit
        return min(len(self.dataloader), self.limit)


class PeMS04DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        # Use full training loader
        return train_loader
    
    def val_dataloader(self):
        # Use limited validation loader for faster validation
        return LimitedDataLoader(valid_loader, limit=20)
    
    def test_dataloader(self):
        # Use limited test loader to speed up testing
        #return test_loader
        return LimitedDataLoader(test_loader, limit=1000)

    


if __name__ == "__main__":
    dm = PeMS04DataModule()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        this = batch
        break
    print(this)
  