from pems04_datamodule import PeMS04DataModule
from pems04_lightningmodule import PeMS04LightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch

device_num = 0
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="traffic.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
parser.add_argument("--nsample", type=int, default=10)


args = parser.parse_args()
print(args)

path = "./PriSTI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'pems-04'
config["seed"] = 42

seeds = [1, 2, 3, 4, 5]
results = []  # To store results for each seed

for seed in seeds:
    print(f"Running with seed: {seed}")
    config["seed"] = seed
    seed_everything(seed)  # Set random seed for reproducibility

    # Initialize the model with given config and device settings
    model = PeMS04LightningModule(config, device=args.device, target_dim=307, seq_len=24)
    
    # Load pretrained generator weights
    model.generator.load_state_dict(torch.load("./PriSTI-TurboDTI/save/PEMS04_point_20241108_205556/model.pth"))

   
    dm = PeMS04DataModule()
    #model.showcoeff()

 
    model.list1 = [49, 42, 41, 10] 

    trainer = pl.Trainer(
        #accelerator="auto",
        devices = [device_num],
    )

    #trainer.test(model=model,dataloaders=dm.test_dataloader())
    trainer.fit(model=model,train_dataloaders=dm.train_dataloader())
    with open("Pems04_results.txt", "a") as file:  # "a" mode to append each epoch result
        file.write(f"Seed {seed}: MAE={model.MAE}, MSE={model.MSE}\n")