from pems08_datamodule import PeMS08DataModule
from pems08_lightningmodule import PeMS08LightningModule
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
############
parser.add_argument(
    "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
)
############



args = parser.parse_args()
print(args)

path = "./CSDI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'pems-08'
config["seed"] = 42

# Set the list of random seeds for reproducibility
seeds = [1, 2, 3, 4, 5]

# Store MAE and MSE results for each seed
MAE_results = []  
MSE_results = []

for seed in seeds:
    print(f"Running with seed: {seed}")
    config["seed"] = seed
    seed_everything(seed)  # Set seed for reproducibility

    # Initialize model with given config and device
    model = PeMS08LightningModule(config, device=args.device, target_dim=170, seq_len=24)

    # Load pre-trained generator weights
    model.generator.load_state_dict(
        torch.load("./CSDI-TurboDTI/save/pems08_coeffs_new2.0lr0_20250708_081248/model.pth", map_location="cuda:0")
    )

    # Create data module for PeMS08
    dm = PeMS08DataModule()

    
    #model.showcoeff()

###########################################################################################
 
    model.list1 = [49, 38, 24, 18, 17, 14, 5] 

    trainer = pl.Trainer(
        devices = [device_num],
    )
    #trainer.fit(model=model, train_dataloaders=dm.train_dataloader()) # prepare for S2 training
    trainer.test(model=model, dataloaders=dm.test_dataloader())  # Run testing on the test dataloader for evaluating S1
    with open("Pems08_results.txt", "a") as file:  # "a" mode to append each epoch result
        file.write(f"Seed {seed}: MAE={model.MAE}, MSE={model.MSE}\n")
    MAE_results.append(model.MAE)
    MSE_results.append(model.MSE)

import math

def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_variance(numbers, mean):
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)

def calculate_standard_deviation(variance):
    return math.sqrt(variance)



MAE_mean = calculate_mean(MAE_results)
MAE_variance = calculate_variance(MAE_results, MAE_mean)
MAE_standard_deviation = calculate_standard_deviation(MAE_variance)

# Display the results
print(f"MAE_Mean: {MAE_mean}")
print(f"MAE_Variance: {MAE_variance}")
print(f"MAE_Standard Deviation: {MAE_standard_deviation}")

print(MSE_results)
MSE_results = [math.sqrt(i) for i in MSE_results]
MSE_mean = calculate_mean(MSE_results)
MSE_variance = calculate_variance(MSE_results, MSE_mean)
MSE_standard_deviation = calculate_standard_deviation(MSE_variance)

# Display the results
print(f"MSE_Mean: {MSE_mean}")
print(f"MSE_Variance: {MSE_variance}")
print(f"MSE_Standard Deviation: {MSE_standard_deviation}")
print(model.n_samples)