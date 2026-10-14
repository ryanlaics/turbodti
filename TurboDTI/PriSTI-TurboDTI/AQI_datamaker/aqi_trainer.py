import sys
sys.path.append("./PriSTI-TurboDTI")
from aqi_datamodule import AQIDataModule
from aqi_lightningmodule import AQILightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch


def uniform_descending_list(t):
    return [round(x) for x in 
            list(reversed([i * (99 / (t - 1)) for i in range(t)]))]
device_num = 0
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
#parser.add_argument("--nsample", type=int, default=100)


args = parser.parse_args()
print(args)

path = "./PriSTI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'AQI36'
config["seed"] = 42



seeds = [1,2,3,4,5]

results = [] 
MAE_results = []
MSE_results = []

for seed in seeds:
    print(f"Running with seed: {seed}")
    
    # Set the random seed for reproducibility
    config["seed"] = seed
    seed_everything(seed)

    # Initialize the model with the current config and device
    model = AQILightningModule(config=config, device=args.device, target_dim=36, seq_len=36)

    # Load pretrained weights for the generator
    model.generator.load_state_dict(torch.load("./PriSTI-TurboDTI/save/aqi36/model.pth"))

  

    dm = AQIDataModule()
    #model.showcoeff()

    model.list1 =    [99, 85, 83, 47, 44, 40, 13]
    
   
    trainer = pl.Trainer(
            #accelerator="auto",
            devices = [device_num],
        )
 

    trainer.fit(model=model,train_dataloaders=dm.train_dataloader())
    #trainer.test(model=model,dataloaders=dm.test_dataloader())
    
    with open("AQI_results.txt", "a") as file:  # "a" mode to append each epoch result
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
#MSE_results = [math.sqrt(i) for i in MSE_results]
MSE_mean = calculate_mean(MSE_results)
MSE_variance = calculate_variance(MSE_results, MSE_mean)
MSE_standard_deviation = calculate_standard_deviation(MSE_variance)

# Display the results
print(f"MSE_Mean: {MSE_mean}")
print(f"MSE_Variance: {MSE_variance}")
print(f"MSE_Standard Deviation: {MSE_standard_deviation}")