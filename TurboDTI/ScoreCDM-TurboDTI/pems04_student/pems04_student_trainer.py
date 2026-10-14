#from PeMS04_Datamodule_Pems04 import PeMS04DataModule
import sys
sys.path.append("./")
from pems04_student_datamodule import PeMS04DataModule
from pems04_student_lightningmodule import Pems04StudentLightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch
import random

# device_num = random.randint(0,4)
device_num = 0
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="traffic1.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
#parser.add_argument("--nsample", type=int, default=100)


args = parser.parse_args()
print(args)

path = "./config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'pems-04'
config["seed"] = 42


seed = 42
print(f"Running with seed: {seed}")
config["seed"] = seed
seed_everything(seed)


alpha_gt = 0.05  # Weight for ground truth loss
alpha_kd = 1.0   # Weight for knowledge distillation loss
epoch_num = 200  # Total number of training epochs
total_samples = 742 * 1  # Total samples to use (here just 780)
DDPM_ratio = 1.0  # Ratio of DDPM data in hybrid dataset

###########################
# Flag to control model initialization from checkpoint
is_init_flag = True
###########################

train_flag = True  # Flag to enable/disable training
print("alpha_gt:", alpha_gt)
print("alpha_kd:", alpha_kd)
print("epoch_num:", epoch_num)


if train_flag:
    student = Pems04StudentLightningModule(config=config,device=args.device,target_dim=307,seq_len=24,alpha_gt=alpha_gt,alpha_kd=alpha_kd,is_init=is_init_flag)
else:
    checkpoint_path = "./PriSTI-TurboDTI/experiments/parameter_study/pems04_pristi/1*780/last-v1.ckpt"
    student = Pems04StudentLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=307,seq_len=24,alpha_gt=alpha_gt,alpha_kd=alpha_kd)


dm = PeMS04DataModule(batch_size=16,device=f'cuda:{device_num}',num_workers=16,data_type="hybrid",DDPM_ratio=DDPM_ratio,num_samples=total_samples)


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',                  
    dirpath=f'./experiments/parameter_study/pems04_score/mix-ratio/0_10', 
    filename='pems04-{alpha_gt:.4f}-{alpha_kd:.4f}-{lr:.4f}-{epoch:02d}-{val_loss:.4f}',  
    save_top_k=1,                       
    save_last=True,
    mode='min',                          
    verbose=True                         
)

trainer = pl.Trainer(
        #accelerator="auto",
        devices = [device_num],
        #max_epochs = epoch_num,
        max_epochs = 200,
        #max_steps = 780*2*100,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch = 1,
    )
if train_flag:
    trainer.fit(model=student,train_dataloaders=dm.train_dataloader())
trainer.test(model=student,dataloaders=dm.test_dataloader())
#trainer.validate(model=student,dataloaders=dm.test_dataloader())
print("alpha_gt:",alpha_gt)
print("alpha_kd",alpha_kd)
print("epoch_num:",epoch_num)
print("DDPM_ratio:",DDPM_ratio)