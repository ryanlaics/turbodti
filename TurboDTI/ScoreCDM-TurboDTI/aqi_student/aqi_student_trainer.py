#from PeMS04_Datamodule_Pems04 import PeMS04DataModule
import sys
sys.path.append("./")
from aqi_student_datamodule import AQI36DataModule
from aqi_student_lightningmodule import AQI36StudentLightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch
import random

from pytorch_lightning.loggers import CSVLogger
# device_num = random.randint(0,4)
device_num = 1
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
#parser.add_argument("--nsample", type=int, default=100)


args = parser.parse_args()
print(args)

path = "./ScoreCDM-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'AQI36'
config["seed"] = 42


seed = 42
print(f"Running with seed: {seed}")
config["seed"] = seed
seed_everything(seed)


alpha_gt = 0.05  # Weight for ground truth loss
alpha_kd = 1.0   # Weight for knowledge distillation loss
epoch_num = 200  # Total number of training epochs
total_samples = 329  # Total samples to use (here just 780)
DDPM_ratio = 1.0  # Ratio of DDPM data in hybrid dataset
quantile = 0.9
###########################
# Flag to control model initialization from checkpoint
is_init_flag = True
###########################

train_flag = True  # Flag to enable/disable training
print("alpha_gt:", alpha_gt)
print("alpha_kd:", alpha_kd)
print("epoch_num:", epoch_num)


if train_flag:
    student = AQI36StudentLightningModule(config=config,device=args.device,target_dim=36,seq_len=36,alpha_gt=alpha_gt,alpha_kd=alpha_kd,is_init=is_init_flag,quantile=quantile)
else:
    checkpoint_path = "./PriSTI-TurboDTI/experiments/parameter_study/pems04_pristi/1*780/last-v1.ckpt"
    student = AQI36StudentLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=36,seq_len=36,alpha_gt=alpha_gt,alpha_kd=alpha_kd)


dm = AQI36DataModule(batch_size=16,device=f'cuda:{device_num}',num_workers=16,data_type="hybrid",DDPM_ratio=DDPM_ratio,num_samples=total_samples)
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',                  
    dirpath=f'./experiments/parameter_study/aqi_score/mix-ratio/0_10', 
    filename='pems04-{alpha_gt:.4f}-{alpha_kd:.4f}-{lr:.4f}-{epoch:02d}-{val_loss:.4f}',  
    save_top_k=1,                       
    save_last=True,
    mode='min',                          
    verbose=True                         
)
logger = CSVLogger(save_dir="experiments/logs", name="my_aqi_experiment")
trainer = pl.Trainer(
     
        devices = [device_num],
        max_steps = 329*28*30,
        max_epochs= -1,
        #max_epochs = epoch_num,
        #callbacks=[checkpoint_callback],
        #val_check_interval=5,
        check_val_every_n_epoch = 1,
        logger = logger,
    )
if train_flag:
    trainer.fit(model=student,train_dataloaders=dm.train_dataloader())
trainer.test(model=student,dataloaders=dm.test_dataloader())
trainer.logger.save()
#trainer.validate(model=student,dataloaders=dm.test_dataloader())
print("alpha_gt:",alpha_gt)
print("alpha_kd",alpha_kd)
print("epoch_num:",epoch_num)
print("DDPM_ratio:",DDPM_ratio)
