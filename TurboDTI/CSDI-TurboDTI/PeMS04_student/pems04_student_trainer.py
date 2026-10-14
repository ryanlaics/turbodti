
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
device_num  = 0
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="traffic.yaml")
parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)



args = parser.parse_args()
print(args)

path = "./CSDI-TurboDTI/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)


config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'pems-04'

seed = 42
print(f"Running with seed: {seed}")
config["seed"] = seed
seed_everything(seed)


# loss coefficients
alpha_gt = 0.05
alpha_kd = 1.0
# for parameter warm up
is_initialized = True
lr =  1e-4
epoch_num = 200
DDPM_ratio = 1.0
print("alpha_gt:",alpha_gt)
print("alpha_kd",alpha_kd)
print("epoch_num:",epoch_num)
print("lr:",lr)

checkpoint_path = "./CSDI-TurboDTI/experiments/ablation_study/dominant_kd/last.ckpt"

#student = AqiStudentLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=36,seq_len=36,alpha_gt=alpha_gt,alpha_kd=alpha_kd,map_location=f"cuda:{device_num}")
student = Pems04StudentLightningModule(config=config,device=args.device,target_dim=307,seq_len=24,alpha_gt=alpha_gt,alpha_kd=alpha_kd,epoch=epoch_num,lr=lr,is_initialized=is_initialized)

dm = PeMS04DataModule(batch_size=16,device=f'cuda:{device_num}',num_workers=32,data_type="hybrid",DDPM_ratio=DDPM_ratio,num_samples=742*1)



checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',                
    dirpath=f'./CSDI-TurboDTI/experiments/ablation_study/pems04/dominant_kd',
    filename='pems04-{alpha_gt:.4f}-{alpha_kd:.4f}-{lr:.4f}-{epoch:02d}-{val_loss:.4f}', 
    save_top_k=1,                       
    save_last=True,
    mode='min',                          
    verbose=True                         
) # save checkpoints 
trainer = pl.Trainer(
     
        devices = [device_num],
        #max_steps = 742*200,
        #max_epochs= -1,
        max_epochs = epoch_num,
        callbacks=[checkpoint_callback],
        #val_check_interval=5,
        check_val_every_n_epoch = 1,
    )

#trainer.fit(model=student,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())
trainer.test(model=student,dataloaders=dm.test_dataloader())

print("alpha_gt:",alpha_gt)
print("alpha_kd",alpha_kd)
print("epoch_num:",epoch_num)
print("lr:",lr)
print("hybrid_ratio:",DDPM_ratio)
print("is init:",is_init_flag)