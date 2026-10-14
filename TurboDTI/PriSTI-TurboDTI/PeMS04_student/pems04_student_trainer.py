
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
parser.add_argument("--config", type=str, default="traffic.yaml")
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
config["diffusion"]["adj_file"] = 'pems-04'
config["seed"] = 42


seed = 42
print(f"Running with seed: {seed}")
config["seed"] = seed
seed_everything(seed)


# Set loss weighting factors for ground truth and knowledge distillation
alpha_gt = 0.05
alpha_kd = 1.0

# Define training epoch number
epoch_num = 200

# Total number of training samples
total_samples = 742 * 1

# Ratio of DDPM data samples in the dataset
DDPM_ratio = 1.0

# Flag to indicate whether to initialize the student model from a pretrained checkpoint
is_init_flag = False

# Flag to indicate whether to start training
train_flag = True

# Print hyperparameter settings
print("alpha_gt:", alpha_gt)
print("alpha_kd:", alpha_kd)
print("epoch_num:", epoch_num)


if train_flag:
    student = Pems04StudentLightningModule(config=config,device=args.device,target_dim=307,seq_len=24,alpha_gt=alpha_gt,alpha_kd=alpha_kd,is_init=is_init_flag)
else:
    checkpoint_path = "./PriSTI-TurboDTI/experiments/parameter_study/pems04_pristi/2*742/last-v1.ckpt"
    student = Pems04StudentLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=307,seq_len=24,alpha_gt=alpha_gt,alpha_kd=alpha_kd)


dm = PeMS04DataModule(batch_size=16,device=f'cuda:{device_num}',num_workers=16,data_type="hybrid",DDPM_ratio=DDPM_ratio,num_samples=total_samples)


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',                  # Monitor validation loss to decide best checkpoint
    dirpath=f'./experiments/ablation_study/pems04_pristi/1_742/woinit',  # Directory to save checkpoints
    filename='pems04-{alpha_gt:.4f}-{alpha_kd:.4f}-{lr:.4f}-{epoch:02d}-{val_loss:.4f}',  
    save_top_k=1,                      # Save only the best checkpoint
    save_last=True,                    # Also save the last checkpoint
    mode='min',                       
    verbose=True                      
)

trainer = pl.Trainer(
        devices = [device_num],          # Specify the device(s) to use
        #max_epochs = epoch_num,
        max_epochs = 200,                # Set maximum number of epochs
        #max_steps = 742*2*100,
        callbacks=[checkpoint_callback],# Register the checkpoint callback
        check_val_every_n_epoch = 1,    # Validate every epoch
    )


if train_flag:
    trainer.fit(model=student,train_dataloaders=dm.train_dataloader())
trainer.test(model=student,dataloaders=dm.test_dataloader())
#trainer.validate(model=student,dataloaders=dm.test_dataloader())
print("alpha_gt:",alpha_gt)
print("alpha_kd",alpha_kd)
print("epoch_num:",epoch_num)
print("DDPM_ratio:",DDPM_ratio)
print("total_samples",total_samples)

