import pytorch_lightning as pl
from main_model_aqi import CSDI_PEMS08
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def compute_dynamic_huber_loss(label_data, imputed_results, eval_mask):
    abs_errors = torch.abs(label_data - imputed_results) * eval_mask
    valid_errors = abs_errors[eval_mask > 0]
    
    if valid_errors.numel() == 0:
        return torch.tensor(0.0, device=abs_errors.device, requires_grad=True)
    
    theta = torch.quantile(valid_errors, 0.75).detach()

    loss_matrix = torch.where(
        abs_errors <= theta,
        0.5 * (abs_errors ** 2),
        theta * (abs_errors - 0.5 * theta)
    )
    return loss_matrix.sum() / (eval_mask.sum() + 1e-8)


class Pems08StudentLightningModule(pl.LightningModule):
    def __init__(self, config, device, target_dim=170, seq_len=24, alpha_gt=1, alpha_kd=1, epoch=100, lr=0.001, is_initialized=True):
        super().__init__()
        # Load training data mean and std for normalization
        path = "./CSDI-TurboDTI/data/PEMS08/pems08_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

        self.lr = lr
        self.epoch = epoch

        # Initialize teacher and student models
        self.teacher = CSDI_PEMS08(config, device, target_dim)
        self.student = CSDI_PEMS08(config, device, target_dim)

        # Load pretrained weights for student if specified
        if is_initialized:
            self.student.load_state_dict(torch.load(
                "./CSDI-TurboDTI/save/pems08_coeffs_new2.0lr0_20250708_081248/model.pth",
                map_location=device
            ))

        # Always load pretrained weights for teacher
        self.teacher.load_state_dict(torch.load(
            "./CSDI-TurboDTI/save/pems08_coeffs_new2.0lr0_20250708_081248/model.pth",
            map_location=device
        ))

        # Freeze teacher parameters during training
        self.freeze_teacher()

        self.step_counter = 0
        self.alpha_gt = alpha_gt  # Weight for ground truth loss
        self.alpha_kd = alpha_kd  # Weight for knowledge distillation loss

    def freeze_teacher(self):
        # Disable gradients for teacher model parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

    def on_epoch_start(self):
        self.step_counter = 0  

    def training_step(self,batch,batch_idx):
        
        
        observed_data = batch['observed_data'].squeeze(0).to(self.device)
        observed_mask = batch['observed_mask'].squeeze(0).to(self.device)
        observed_tp = batch['observed_tp'].squeeze(0).to(self.device)
        cond_mask = batch['cond_mask'].squeeze(0).to(self.device)
        coeffs = batch['coeffs'].squeeze(0).to(self.device)
        teacher_sample = batch['teacher_sample'].squeeze(0).to(self.device)

     
        side_info = self.student.get_side_info(observed_tp, cond_mask)
        cond_obs = coeffs.unsqueeze(1)
        eval_mask = observed_mask - cond_mask
        num_eval = eval_mask.sum()

        x_T = torch.randn_like(observed_data)
        noisy_target = ((1-cond_mask)*x_T).unsqueeze(1)
        diff_input = torch.cat([cond_obs,noisy_target],dim=1) #(B,2,K,L)
        predicted = self.student.diffmodel(diff_input,side_info,torch.tensor([self.student.num_steps-1]).to(self.device))
        predicted_x0 = x_T - predicted # parameterization of one-pass distilled model
        
        loss_kd = 0
        loss_gt = 0
        if self.alpha_kd != 0:
            #loss_kd = ((torch.abs(predicted_x0-teacher_sample)**2)*eval_mask).sum()/num_eval
            loss_kd = compute_dynamic_huber_loss(teacher_sample,  predicted_x0, eval_mask)
        if self.alpha_gt != 0:
            #loss_gt = ((torch.abs(predicted_x0-observed_data)**2)*eval_mask).sum()/num_eval
            loss_gt = compute_dynamic_huber_loss(observed_data, predicted_x0 ,eval_mask)
        # hybrid loss
        loss = self.alpha_gt*loss_gt + self.alpha_kd*loss_kd
        self.log("train_loss",loss,prog_bar=True,on_epoch=True)
        self.log("kd_loss",self.alpha_kd*loss_kd,prog_bar=True,on_epoch=True)
        self.log("gt_loss",self.alpha_gt*loss_gt,prog_bar=True,on_epoch=True)
        self.step_counter += 1
        return loss

        
     
    
    def validation_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        ) = self.student.process_data(batch)
        side_info = self.student.get_side_info(observed_tp, cond_mask)
        eval_mask = observed_mask - cond_mask
        num_eval = eval_mask.sum()
        cond_obs = coeffs.unsqueeze(1)

        x_T = torch.randn_like(observed_data)
        noisy_target = ((1-cond_mask)*x_T).unsqueeze(1)
        diff_input = torch.cat([cond_obs,noisy_target],dim=1) #(B,2,K,L)
        predicted = self.student.diffmodel(diff_input,side_info,torch.tensor([self.student.num_steps-1]).to(self.device))
        predicted_x0 = x_T - predicted
        loss = (torch.abs(predicted_x0-observed_data)**2*eval_mask).sum().item()/num_eval
        self.log("val_loss",loss,prog_bar=True,on_epoch=True)
        return loss
    
    def on_test_start(self):
        self.cnt = 0
        self.total_mae = 0
        self.total_mse = 0
        self.total_mre = 0
        self.evalpoints_total = 0


    def test_step(self, batch, batch_idx):
        # Convert train std and mean to torch tensors on the device
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()

        # Unpack batch data via student's data processing method
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.student.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.student.get_side_info(observed_tp, cond_mask)

            cond_obs = coeffs.unsqueeze(1)
            B, K, L = cond_mask.size()

            # Generate random noise tensor matching observed data shape
            x_T = torch.randn_like(observed_data)
            noisy_target = ((1 - cond_mask) * x_T).unsqueeze(1)
            diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # Shape: (B, 2, K, L)

            # Model prediction at the last diffusion step
            predicted = self.student.diffmodel(
                diff_input, side_info, torch.tensor([self.student.num_steps - 1]).to(self.device)
            )
            predicted_x0 = x_T - predicted

            # Avoid double evaluation by masking positions indicated by cut_length
            for i in range(len(cut_length)):
                target_mask[i, ..., 0 : cut_length[i].item()] = 0

            # Prepare tensors for error calculation by permuting dimensions
            final_sample = predicted_x0.permute(0, 2, 1).to(self.device)  # (B, L, K)
            c_target = observed_data.permute(0, 2, 1)
            eval_points = target_mask.permute(0, 2, 1)
            observed_points = observed_mask.permute(0, 2, 1)

            for i in range(len(cut_length)):
                eval_points[i, ..., 0 : cut_length[i].item()] = 0

            # Calculate MAE, MSE and MRE weighted by scaler and evaluation mask
            mae_current = torch.abs((final_sample - c_target) * eval_points) * scaler
            mse_current = ((final_sample - c_target) * eval_points) ** 2 * (scaler ** 2)
            mre_current = (
                torch.abs((final_sample * scaler) - (c_target * scaler))
                / (torch.abs(c_target) * scaler + mean_scaler + 1e-8)  # prevent division by zero
            ) * eval_points

            # Accumulate metrics for the epoch
            self.total_mae += mae_current.sum().item()
            self.total_mse += mse_current.sum().item()
            self.total_mre += mre_current.sum().item()
            self.evalpoints_total += eval_points.sum().item()


        
    def on_test_epoch_end(self):
        
        
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        MRE = self.total_mre / self.evalpoints_total
      
        
        # logs
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)
        self.log('test_MRE', MRE)


        self.MAE = MAE
        self.MSE = MSE
        self.MRE = MRE

        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")
        print(f"Test Set Average MRE: {MRE}")
        pass
    

    # Optimizer config
    def configure_optimizers(self):
   
        optimizer_S = torch.optim.AdamW(
            self.student.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.999)
        )
        return optimizer_S
    


