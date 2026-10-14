import sys
sys.path.append("./PriSTI-TurboDTI")

import pytorch_lightning as pl
from main_model import PriSTI_Pems04
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
    
    theta = torch.quantile(valid_errors, 0.6).detach()

    loss_matrix = torch.where(
        abs_errors <= theta,
        0.5 * (abs_errors ** 2),
        theta * (abs_errors - 0.5 * theta)
    )
    return loss_matrix.sum() / (eval_mask.sum() + 1e-8)

class Pems04StudentLightningModule(pl.LightningModule):
    def __init__(self, config, device, target_dim=307, seq_len=24, alpha_gt=1, alpha_kd=1, is_init=True):
        super().__init__()
        # Load training data mean and std for normalization
        path = "./PriSTI-TurboDTI/data/PEMS04/pems04_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

        # Initialize teacher and student models with given configuration and dimensions
        self.teacher = PriSTI_Pems04(config, device, target_dim, seq_len)
        self.student = PriSTI_Pems04(config, device, target_dim, seq_len)

        # Load pre-trained weights into student if specified
        if is_init:
            self.student.load_state_dict(torch.load("./PriSTI-TurboDTI/save/PEMS04_point_20241108_205556/model.pth"))

        # Load pre-trained weights into teacher model
        self.teacher.load_state_dict(torch.load("./PriSTI-TurboDTI/save/PEMS04_point_20241108_205556/model.pth"))

        # Freeze teacher parameters to prevent updates during training
        self.freeze_teacher()

        self.step_counter = 0 
        self.alpha_gt = alpha_gt
        self.alpha_kd = alpha_kd
    
    def freeze_teacher(self):
        # Disable gradient computation for all teacher model parameters
        for param in self.teacher.parameters():
            param.requires_grad = False


    def on_epoch_start(self):
        self.step_counter = 0  



    def training_step(self, batch, batch_idx):
        # Move batch data to the target device and squeeze batch dimension
        observed_data = batch['observed_data'].squeeze(0).to(self.device)
        observed_mask = batch['observed_mask'].squeeze(0).to(self.device)
        observed_tp = batch['observed_tp'].squeeze(0).to(self.device)
        cond_mask = batch['cond_mask'].squeeze(0).to(self.device)
        coeffs = batch['coeffs'].squeeze(0).to(self.device)
        teacher_sample = batch['teacher_sample'].squeeze(0).to(self.device)

        # Get additional side information from the student model
        side_info = self.student.get_side_info(observed_tp, cond_mask)
        itp_info = coeffs.unsqueeze(1)
        eval_mask = observed_mask - cond_mask  # Mask indicating evaluation points
        num_eval = eval_mask.sum()  # Number of evaluation points

        # Sample initial noise for diffusion input
        x_T = torch.randn_like(observed_data)
        diff_input = ((1 - cond_mask) * x_T).unsqueeze(1)

        # Predict denoised output using the student's diffusion model
        predicted = self.student.diffmodel(diff_input, torch.tensor([self.student.num_steps - 1]).to(self.device), side_info, itp_info, cond_mask)
        predicted_x0 = x_T - predicted

        # Calculate knowledge distillation loss (student vs teacher)
        loss_kd = 0
        if self.alpha_kd != 0:
            #loss_kd = ((torch.abs(predicted_x0 - teacher_sample) ** 2) * eval_mask).sum() / num_eval
            loss_kd = compute_dynamic_huber_loss(teacher_sample,  predicted_x0, eval_mask)
        # Calculate ground truth reconstruction loss (student vs observed data)
        loss_gt = 0
        if self.alpha_gt != 0:
            #loss_gt = ((torch.abs(predicted_x0 - observed_data) ** 2) * eval_mask).sum() / num_eval
            loss_gt = compute_dynamic_huber_loss(observed_data, predicted_x0 ,eval_mask)
        # Combine losses with respective weights
        loss = self.alpha_gt * loss_gt + self.alpha_kd * loss_kd

        # Log losses for monitoring
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("kd_loss", self.alpha_kd * loss_kd, prog_bar=True, on_epoch=True)
        self.log("gt_loss", self.alpha_gt * loss_gt, prog_bar=True, on_epoch=True)

        self.step_counter += 1
        return loss



    def validation_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
            rank_of_cond_mask
        ) = self.student.process_data(batch)
        side_info = self.student.get_side_info(observed_tp, cond_mask)
        itp_info = coeffs.unsqueeze(1)
        eval_mask = observed_mask - cond_mask
        num_eval = eval_mask.sum()
        x_T = torch.randn_like(observed_data)
        diff_input = ((1-cond_mask)*x_T).unsqueeze(1)
        predicted = self.student.diffmodel(diff_input,torch.tensor([self.student.num_steps-1]).to(self.device),side_info,itp_info,cond_mask)
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
        self.total_time = 0
        self.test_start_time = time.time()  

    def test_step(self, batch, batch_idx):
        # Convert normalization stats to tensors on the correct device
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()
        
        # Process input batch to extract necessary components
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
            itp_info = coeffs.unsqueeze(1)
 

            

            x_T = torch.randn_like(observed_data)
            diff_input = ((1-cond_mask)*x_T).unsqueeze(1)
            predicted = self.student.diffmodel(diff_input,torch.tensor([self.student.num_steps-1]).to(self.device),side_info,itp_info,cond_mask)
            sample = x_T - predicted

            
            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0

            final_sample = sample.permute(0,2,1).to(self.device) #(B,L,K)
            c_target = observed_data.permute(0,2,1)
            eval_points = target_mask.permute(0,2,1)
            observed_points = observed_mask.permute(0,2,1)
            
         


            self.test_end_time = time.time()  
            self.test_duration = self.test_end_time - self.test_start_time 

         
            mae_current = (
                torch.abs((final_sample-c_target)*eval_points)
            )*scaler
            mse_current = (
                ((final_sample-c_target)*eval_points)**2
            )*(scaler**2)
    
            mre_current = (
                torch.abs(((final_sample*scaler) - (c_target*scaler))) / ((torch.abs(c_target)*scaler)+mean_scaler + 1e-8)  
            ) * eval_points
            

         
           
            self.total_mae += mae_current.sum().item()
            self.total_mse += mse_current.sum().item()
            self.total_mre += mre_current.sum().item()
            self.evalpoints_total += eval_points.sum().item()
          
            self.cnt += 1  
        
    def on_test_epoch_end(self):
        
        
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        MRE = self.total_mre / self.evalpoints_total
      
        
        # logs
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)
        self.log('test_MRE', MRE)
        self.log('test_duration', self.test_duration)  

        self.MAE = MAE
        self.MSE = MSE
        self.MRE = MRE

        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")
        print(f"Test Set Average MRE: {MRE}")
        print(f"Test Set Total Duration: {self.test_duration:.2f} seconds")  
        pass
    
    
    def configure_optimizers(self):
   
        optimizer_S = torch.optim.AdamW(
            self.student.parameters(), 
            lr=0.0001, 
            betas=(0.9, 0.999)
        )
        return optimizer_S
    
   
    
