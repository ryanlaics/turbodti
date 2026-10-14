n_samples = 10
list1 = [99, 85, 81, 41, 40, 25, 10]
import pytorch_lightning as pl
from main_model import PriSTI_aqi36
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from dpm_solver_pytorch import NoiseScheduleVP,model_wrapper,DPM_Solver
import torchcde

#from thop import profile 


class AQILightningModule(pl.LightningModule):
    def __init__(self,config,device,target_dim=36,seq_len=36):
        super().__init__()
        path = "./PriSTI-TurboDTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.MAE = 0
        self.MSE = 0
        
        self.list2 = [99]
        # noise predictor epsilon theta
        self.generator = PriSTI_aqi36(config,device,target_dim,seq_len)
        self.generator.load_state_dict(torch.load("./PriSTI-TurboDTI/save/aqi36/model.pth"))
        self.factor_KL = 1.0
        self.step_counter = 0 
        self.batch_counter = 0
        self.n_samples = 1
        self.list1 = [99]

    def on_epoch_start(self):

        pass

    def training_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
        ) = self.generator.process_data(batch)
        
        # Path to save generated samples for current training step
        # path = f"./PriSTI-TurboDTI/sample_AQI_DDPM/sample_{self.step_counter}.pt"
        path = f"./PriSTI-TurboDTI/sample_AQI_CM/sample_{self.step_counter}.pt"
        
        # Obtain side information used for imputation conditioning
        side_info = self.generator.get_side_info(observed_tp, cond_mask)
        itp_info = coeffs.unsqueeze(1)
        
        with torch.no_grad():  
            # Generate multiple imputed samples using the model

            # samples = self.CMimpute(observed_data, cond_mask, side_info, self.n_samples, itp_info, consis_flag=1, phased_flag=1, observed_mask=observed_mask)
            samples = self.impute(observed_data, cond_mask, side_info, 100, itp_info)
            
            # Aggregate samples by median along the sample dimension
            teacher_sample = samples.median(dim=1).values

          
        data = {}

 
        data['observed_data'] = torch.tensor(observed_data, dtype=torch.float32)  
        data['cond_mask'] = torch.tensor(cond_mask, dtype=torch.float32) 
        data['observed_tp'] = torch.tensor(observed_tp, dtype=torch.float32)
        data['observed_mask'] = torch.tensor(observed_mask, dtype=torch.float32)
        data['coeffs'] = torch.tensor(coeffs, dtype=torch.float32)
        data['teacher_sample'] = torch.tensor(teacher_sample, dtype=torch.float32)

    
        if self.step_counter < 20001:
            torch.save(data, path)

        self.step_counter += 1
    
        

             
    # no need for validation
    def validation_step(self, batch, batch_idx):
        pass
        

    def on_test_start(self):
        self.cnt = 0
        self.total_mae = 0
        self.total_mse = 0
        self.total_mre = 0
        self.evalpoints_total = 0
        self.total_latency = 0 
        self.total_flops = 0 
        self.batch_count = 0 

    def test_step(self,batch,batch_idx):
        
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.generator.process_data(batch)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  
        with torch.no_grad():
           
            cond_mask = gt_mask
           
            target_mask = observed_mask - cond_mask
            side_info = self.generator.get_side_info(observed_tp, cond_mask)
            itp_info = coeffs.unsqueeze(1)
            for i in range(1):
                start_event.record()
                #################################################
                # current_sample = torch.randn_like(observed_data)
                # diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                # t = torch.tensor([1], dtype=torch.long).to(self.device)
                # predicted = self.generator.diffmodel(diff_input,torch.tensor([t]).to(self.device), side_info,  itp_info, cond_mask)
                # samples = predicted.unsqueeze(1)
                #################################################
                #samples = self.impute(observed_data, cond_mask, side_info, self.n_samples, itp_info)
                #samples = self.CMimpute(observed_data, cond_mask,  side_info, self.n_samples, itp_info,consis_flag=1,phased_flag=1,observed_mask=observed_mask)
                #samples = self.dpm_solver(observed_data,cond_mask,side_info,self.n_samples,itp_info)
                samples = self.FastSTI_Sampling_order2(observed_data,cond_mask,side_info,self.n_samples,itp_info)
                end_event.record()
        
                
                if self.batch_count<5:
                    torch.cuda.synchronize()  
                    batch_latency = start_event.elapsed_time(end_event) / 1000.0 
                    self.total_latency += batch_latency
                    
                #################################################
                if self.batch_count == 0:  
                    try:
                        current_sample = torch.randn_like(observed_data)
                        diff_input = ((1-cond_mask)*current_sample).unsqueeze(1) #(B,1,K,L)
                       
                        t = torch.tensor([1], dtype=torch.long).to(self.device)
                        
                        flops_per_diffmodel, _ = profile(
                            self.generator.diffmodel,
                            inputs=(diff_input,t,side_info, itp_info,cond_mask),
                            verbose=False
                        )
    
                        CM_steps = 7 # len([49, 38, 24, 18, 17, 14, 5])
                        B, K, L = observed_data.shape
                        #tensor_ops_flops = (9 * B * K * L + 10) * CM_steps * self.n_samples
                        self.total_flops = (flops_per_diffmodel * CM_steps * self.n_samples) #+ tensor_ops_flops
                    except Exception as e:
                        print(f"Warning: FLOPS calculation failed: {e}")
                        self.total_flops = 0
            
                self.batch_count += 1    
                #####################################################

                
                for i in range(len(cut_length)):  # to avoid double evaluation
                    target_mask[i, ..., 0 : cut_length[i].item()] = 0

                samples = samples.permute(0,1,3,2).to(self.device) #(B,n_samples,L,K)
        
                c_target = observed_data.permute(0,2,1)
                eval_points = target_mask.permute(0,2,1)
                observed_points = observed_mask.permute(0,2,1)

    
                selected_samples = samples * eval_points.unsqueeze(1)  # (B, n_samples, L, K)

        

                samples_median = samples.median(dim=1).values
                final_sample = samples_median
            

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
          
        
    def on_test_epoch_end(self):
        
        
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        MRE = self.total_mre / self.evalpoints_total
        avg_latency = self.total_latency / (self.batch_count-1) 
        flops = self.total_flops 
        
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
        print(f"Test Set Average Latency: {avg_latency:.6f} seconds")
        print(f"Test Set Average FLOPS: {flops / 1e9:.6f} GFLOPS")  
      
        pass

    # no need for optimizer
    def configure_optimizers(self):

        optimizer_G = torch.optim.AdamW(
            self.generator.parameters(), 
            lr=0.0001, 
            betas=(0.9, 0.999)
        )
        return optimizer_G
    

     
    def  CMimpute(self, observed_data, cond_mask, side_info, n_samples,itp_info,consis_flag,phased_flag,observed_mask=None):
 
        CM_seq1 = self.list1
        CM_seq2 = CM_seq1[1:]
        CM_seq2.append(0)
        
   
        

        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)


        for i in range(n_samples):
         
            current_sample = torch.randn_like(observed_data) 
            current_sample_1 = torch.randn_like(observed_data)     
            if consis_flag:
                scale = [1 for i in range(1000)] 
            else:
                scale = [0 for i in range(1000)]
            for s in range(0,len(CM_seq1)):
                if s <= -1:
                    t_minus1 = CM_seq2[s]
                    t_minus1 = torch.tensor(t_minus1)
                    tminus1_low_idx = torch.floor(t_minus1).long()
                    tminus1_high_idx = torch.ceil(t_minus1).long()
                    alpha_tminus1_low = self.generator.alpha_torch[tminus1_low_idx]
                    alpha_tminus1_high = self.generator.alpha_torch[tminus1_high_idx]
                    alpha_tminus1 = alpha_tminus1_low + (alpha_tminus1_high-alpha_tminus1_low)*(t_minus1-tminus1_low_idx)
                    noise = torch.randn_like(observed_data)
                    current_sample = (alpha_tminus1**0.5)*itp_info.squeeze(1) + (1-alpha_tminus1)**0.5*noise
                else:
                    t = CM_seq1[s]
                    t_minus1 = CM_seq2[s]
                    t = torch.tensor(t) 
                    t_minus1 = torch.tensor(t_minus1)

                    t_low_idx = torch.floor(t).long()
                    t_high_idx = torch.ceil(t).long()
                    alpha_t_low = self.generator.alpha_torch[t_low_idx]
                    alpha_t_high = self.generator.alpha_torch[t_high_idx]
                    alpha_t = alpha_t_low + (alpha_t_high-alpha_t_low)*(t-t_low_idx)

                    tminus1_low_idx = torch.floor(t_minus1).long()
                    tminus1_high_idx = torch.ceil(t_minus1).long()
                    alpha_tminus1_low = self.generator.alpha_torch[tminus1_low_idx]
                    alpha_tminus1_high = self.generator.alpha_torch[tminus1_high_idx]
                    alpha_tminus1 = alpha_tminus1_low + (alpha_tminus1_high-alpha_tminus1_low)*(t_minus1-tminus1_low_idx)
                    
                
                
                    ### start here for xt-->x0-->xt-1
                    diff_input = ((1-cond_mask)*current_sample).unsqueeze(1) #(B,1,K,L)
                    predicted = self.generator.diffmodel(diff_input,torch.tensor(t).to(self.device), side_info, itp_info,cond_mask)
                    X0_t = (current_sample-predicted*((1-alpha_t)**0.5))/(alpha_t**0.5)

                  
                    eta_star = (1 - alpha_tminus1)**0.5/((1-(alpha_t/alpha_tminus1))*((1-alpha_tminus1)/(1-alpha_t)))**0.5
                    eta = eta_star*scale[s]
                    #eta = 1
                    c1 = (
                            eta * ((1-(alpha_t/alpha_tminus1))*((1-alpha_tminus1)/(1-alpha_t)))**0.5
                    )
                    c2 = max(0,((1 - alpha_tminus1) - c1**2)**0.5)
                    noise = torch.randn_like(current_sample)                     
                    x_tminus1 = (alpha_tminus1**0.5)*X0_t + c2*predicted + c1*noise
                
                    ### end here for x_t-1 has been acquired
                    
                    #really important
                    #SNR-NI strategy
                    if phased_flag:
                        if t_minus1 > 20:
                            current_sample = x_tminus1
                      
                        else :
                            current_sample = X0_t
                    else:
                        if t_minus1 > 0:
                            current_sample = x_tminus1
                        else :
                            current_sample = X0_t
        

                    
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    

    
    

    
   
    # show SNR
    def showcoeff(self):
        for t in range(0,100):
          
            coeff1 = self.generator.alpha_torch[t]**0.5
            coeff2 = (1-self.generator.alpha_torch[t])**0.5
            print(f"{t}:{coeff1},{coeff2}")
            SNR = coeff1**2/coeff2**2
            print(f"the signal-to-noise ratio of {t}:{SNR}")

   
    # DDPM sampling 
    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            

            current_sample = torch.randn_like(observed_data)

            for t in range(self.generator.num_steps - 1, -1, -1):
                
                diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                predicted = self.generator.diffmodel(diff_input,torch.tensor([t]).to(self.device), side_info,  itp_info, cond_mask)

                coeff1 = 1 / self.generator.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.generator.alpha_hat[t]) / (1 - self.generator.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.generator.alpha[t - 1]) / (1.0 - self.generator.alpha[t]) * self.generator.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    
   

   
    def heun_second_order(self, x_t, t, delta_t, cond_mask, side_info, itp_info):
        

        def f(x, t):
            diff_input = ((1 - cond_mask) * x).unsqueeze(1)  # (B,1,K,L)
            epsilon_t = self.generator.diffmodel(diff_input,torch.tensor([t]).to(self.device), side_info,  itp_info, cond_mask)
            return epsilon_t

        def nu(x, e_t, t, t_minus1):
            alpha_t = self.generator.alpha_torch[t]
            alpha_t_minus1 = self.generator.alpha_torch[t_minus1]
            res = (alpha_t_minus1**0.5 / alpha_t**0.5) * x - ((alpha_t_minus1 - alpha_t) / 
                    (alpha_t**0.5 * (((1 - alpha_t_minus1) * alpha_t)**0.5 + ((1 - alpha_t) * alpha_t_minus1)**0.5))) * e_t
            return res

        et_1 = f(x_t, t)
        xt_1 = nu(x_t, et_1, t, int(t + delta_t))
        et_2 = f(xt_1, int(t + delta_t))

        et_true = 0.5 * (et_1 + et_2) 
        x_prev = nu(x_t, et_true, t, int(t + delta_t))

        return x_prev, et_true

    def linear_multistep_second_order(self, x_t, t, delta_t, cond_mask, side_info, itp_info, gradients):
        delta_t = -1

        def f(x, t):
            diff_input = ((1 - cond_mask) * x).unsqueeze(1)  # (B,1,K,L)
            epsilon_t = self.generator.diffmodel(diff_input,torch.tensor([t]).to(self.device), side_info,  itp_info, cond_mask)
            return epsilon_t

        def nu(x, e_t, t, t_minus1):
            alpha_t = self.generator.alpha_torch[t]
            alpha_t_minus1 = self.generator.alpha_torch[t_minus1]
            res = (alpha_t_minus1**0.5 / alpha_t**0.5) * x - ((alpha_t_minus1 - alpha_t) /
                    (alpha_t**0.5 * (((1 - alpha_t_minus1) * alpha_t)**0.5 + ((1 - alpha_t) * alpha_t_minus1)**0.5))) * e_t
            return res

        if len(gradients) >= 1:
            e_t = f(x_t, t)
            e_t_true = (1.0 / 2.0) * (3 * e_t - gradients[-1])
            x_prev = nu(x_t, e_t_true, t, int(t + delta_t))
        else:
            print("Error! No enough gradients for PLMS2")

        gradients.append(e_t_true)
        if len(gradients) > 1:
            gradients.pop(0)
        return x_prev


    def FastSTI_Sampling_order2(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        T = 100
    
        B, K, L = observed_data.shape
        #delta_t_list = [-1,-16,-10,-9,-11,-2]
        #delta_t_list = [-7,-14,-6,-3,-17,-2]
        #delta_t_list = [-24,-25]
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            t = T - 1
            x_T = torch.randn_like(observed_data)
            x_t = x_T
            gradients = []  
            cnt = 0

            while t > 0 :
                #print(t)
                delta_t = -1
                #delta_t = delta_t_list[cnt]
                if cnt == 0:
                    
                    x_t, e_t = self.heun_second_order(x_t, t,delta_t, cond_mask, side_info, itp_info)
                    gradients.append(e_t)
                    t = t + delta_t
                else:
                    
                    x_t = self.linear_multistep_second_order(x_t, t,delta_t, cond_mask, side_info, itp_info, gradients)
                    t = t + delta_t
                cnt += 1

            imputed_samples[:, i] = x_t.detach()

        return imputed_samples
    
    
    def x0_predictor(self,x_t,side_info,t,itp_info,cond_mask):
        ########################## one-order 
        # diff_input = ((1-cond_mask)*x_t).unsqueeze(1) #(B,1,K,L)
        # predicted = self.generator.diffmodel(diff_input,torch.tensor(t).to(self.device), side_info, itp_info,cond_mask)  
        # X0_t = (x_t-predicted*((1-self.generator.alpha_torch[t])**0.5))/(self.generator.alpha_torch[t]**0.5)
        # return X0_t
        ##########################
        ########################## 2-order
        diff_input = ((1-cond_mask)*x_t).unsqueeze(1) #(B,1,K,L)
        et_1 = self.generator.diffmodel(diff_input,torch.tensor(t).to(self.device),side_info,itp_info,cond_mask)
        def nu(x, e_t, t, t_minus1):
            alpha_t = self.generator.alpha_torch[t]
            alpha_t_minus1 = self.generator.alpha_torch[t_minus1]
            res = (alpha_t_minus1**0.5 / alpha_t**0.5) * x - ((alpha_t_minus1 - alpha_t) /
                    (alpha_t**0.5 * (((1 - alpha_t_minus1) * alpha_t)**0.5 + ((1 - alpha_t) * alpha_t_minus1)**0.5))) * e_t
            return res
        x0_halft = nu(x_t,et_1,t,int(t/2))

        # x0_t_1 = (x_t-et_1*((1-self.generator.alpha_torch[t])**0.5))/(self.generator.alpha_torch[t]**0.5)
        # noise = torch.randn_like(x_t)
        # x0_halft = self.generator.alpha_torch[int(t/2)]**0.5*x0_t_1 + (1.0-self.generator.alpha_torch[int(t/2)])**0.5*et_1
        diff_input_halft = ((1-cond_mask)*x0_halft).unsqueeze(1) #(B,1,K,L)
        et_2 = self.generator.diffmodel(diff_input_halft,torch.tensor(int(t/2)).to(self.device),side_info,itp_info,cond_mask)
        x0_t_2 = (x0_halft-et_2*((1-self.generator.alpha_torch[int(t/2)])**0.5))/(self.generator.alpha_torch[int(t/2)]**0.5)
        return x0_t_2
        
    
    
    
    def x0_predictor_midpoint(self, x_t, side_info, t, itp_info, cond_mask):
        diff_input = ((1 - cond_mask) * x_t).unsqueeze(1)  # (B, 1, K, L)
        predicted_t = self.generator.diffmodel(
            diff_input,torch.tensor(t).to(self.device), side_info,  itp_info, cond_mask
        )  # f(t, x_t)
        coeff = self.generator.alpha_torch[t]/self.generator.alpha_torch[int(t/2)]
        x_midpoint = (x_t- (1-coeff)**0.5*predicted_t)/(coeff**0.5)

    
        diff_input_midpoint = ((1 - cond_mask) * x_midpoint).unsqueeze(1)  # (B, 1, K, L)
        predicted_midpoint = self.generator.diffmodel(
            diff_input_midpoint,torch.tensor(0).to(self.device), side_info,  itp_info, cond_mask
        )  # f(0, x0_predict)

        X0_t = (x_midpoint-(1-self.generator.alpha_torch[int(t/2)])**0.5*predicted_midpoint)/(self.generator.alpha_torch[int(t/2)]**0.5)
        return X0_t



    def dpm_solver(self, observed_data, cond_mask, side_info, n_samples, itp_info):
       ### for dpm solver 
        betas = torch.from_numpy(self.generator.beta)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete',betas=betas)
        #self.generator.diffmodel()
       
        model_kwargs = {
            'cond_mask': cond_mask,
            'side_info': side_info,
            'itp_x': itp_info
        }

        self.model_fn = model_wrapper(
            self.generator.diffmodel,
            self.noise_schedule,
            model_type="noise",
            #model_type="x_start",
            #model_type="v",
            #model_type="score",
            model_kwargs=model_kwargs,  
        )
        self.solver = DPM_Solver(self.model_fn, self.noise_schedule, algorithm_type="dpmsolver++")  
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):
            x_T = torch.randn_like(observed_data)
            x_sample = self.solver.sample(  
                x_T,  
                steps=7,  
                order=2,  
                #skip_type="logSNR",  
                skip_type="time_uniform",  
                #skip_type="time_quadratic", 
                method="multistep",  
                #method="singlestep_fixed",  
            )
            imputed_samples[:, i] = x_sample.detach()
        return imputed_samples
        ### end for 

    