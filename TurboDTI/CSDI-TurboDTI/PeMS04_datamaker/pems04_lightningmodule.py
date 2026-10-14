import pytorch_lightning as pl
from main_model_aqi import CSDI_PEMS04
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import time
# from thop import profile  # Uncomment if model profiling is required (ensure thop is installed)
from dpm_solver_pytorch import NoiseScheduleVP,model_wrapper,DPM_Solver
    


class PeMS04LightningModule(pl.LightningModule):
    def __init__(self,config,device,target_dim=307,seq_len=36):
        super().__init__()
        path = "./CSDI-TurboDTI/data/PEMS04/pems04_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.MAE = 0
        self.MSE = 0
        self.list1 = [49,42,35,27,13]
        self.randomness_list = [1,0,1,0]
        self.generator = CSDI_PEMS04(config,device,target_dim=307)
        self.generator.load_state_dict(torch.load("./CSDI-TurboDTI/save/pems04_coeffs_new2.0lr0_20250708_081319/model.pth",map_location="cuda:0"))

        self.automatic_optimization = False
        self.step_counter = 0 
        self.skipping_steps = 10
        self.n_samples = 1


    def on_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        ) = self.generator.process_data(batch)
        path = f"./CSDI-TurboDTI/sample_PeMS04_DDPM/sample_{self.step_counter}.pt"
        #path = f"./CSDI-TurboDTI/sample_PeMS04_CN/sample_{self.step_counter}.pt"
        side_info = self.generator.get_side_info(observed_tp, cond_mask)
        itp_info = coeffs.unsqueeze(1)
        eval_mask = observed_mask - cond_mask
        num_eval = eval_mask.sum()
        x_T = torch.randn_like(observed_data)
        diff_input = ((1-cond_mask)*x_T).unsqueeze(1)
        if 0 <= self.step_counter < 742*5:
            with torch.no_grad():
                samples = self.impute(observed_data, cond_mask, side_info, 100, coeffs)
                # samples = self.CMimpute(observed_data, cond_mask, side_info, 10, coeffs, eta=1)
                samples_median = samples.median(dim=1).values
        data = {}

        # Convert data to torch.Tensor with proper dtype
        data['observed_data'] = torch.tensor(observed_data, dtype=torch.float32)
        data['cond_mask'] = torch.tensor(cond_mask, dtype=torch.float32)  # Consider bool type if mask
        data['observed_tp'] = torch.tensor(observed_tp, dtype=torch.float32)
        data['observed_mask'] = torch.tensor(observed_mask, dtype=torch.float32)
        data['coeffs'] = torch.tensor(coeffs, dtype=torch.float32)
        data['teacher_sample'] = torch.tensor(samples_median, dtype=torch.float32)

        print("Saving sample", self.step_counter)
        torch.save(data, path)


        self.step_counter += 1
    
    # no need for validation(training free)       
    def validation_step(self, batch, batch_idx):
        pass 

    def on_test_start(self):
        self.total_mae = 0
        self.total_mse = 0
        self.total_mre = 0
        self.evalpoints_total = 0
        self.total_var = 0
        self.total_latency = 0  # record total latency
        self.total_flops = 0   # record total FLOPs
        self.batch_count = 0   # record total number of batches

    def test_step(self, batch, batch_idx):
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
        
        ###########################
        # For CUDA timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # ensure GPU state synchronization
        
        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask
        side_info = self.generator.get_side_info(observed_tp, cond_mask)

        start_event.record()
        #####################
        # diff_input = ((1 - cond_mask) * torch.randn_like(observed_data)).unsqueeze(1)
        # cond_obs = coeffs.unsqueeze(1)
        # diff_input = torch.cat([cond_obs, diff_input], dim=1)  # (B, 2, K, L)
        # t = torch.tensor([1], dtype=torch.long).to(self.device)
        # predicted = self.generator.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
        # samples = predicted.unsqueeze(1)
        #####################
        #samples = self.impute(observed_data, cond_mask, side_info, self.n_samples, coeffs)
        samples = self.CMimpute(observed_data, cond_mask, side_info, self.n_samples, coeffs,eta=1)
        #samples = self.dpm_solver(observed_data,cond_mask,side_info,self.n_samples,coeffs)
        #samples = self.FastSTI_Sampling_order2(observed_data,cond_mask,side_info,self.n_samples,coeffs)
        end_event.record()

        if self.batch_count < 8:
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            batch_latency = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            self.total_latency += batch_latency

        # Calculate FLOPS (for CMimpute)
        if self.batch_count == 0:  # Calculate FLOPS only for the first batch
            try:
                # Estimate FLOPS of diffmodel
                diff_input = ((1 - cond_mask) * torch.randn_like(observed_data)).unsqueeze(1)
                cond_obs = coeffs.unsqueeze(1)
                diff_input = torch.cat([cond_obs, diff_input], dim=1)  # (B, 2, K, L)
                t = torch.tensor([1], dtype=torch.long).to(self.device)
                flops_per_diffmodel, _ = profile(
                    self.generator.diffmodel,
                    inputs=(diff_input, side_info, t),
                    verbose=False
                )
                # Number of CMimpute iterations
                CM_steps = 50  # len([49, 38, 24, 18, 17, 14, 5])
                # Tensor operation FLOPS (commented out)
                # B, K, L = observed_data.shape
                # tensor_ops_flops = (9 * B * K * L + 10) * CM_steps * self.n_samples
                # Total FLOPS
                self.total_flops = (flops_per_diffmodel * CM_steps * self.n_samples)  # + tensor_ops_flops
            except Exception as e:
                print(f"Warning: FLOPS calculation failed: {e}")
                self.total_flops = 0

        self.batch_count += 1


        for i in range(len(cut_length)):  # avoid double evaluation
            target_mask[i, ..., 0 : cut_length[i].item()] = 0

        samples = samples.permute(0, 1, 3, 2).to(self.device)
        c_target = observed_data.permute(0, 2, 1)
        eval_points = target_mask.permute(0, 2, 1)
        observed_points = observed_mask.permute(0, 2, 1)

        # Use eval_points mask to select corresponding sample values
        selected_samples = samples * eval_points.unsqueeze(1)  # (B, n_samples, L, K)

        # Calculate variance along n_samples dimension, i.e., variance at each (B, L, K) location
        variance = selected_samples.var(dim=1, unbiased=False)  # (B, L, K)

        samples_median = samples.median(dim=1).values
        final_sample = samples_median


        mae_current = (
            torch.abs((final_sample - c_target) * eval_points)
        ) * scaler
        mse_current = (
            ((final_sample - c_target) * eval_points) ** 2
        ) * (scaler ** 2)
        mre_current = (
            torch.abs(((final_sample * scaler) - (c_target * scaler))) / ((torch.abs(c_target) * scaler) + mean_scaler + 1e-8)  # avoid division by zero
        ) * eval_points

        self.total_mae += mae_current.sum().item()
        self.total_mse += mse_current.sum().item()
        self.total_mre += mre_current.sum().item()

        self.total_var += variance.sum().item()
        self.evalpoints_total += eval_points.sum().item()

    def on_test_epoch_end(self):
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        MRE = self.total_mre / self.evalpoints_total
        avg_latency = self.total_latency / (self.batch_count - 1)  # Average latency (seconds)
        flops = self.total_flops  # Average FLOPS

        # Log and print results
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)
        self.log('test_MRE', MRE)
        # self.log('test_avg_small_MRE', avg_small_mre)  # Log average of smaller MRE

        self.MAE = MAE
        self.MSE = MSE

        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")
        print(f"Test Set Average MRE: {MRE}")
        print(f"Test Set Average Latency: {avg_latency:.6f} seconds")
        print(f"Test Set Average FLOPS: {flops / 1e9:.6f} GFLOPS")  # Convert to GFLOPS
        # print(f"Test Set Average of Small MRE: {avg_small_mre}")  # Print average of smaller MRE



    def configure_optimizers(self):
    
        optimizer_G = torch.optim.AdamW(
            self.generator.parameters(), 
            lr=0.0001, 
            betas=(0.9, 0.999)
        )
        return optimizer_G
   
    def impute(self, observed_data, cond_mask, side_info, n_samples,coeffs):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
     

            current_sample = torch.randn_like(observed_data)

            for t in range(self.generator.num_steps - 1, -1, -1):
             

                cond_obs = (coeffs).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.generator.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

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
    
    def CMimpute(self, observed_data, cond_mask, side_info, n_samples, coeffs, eta=1):
        # # Dense first, then sparse
        # CM_seq1 = [49, 42, 41, 10]
        # CM_seq1 = [49, 38, 24, 18, 17, 14, 5]
        CM_seq1 = self.list1
        CM_seq2 = CM_seq1[1:]
        CM_seq2.append(0)
        
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):     
            current_sample = torch.randn_like(observed_data)      
            scale = [1] * 1000
            # scale = [0] * 1000
            for s in range(len(CM_seq1)):
                t = CM_seq1[s]
                t_minus1 = CM_seq2[s]
                    
                if s == -1:
                    predicted = current_sample
                    X0_t = (current_sample - predicted * ((1 - self.generator.alpha_torch[t])**0.5)) / (self.generator.alpha_torch[t]**0.5)
                else:
                    cond_obs = coeffs.unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
                    predicted = self.generator.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                    X0_t = (current_sample - predicted * ((1 - self.generator.alpha_torch[t])**0.5)) / (self.generator.alpha_torch[t]**0.5)
   
                eta_star = (1 - self.generator.alpha_torch[t_minus1])**0.5 / ((1 - (self.generator.alpha_torch[t] / self.generator.alpha_torch[t_minus1])) * ((1 - self.generator.alpha_torch[t_minus1]) / (1 - self.generator.alpha_torch[t])))**0.5
                eta = eta_star * scale[s]
                c1 = eta * ((1 - (self.generator.alpha_torch[t] / self.generator.alpha_torch[t_minus1])) * ((1 - self.generator.alpha_torch[t_minus1]) / (1 - self.generator.alpha_torch[t])))**0.5
                c2 = max(0, ((1 - self.generator.alpha_torch[t_minus1]) - c1**2)**0.5)
                noise = torch.randn_like(current_sample)
                x_tminus1 = (self.generator.alpha_torch[t_minus1]**0.5) * X0_t + c2 * predicted + c1 * noise
                c2 = (1 - self.generator.alpha_torch[t_minus1])**0.5
        
                # Important
                # current_sample = x_tminus1
                # SNR-NI strategy
                if t_minus1 >= 10:
                    current_sample = x_tminus1
                else:
                    current_sample = X0_t
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    # show SNR
    def showcoeff(self):
        for t in range(0,50):
        
            coeff1 = self.generator.alpha_torch[t]
            coeff2 = (1-self.generator.alpha_torch[t])**0.5
         
            print(f"{t}:{coeff1},{coeff2}")
            SNR = coeff1**2/coeff2**2
            print(f"the signal-to-noise ratio of {t}:{SNR}")

    def heun_second_order(self, x_t, t, delta_t, cond_mask, side_info, coeffs):
        def f(x, t):
            diff_input = ((1 - cond_mask) * x).unsqueeze(1)
            cond_obs = coeffs.unsqueeze(1)
            diff_input = torch.cat([cond_obs, diff_input], dim=1)  # (B, 2, K, L)
            epsilon_t = self.generator.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
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


    def linear_multistep_second_order(self, x_t, t, delta_t, cond_mask, side_info, coeffs, gradients):
        delta_t = -1

        def f(x, t):
            diff_input = ((1 - cond_mask) * x).unsqueeze(1)
            cond_obs = coeffs.unsqueeze(1)
            diff_input = torch.cat([cond_obs, diff_input], dim=1)  # (B, 2, K, L)
            epsilon_t = self.generator.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
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

    #FastSTI-2NA T=50
    def FastSTI_Sampling_order2(self, observed_data, cond_mask, side_info, n_samples, coeffs):
        T = 50

    
        B, K, L = observed_data.shape
        # tc = [78.0943,54.9386,41.7680,29.7190,2.8100,0]
        # ts = [99,78.0943,54.9386,41.7680,29.7190,2.8100]
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
                    
                    x_t, e_t = self.heun_second_order(x_t, t,delta_t, cond_mask, side_info, coeffs)
                    gradients.append(e_t)
                    t = t + delta_t
                else:
                    
                    x_t = self.linear_multistep_second_order(x_t, t,delta_t, cond_mask, side_info, coeffs, gradients)
                    t = t + delta_t
                cnt += 1

            imputed_samples[:, i] = x_t.detach()

        return imputed_samples
    
    def dpm_solver(self, observed_data, cond_mask, side_info, n_samples, coeffs):
       ### for dpm solver 
        betas = torch.from_numpy(self.generator.beta)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete',betas=betas)
        #self.generator.diffmodel()
       
        model_kwargs = {
            'coeffs': coeffs,
            'side_info': side_info,
            'cond_mask' : cond_mask
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

    
   