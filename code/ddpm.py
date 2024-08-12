import time
import copy
import torch
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from vqgan import VQGAN
from pathlib import Path
from functools import partial
from inspect import isfunction
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.distributions import Categorical
from dl import restore_img, calculate_metrices
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def make_beta_schedule(schedule, n_timestep, linear_start= 1e-4, linear_end= 2e-2, cosine_s= 8e-3):
    if schedule== "linear":
        betas= (
                torch.linspace(linear_start** 0.5, linear_end** 0.5, n_timestep, dtype= torch.float64)** 2
        )

    elif schedule== "cosine":
        timesteps= (
                torch.arange(n_timestep+ 1, dtype=torch.float64)/ n_timestep+ cosine_s
        )
        alphas= timesteps/ (1 + cosine_s)* np.pi/ 2
        alphas= torch.cos(alphas).pow(2)
        alphas= alphas/ alphas[0]
        betas= 1- alphas[1:]/ alphas[:-1]
        betas= np.clip(betas, a_min= 0, a_max= 0.999)

    elif schedule== "sqrt_linear":
        betas= torch.linspace(linear_start, linear_end, n_timestep, dtype= torch.float64)
    elif schedule== "sqrt":
        betas= torch.linspace(linear_start, linear_end, n_timestep, dtype= torch.float64)** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        lr_encoder,
        vqgan,
        image_size,
        depth_size,
        channels= 8,
        timesteps= 1000,
        loss_type= 'l1',
        betas= None,
        with_condition= True,
        with_pairwised= False,
        apply_bce= False,
        device= 'cuda:1',
        ema_decay= 0.95,
        lambda_bce= 0.0,
        sample_steps= 100        
    ):
        super().__init__()
        self.channels= channels
        self.image_size= image_size
        self.depth_size= depth_size
        self.denoise_fn= denoise_fn
        self.with_condition= with_condition
        self.with_pairwised= with_pairwised
        self.apply_bce= apply_bce
        self.lambda_bce= lambda_bce
        self.vqgan= vqgan
        self.lr_encoder= lr_encoder
        self.l_bound, self.h_bound, self.mean_, self.std_= self.vqgan.codebook.embeddings.min(), self.vqgan.codebook.embeddings.max(), self.vqgan.codebook.embeddings.mean(), self.vqgan.codebook.embeddings.std()
        self.vqgan.eval()
        self.lr_encoder.eval()
        self.loss_hist= torch.ones(timesteps, device= device)   
        self.ema_decay= ema_decay
        self.sample_timesteps= torch.linspace(timesteps- 1, 0, sample_steps).long().to(device)
        betas= make_beta_schedule(schedule= 'linear', n_timestep= timesteps)
        alphas= 1. - betas
        alphas_cumprod= np.cumprod(alphas, axis=0)
        alphas_cumprod_prev= np.append(1., alphas_cumprod[:-1])
        # timesteps, = betas.shape
        self.num_timesteps= timesteps
        self.loss_type= loss_type
        to_torch= partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance= betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        # self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))))        
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        x_hat= 0
        mean= extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance= extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance= extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat= 0.
        posterior_mean= (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance= extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped= extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def norm_min_max(self, x):
        return ((x - self.l_bound) / (self.h_bound - self.l_bound)) * 2.0 - 1.0
    
    def de_norm_min_max(self, x):
        return (x+ 1.0)/ 2* (self.h_bound- self.l_bound)+ self.l_bound
    
    def norm_z(self, x):
        return (x- self.mean_)/ self.std_
    
    def de_norm_z(self, x):
        return x* self.std_+ self.mean_

    def norm_img(self, x):
        return (x- self.img_mean)/ self.img_std

    def de_norm_img(self, x):
        return x* self.img_std+ self.img_mean
    
    def p_mean_variance(self, x, t, clip_denoised: bool, c = None):
        if self.with_condition:
            x_recon= self.predict_start_from_noise(x, t= t, noise=self.denoise_fn(torch.cat([x, c], dim= 1), t, c))
        else:
            x_recon= self.predict_start_from_noise(x, t= t, noise=self.denoise_fn(x, t))
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance= self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors= None, clip_denoised= False, repeat_noise= False):
        b, *_, device= *x.shape, x.device
        model_mean, _, model_log_variance= self.p_mean_variance(x= x, t= t, c= condition_tensors, clip_denoise= clip_denoised)
        noise= noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask= (1- (t== 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        noise_coeff= torch.sqrt(extract(self.betas, t, x.shape))
        # return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return model_mean+ nonzero_mask* noise_coeff* noise

    @torch.no_grad()
    def p_sample_mine(self, x, t, condition_tensors= None, clip_denoised= False):
        b, *_, device= *x.shape, x.device
        noise= noise_like(x.shape, device)
        # no noise when t == 0
        nonzero_mask= (1- (t== 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        noise_coeff= torch.sqrt(extract(self.betas, t, x.shape))
        # condition
        if self.with_condition:
            pred_noise= self.denoise_fn(x, t, lowres_cond_img= condition_tensors)
        else:
            pred_noise= self.denoise_fn(x, t)
        alphas= extract(self.alphas, t, x.shape)
        alphas_cumprod= extract(self.alphas_cumprod, t, x.shape)
        x= 1/ torch.sqrt(alphas)* (x- ((1 - alphas)/ (torch.sqrt(1- alphas_cumprod)))* pred_noise)+ nonzero_mask* noise_coeff* noise
        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors = None):
        device= self.betas.device
        b= shape[0]
        img= torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc= 'sampling loop time step', total= self.num_timesteps):
            if self.with_condition:
                t= torch.full((b,), i, device= device, dtype= torch.long)
                img= self.p_sample(img, t, condition_tensors= condition_tensors)
            else:
                img= self.p_sample(img, torch.full((b,), i, device= device, dtype= torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size= 2, condition_tensors = None):
        image_size= self.image_size
        depth_size= self.depth_size
        channels= self.channels
        return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors= condition_tensors)

    @torch.no_grad()
    def interpolate(self, x1, x2, t= None, lam= 0.5):
        b, *_, device= *x1.shape, x1.device
        t= default(t, self.num_timesteps- 1)
        assert x1.shape== x2.shape
        t_batched= torch.stack([torch.tensor(t, device= device)] * b)
        xt1, xt2= map(lambda x: self.q_sample(x, t= t_batched), (x1, x2))
        img= (1- lam)* xt1 + lam* xt2
        for i in tqdm(reversed(range(0, t)), desc= 'interpolation sample time step', total= t):
            img= self.p_sample(img, torch.full((b,), i, device= device, dtype= torch.long))
        return img

    @torch.no_grad()
    def ddim_sample(self, xt, condition_tensors= None, eta= 0):
        for t, tau in list(zip(self.sample_timesteps[:-1], self.sample_timesteps[1:]))[::-1]:
            t_= torch.ones((xt.shape[0], ), device= xt.device, dtype= torch.long)* t
            pred_noise= self.denoise_fn(xt, t_, lowres_cond_img= condition_tensors)
            sigma= (eta* torch.sqrt(extract(self.betas, t_, xt.shape)))
            sqrt_alphas_cumprod_tau= self.alphas_cumprod[tau]** 0.5
            sqrt_alphas_cumprod_t= self.alphas_cumprod[t]** 0.5
            sqrt_one_minus_alphas_cumprod_t= (1.0- self.alphas_cumprod[t])** 0.5
            fist_term= sqrt_alphas_cumprod_tau* (xt- sqrt_one_minus_alphas_cumprod_t* pred_noise)/ sqrt_alphas_cumprod_t
            sec_term= ((1.0- self.alphas_cumprod[tau]- sigma** 2)** 0.5)* pred_noise
            eps= torch.randn_like(xt)
            xt= fist_term+ sec_term+ sigma* eps
        return self.de_norm_z(xt)

    # get xt according to x0 and t
    def q_sample(self, x_start, t, noise= None, c= None):
        noise= default(noise, lambda: torch.randn_like(x_start))
        x_hat= 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape)* x_start+
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)* noise+ x_hat
        )

    # predict noise
    def p_losses(self, x_hr, t, condition_tensors= None, noise= None):
        noise= default(noise, lambda: torch.randn_like(x_hr))
        x_noisy= self.q_sample(x_start= x_hr, t=t, noise= noise)
        x_recon= self.denoise_fn(x= x_noisy, timesteps= t, lowres_cond_img= condition_tensors)
        if self.loss_type== 'l1':
            loss= F.l1_loss(x_recon, noise.to(x_recon.device))
        elif self.loss_type== 'l2':
            loss= F.mse_loss(x_recon, noise.to(x_recon.device))
        else:
            raise NotImplementedError()

        return loss
    
    def sample_t(self, b):
        prob= self.loss_hist/ self.loss_hist.sum()
        m= Categorical(prob)
        t= m.sample((b,))
        return t
    
    def update_loss_history(self, t, loss):
        self.loss_hist[t]= self.ema_decay* self.loss_hist[t] + (1- self.ema_decay)* loss
        
    def forward(self, x_hr, condition_tensors=None, *args, **kwargs):
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                x_hr= self.vqgan.encode(x_hr, quantize= False, include_embeddings= False)
                x_hr= self.norm_z(x_hr)
                if condition_tensors!= None:
                    condition_tensors= self.norm_z(self.lr_encoder(condition_tensors))
        b, c, d, h, w, device, img_size, depth_size= *x_hr.shape, x_hr.device, self.image_size, self.depth_size
        assert h== img_size and w== img_size and d== depth_size, f'Expected dimensions: height= {img_size}, width= {img_size}, depth= {depth_size}. Actual: height= {h}, width= {w}, depth= {d}.'
        t = torch.randint(0, self.num_timesteps, (b,), device= device).long()
        loss= self.p_losses(x_hr, t, condition_tensors= condition_tensors, *args, **kwargs)
        return loss

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta= beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight= ma_params.data, current_params.data
            ma_params.data= self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old* self.beta + (1- self.beta)* new

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_dataset= None,
        val_dataset= None,
        val_sub_num= 3,
        ema_decay= 0.995,
        train_batch_size= 2,
        valid_batch_size= 2,
        train_lr= 1e-4,
        weight_decay= 5e-4,
        train_num_steps= 100000,
        gradient_accumulate_every= 2,
        device= 'cuda:1',
        step_start_ema= 2000,
        update_ema_every= 10,
        save_and_sample_every= 2000,
        patch= 64,
        stride= 36,
        patience= 1000000,
        image_size= (3, 136, 172, 136),
        pt_folder= '../pt',
    ):
        super().__init__()
        self.model= diffusion_model
        self.ema= EMA(ema_decay)
        self.device= device
        self.ema_model= copy.deepcopy(self.model)
        self.update_ema_every= update_ema_every
        self.step_start_ema= step_start_ema
        self.save_and_sample_every= save_and_sample_every
        self.batch_size= train_batch_size
        self.image_size= diffusion_model.image_size
        self.gradient_accumulate_every= gradient_accumulate_every
        self.train_num_steps= train_num_steps
        tdl= DataLoader(train_dataset, batch_size= train_batch_size, shuffle= True, pin_memory= False)
        self.vdl= DataLoader(val_dataset, batch_size= valid_batch_size, shuffle= False, pin_memory= False)
        self.val_sub_num= val_sub_num
        self.len_dataloader= len(tdl)
        self.tdl= cycle(tdl)
        self.opt= Adam(diffusion_model.parameters(), lr= train_lr, weight_decay= weight_decay, betas= (0.9, 0.99))
        self.step= 0
        self.patch= patch
        self.stride= stride
        self.image_size= image_size
        self.pt_folder= Path(pt_folder)
        self.pt_folder.mkdir(exist_ok= True, parents= True)
        self.clock= 0
        self.hist_val= 1e8
        self.patience= patience
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def early_stopping(self, metric, milestone):
        if metric< self.hist_val:
            self.hist_val= metric
            self.clock= 0
            self.save(milestone)
        else:
            self.clock+= 1
            
    def step_ema(self):
        if self.step< self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data= {
            'step': self.step,
            'model': self.model.state_dict(),
            'loss_hist': self.model.loss_hist,
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict()
        }
        torch.save(data, str(self.pt_folder/ f'model-{milestone}.pt'))

    def load(self, milestone, map_location= None, **kwargs):
        if map_location:
            data= torch.load(milestone, map_location= map_location)
        else:
            data= torch.load(milestone)
        self.step= data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.model.loss_hist= data['loss_hist']
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.opt.load_state_dict(data['opt'])

    def train(self):
        while self.step< self.train_num_steps:
            loss_item= []
            self.opt.zero_grad()
            for i in range(self.gradient_accumulate_every):
                data= next(self.tdl)
                hr_data, lr_data= data['hr'].cuda(), data['lr'].cuda()
                loss= self.model(hr_data, condition_tensors= lr_data)
                loss.backward()
                loss_item.append(loss.item())
            print(f'{self.step}: {torch.tensor(loss_item).mean()}')
            self.opt.step()
            # update ema
            if self.step% self.update_ema_every== 0:
                self.step_ema()
            # valid
            if (self.step+ 1)% self.save_and_sample_every== 0:
                self.ema_model.eval()
                milestone= self.step// self.save_and_sample_every
                with torch.no_grad():
                    start_time= time.time()
                    recon_x, ori_x= [], []
                    for _, data in tqdm(enumerate(self.vdl)):
                        hr_data, lr_data= data['hr'].to('cuda'), data['lr'].to('cuda')
                        condition_tensors= self.ema_model.norm_z(self.ema_model.lr_encoder(lr_data))
                        xt= torch.randn((condition_tensors.shape[0], condition_tensors.shape[1], 32, 32, 32)).to('cuda')
                        x0= self.ema_model.ddim_sample(xt, condition_tensors= condition_tensors)
                        recon_x.append(self.ema_model.vqgan.decode(x0.to('cuda'), quantize= True).to('cpu'))
                        ori_x.append(hr_data.to('cpu'))
                    recon_x, ori_x= torch.cat(recon_x, dim= 0), torch.cat(ori_x, dim= 0)
                    recon_x, ori_x= recon_x.chunk(self.val_sub_num, dim= 0), ori_x.chunk(self.val_sub_num, dim= 0)
                    mean_psnr, mean_ssim= [], []
                    for i in range(len(recon_x)):
                        img_hat, img= recon_x[i], ori_x[i]
                        recon_x, x= restore_img(img.cpu().numpy().reshape(-1, 3, 64, 64, 64), img_hat.cpu().numpy().reshape(-1, 3, 64, 64, 64), self.patch, self.stride, self.image_size)
                        p_val, s_val= calculate_metrices(x, recon_x)
                        mean_psnr.append(p_val);mean_ssim.append(s_val)
                    mean_ssim, mean_psnr= torch.tensor(mean_ssim).mean(), torch.tensor(mean_psnr).mean()
                    end_time= time.time()
                print(f'step: {self.step+ 1}, cost time: {(end_time - start_time)/60}, p_val: {mean_psnr}, s_val: {mean_ssim}')
                np.save(f'../results/recon_x{self.step+ 1}.npy', recon_x)
                np.save(f'../results/ori_x{self.step+ 1}.npy', ori_x)
                self.early_stopping(-(mean_psnr+ mean_ssim), milestone)
            self.step+= 1
        print('training completed')