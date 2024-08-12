import torch
import argparse
import numpy as np
from vqgan import VQGAN
from torch.utils.data import DataLoader
from dl import VQDataset, calculate_metrices, restore_img, Early_stopping

## define parameters
def setting_para():
    parser= argparse.ArgumentParser(description= 'Parser for Arguments')
    parser.add_argument('-seed', type= int, default= 1)
    parser.add_argument('-embedding_dim', type= int, default= 8)
    parser.add_argument('-train_NODDI', type= str, default= '../../../patch_brain_size64/30noddi_train_hr_30sub.npy')
    parser.add_argument('-val_NODDI', type= str, default= '../../../patch_brain_size64/30noddi_val_hr_3sub.npy')
    parser.add_argument('-lr', type= float, default= 1e-4)
    parser.add_argument('-weight_decay', type= float, default= 5e-5)
    parser.add_argument('-epochs', type= int, default= 10000000000)
    parser.add_argument('-device', type= str, default= 'cuda:1')
    parser.add_argument('-device_ids', type= list, default= [1])
    parser.add_argument('-discriminator_iter_start', type= int, default= 50000)
    parser.add_argument('-use_sigmoid', type= bool, default= True)
    parser.add_argument('-n_codes', type= int, default= 8192)
    parser.add_argument('-n_hiddens', type= int, default= 64)
    parser.add_argument('-disc_layers', type= int, default= 3)
    parser.add_argument('-num_groups', type= int, default= 32)
    parser.add_argument('-image_gan_weight', type= int, default= 1)
    parser.add_argument('-video_gan_weight', type= int, default= 1)
    parser.add_argument('-l1_weight', type= int, default= 4.0)
    parser.add_argument('-norm_type', type= str, default= 'group')
    parser.add_argument('-padding_type', type= str, default= 'replicate')
    parser.add_argument('-disc_loss_type', type= str, default= 'hinge')
    parser.add_argument('-image_channels', type= int, default= 1)
    parser.add_argument('-downsample', type= list, default= [2, 2, 2])
    parser.add_argument('-patience', type= int, default= 100000)
    parser.add_argument('-no_random_restart', type= bool, default= False)
    parser.add_argument('-restart_thres', type= int, default= 1)
    parser.add_argument('-gan_feat_weight', type= float, default= 0)
    parser.add_argument('-perceptual_weight', type= float, default= 0)
    parser.add_argument('-patch_size', type= int, default= 64)
    parser.add_argument('-stride', type= int, default= 36)
    parser.add_argument('-batch_size', type= int, default= 2)
    parser.add_argument('-size', type= tuple, default= (3, 136, 172, 136))
    parser.add_argument('-pt_path_vqgan', type= str, default= 'vqgan.pt')
    args= parser.parse_args([])
    return args

def run(args):
    args= setting_para()
    vqgan= VQGAN(args).to(args.device)
    opt_ae, opt_disc= vqgan.configure_optimizers()[0]
    tr_ds, va_ds= VQDataset(args.train_NODDI), VQDataset(args.val_NODDI)
    train_loader, val_loader= DataLoader(tr_ds, batch_size= args.batch_size, shuffle= True), DataLoader(va_ds, batch_size= args.batch_size, shuffle= False)
    early_stopping= Early_stopping(args.patience, args.pt_path_vqgan)
    for ep in range(args.epochs):
        for step, x in enumerate(train_loader):
            x= x.to(torch.float32).to(args.device)
            g_loss= vqgan.training_step(x, ep* len(train_loader)+ step+ 1, 0)
            d_loss= vqgan.training_step(x, ep* len(train_loader)+ step+ 1, 1)
            opt_disc.zero_grad()
            d_loss.backward(retain_graph= True)
            opt_ae.zero_grad()
            g_loss.backward()
            opt_disc.step()
            opt_ae.step()
            if step% 128== 0:print(f'epoch: {ep+ 1}, step: {step+ 1}, disc loss: {d_loss.item()}, vqgan loss: {g_loss.item()}')
        vqgan.eval();img_hats= [];imgs= []
        with torch.no_grad():
            for step, x in enumerate(val_loader):
                x= x.to(torch.float32).to(args.device)
                recon_x= vqgan.validation_step(x, step)
                img_hats.append(recon_x.cpu());imgs.append(x.cpu())
        img_hats, imgs= torch.cat(img_hats), torch.cat(imgs)
        img_hats, imgs= img_hats.chunk(3, dim= 0), imgs.chunk(3, dim= 0)
        mean_psnr, mean_ssim= [], []
        for i in range(len(img_hats)):
            img_hat, img= img_hats[i], imgs[i]
            recon_x, x= restore_img(img.cpu().numpy().reshape(-1, 3, 64, 64, 64), img_hat.cpu().numpy().reshape(-1, 3, 64, 64, 64), args.patch_size, args.stride, args.size)
            p_val, s_val= calculate_metrices(x, recon_x)
            mean_psnr.append(p_val);mean_ssim.append(s_val)
        mean_ssim, mean_psnr= torch.tensor(mean_ssim).mean(), torch.tensor(mean_psnr).mean()
        # early_stopping
        early_stopping(-(mean_ssim+ mean_psnr), vqgan)
        print(f'epoch: {ep+ 1}, early_stopping: {early_stopping.clock+ 1}/{early_stopping.patience}, p_val: {p_val}, s_val: {s_val}')
        if early_stopping.flag:break

if __name__ == '__main__':
    args= setting_para()
    run(args)
