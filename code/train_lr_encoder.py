import torch
import argparse
import numpy as np
import torch.nn.functional as F
from vqgan import VQGAN, shift_dim
from torch.optim import Adam, AdamW
from ddim.code.lrmri_encoder2 import LREncoder
from torch.utils.data import DataLoader
from dl import VQDataset, calculate_metrices, restore_img, Early_stopping

## define parameters
def setting_para():
    parser= argparse.ArgumentParser(description= 'Parser for Arguments')
    parser.add_argument('-seed', type= int, default= 1)
    parser.add_argument('-embedding_dim', type= int, default= 8)
    parser.add_argument('-train_hr_NODDI', type= str, default= '../../../patch_brain_size64/30noddi_train_hr_30sub.npy')
    parser.add_argument('-train_lr_MRI', type= str, default= '../../../patch_brain_size64/30noddi_train_lr_30sub.npy')
    parser.add_argument('-val_hr_NODDI', type= str, default= '../../../patch_brain_size64/30noddi_val_hr_3sub.npy')
    parser.add_argument('-val_lr_MRI', type= str, default= '../../../patch_brain_size64/30noddi_val_lr_3sub.npy')
    parser.add_argument('-lr', type= float, default= 1e-4)
    parser.add_argument('-weight_decay', type= float, default= 5e-5)
    parser.add_argument('-epochs', type= int, default= 10000000000)
    parser.add_argument('-device', type= str, default= 'cuda:0')
    parser.add_argument('-device_ids', type= list, default= [0])
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
    parser.add_argument('-image_channels', type= int, default= 3)
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
    parser.add_argument('-pt_path_vqgan', type= str, default= '../pt/vqgan.pt')
    parser.add_argument('-pt_path_mri_encoder', type= str, default= '../pt/lr_encoder.pt')
    args= parser.parse_args([])
    return args

def run(args):
    args= setting_para()
    vqgan= torch.load(args.pt_path_vqgan, map_location= args.device).to(args.device)
    vqgan.args.device= args.device
    vqgan.eval()
    mri_encoder= LREncoder(1, 2, 128, 8, (2, 2, 2)).to(args.device)
    opt= AdamW(mri_encoder.parameters(), lr= args.lr, weight_decay= args.weight_decay)
    tr_ds, va_ds= VQDataset(args.train_hr_NODDI, args.train_lr_MRI), VQDataset(args.val_hr_NODDI, args.val_lr_MRI)
    train_loader, val_loader= DataLoader(tr_ds, batch_size= args.batch_size, shuffle= True), DataLoader(va_ds, batch_size= 2, shuffle= False)
    early_stopping= Early_stopping(args.patience, args.pt_path_mri_encoder)
    for ep in range(args.epochs):
        for step, data in enumerate(train_loader):
            hr_data, lr_data= data['hr'].to(args.device), data['lr'].to(args.device)
            comparsion_tensor= None
            with torch.no_grad():
                comparsion_tensor= vqgan.encode(hr_data, quantize= False, include_embeddings= False)
            tensor_quantized= mri_encoder(lr_data)
            loss= F.l1_loss(tensor_quantized, comparsion_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step% 128== 0:print(f'epoch: {ep+ 1}, step: {step+ 1}, l1 loss: {loss.item()}')
        mri_encoder.eval();img_hats= [];imgs= []
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                hr_data, lr_data= data['hr'].to(torch.float32).to(args.device), data['lr'].to(torch.float32).to(args.device)
                tensor= mri_encoder(lr_data)
                recon_x= vqgan.decode(tensor, quantize= True)
                img_hats.append(recon_x.cpu());imgs.append(hr_data.cpu())
        imgs, img_hats= torch.cat(imgs, dim= 0), torch.cat(img_hats, dim= 0)
        img_hats, imgs= img_hats.chunk(3, dim= 0), imgs.chunk(3, dim= 0)
        mean_psnr, mean_ssim= [], []
        for i in range(len(img_hats)):
            img_hat, img= img_hats[i], imgs[i]
            recon_x, x= restore_img(img.cpu().numpy().reshape(-1, 3, 64, 64, 64), img_hat.cpu().numpy().reshape(-1, 3, 64, 64, 64), args.patch_size, args.stride, args.size)
            p_val, s_val= calculate_metrices(x, recon_x)
            mean_psnr.append(p_val);mean_ssim.append(s_val)
        mean_ssim, mean_psnr= torch.tensor(mean_ssim).mean(), torch.tensor(mean_psnr).mean()
        # early_stopping
        early_stopping(-(mean_ssim+ mean_psnr), mri_encoder)
        print(f'epoch: {ep+ 1}, early_stopping: {early_stopping.clock+ 1}/{early_stopping.patience}, p_val: {p_val}, s_val: {s_val}')
        if early_stopping.flag:break

if __name__ == '__main__':
    args= setting_para()
    run(args)
