import torch
import argparse
import warnings
from vqgan import VQGAN
from dl import VQDataset
from unet3d import UNet3D
from lrmri_encoder import LREncoder
from ddpm import GaussianDiffusion, Trainer
warnings.filterwarnings("ignore")

def setting_para():
    parser= argparse.ArgumentParser(description= 'Parser for Arguments')
    parser.add_argument('-train_hr_NODDI', type= str, default= '../../../patch_brain_size64/30noddi_train_hr_30sub.npy')
    parser.add_argument('-train_lr_MRI', type= str, default= '../../../patch_brain_size64/30noddi_train_lr_30sub.npy')
    parser.add_argument('-val_hr_NODDI', type= str, default= '../../../patch_brain_size64/30noddi_val_hr_3sub.npy')
    parser.add_argument('-val_lr_MRI', type= str, default= '../../../patch_brain_size64/30noddi_val_lr_3sub.npy')
    parser.add_argument('-val_sub_num', int, default= 3)
    parser.add_argument('-in_ch', type= int, default= 8)
    parser.add_argument('-out_ch', type= int, default= 8)
    parser.add_argument('-cond_ch', type= int, default= 8)
    parser.add_argument('-basic_ch', type= int, default= 128)
    parser.add_argument('-num_res_blocks', type= int, default= 2)
    parser.add_argument('-with_condition', type= bool, default= True)
    parser.add_argument('-channel_mult', type= list, default= [1, 2, 2, 4])
    parser.add_argument('-attention_resolutions', type= list, default= [2, 4, 8])
    parser.add_argument('-lr', type= float, default= 1e-4)
    parser.add_argument('-epochs', type= int, default= 1e8)
    parser.add_argument('-patience', type= int, default= 1e6)
    parser.add_argument('-loss_type', type= str, default= "l1")
    parser.add_argument('-train_batch_size', type= int, default= 2)
    parser.add_argument('-valid_batch_size', type= int, default= 2)
    parser.add_argument('-weight_decay', type= float, default= 5e-5)
    parser.add_argument('-device', type= str, default= 'cuda:1')
    parser.add_argument('-device_ids', type= list, default= [1])
    parser.add_argument('-main_device_id', type= int, default= '1')
    parser.add_argument('-denoise_step', type= int, default= 1000)
    parser.add_argument('-valid_denoise_step', type= int, default= 1000)
    parser.add_argument('-stride', type= int, default= 36)
    parser.add_argument('-patch_size', type= int, default= 64)
    parser.add_argument('-image_size', type= int, default= 32)
    parser.add_argument('-size', type= tuple, default= (3, 136, 172, 136))
    parser.add_argument('-save_and_sample_every', type= int, default= 1000)
    parser.add_argument('-gradient_accumulate_every', type= int, default= 1)
    parser.add_argument('-results_folder', type= str, default= '../pt')
    parser.add_argument('-ddpm_pt_name', type= str, default= 'ddpm.pt')
    parser.add_argument('-save_img_path', type= str, default= '../image')
    parser.add_argument('-pt_path_vqgan', type= str, default= '../pt/vqgan.pt')
    parser.add_argument('-ori_img_path', type= str, default= '../image/ori_img.npy')
    parser.add_argument('-con_img_path', type= str, default= '../image/con_img.npy')
    parser.add_argument('-pt_path_mri_encoder', type= str, default= '../pt/lrmri_encoder.pt')
    args= parser.parse_args([])
    return args

if __name__== '__main__':
    args= setting_para()
    torch.cuda.set_device(args.main_device_id)
    vqgan= torch.load(args.pt_path_vqgan, map_location= args.device)
    vqgan.args.device= args.device
    unet= UNet3D(
        in_ch= args.in_ch,
        cond_ch= args.cond_ch, 
        basic_ch= args.basic_ch, 
        out_ch= args.out_ch, 
        num_res_blocks= args.num_res_blocks, 
        attention_resolutions= args.attention_resolutions, 
        channel_mult= args.channel_mult)
    lr_encoder= torch.load(args.pt_path_mri_encoder, map_location= args.device)
    train_dataset= VQDataset(args.train_hr_NODDI, args.train_lr_MRI)
    valid_dataset= VQDataset(args.val_hr_NODDI, args.val_lr_MRI)
    diffusion= GaussianDiffusion(
        denoise_fn= unet, 
        lr_encoder= lr_encoder, 
        vqgan= vqgan, 
        image_size= args.image_size, 
        depth_size= args.image_size, 
        timesteps= args.denoise_step, 
        loss_type= args.loss_type, 
        with_condition= args.with_condition, 
        channels= args.latent_in_channel, 
        device= args.device).cuda()
    trainer = Trainer(
        diffusion_model= diffusion,
        train_dataset= train_dataset,
        val_dataset= valid_dataset,
        train_batch_size= args.train_batch_size,
        valid_batch_size= args.valid_batch_size,
        val_sub_num= args.val_sub_num,
        device= args.device,
        save_and_sample_every= args.save_and_sample_every,
        train_lr= args.lr,
        weight_decay= args.weight_decay,
        train_num_steps= args.epochs,
        gradient_accumulate_every= args.gradient_accumulate_every,
        valid_denoise_step= args.valid_denoise_step,
        patch= args.patch_size,
        stride= args.stride,
        image_size= args.size,
        results_folder= args.results_folder,
        patience= args.patience
    )   
    trainer.train()