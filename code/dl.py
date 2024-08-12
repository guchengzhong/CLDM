import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class VQDataset(Dataset):
    def __init__(self, hrNODDI= '../../../patch_brain_size64/30noddi_train_hr_40sub.npy', lrMRI= '../../../patch_brain_size64/30noddi_train_hr_40sub.npy'):
        self.NODDI_hr_img= np.load(hrNODDI).reshape(-1, 1, 64, 64, 64)
        self.MRI_lr_img= np.load(lrMRI).reshape(-1, 1, 32, 32, 32)
    def __len__(self):
        return self.NODDI_hr_img.shape[0]
    def __getitem__(self, index) -> torch.tensor:
        return {'hr': torch.tensor(self.NODDI_hr_img[index]), 'lr': torch.tensor(self.MRI_lr_img[index])}

class Early_stopping:
    def __init__(self, patience, pt_path1, pt_path2= None):
        self.patience= patience
        self.clock= 0
        self.pt_path1= pt_path1
        self.pt_path2= pt_path2
        self.flag= False
        self.rec_loss= 1e8
    def __call__(self, val_loss, net1, net2= None):
        if val_loss< self.rec_loss:
            self.rec_loss= val_loss
            self.clock= 0
            torch.save(net1, self.pt_path1)
            if net2!= None:torch.save(net2, self.pt_path2)
        else:
            self.clock+= 1
            if self.clock>= self.patience:self.flag= True

## compute metrices
def calculate_metrices(img1, img2):
    # evaulateion
    img1_transpose= np.transpose(img1, (1, 2, 3, 0))
    img2_transpose= np.transpose(img2, (1, 2, 3, 0))
    img1_2d= img1_transpose.reshape(-1, img1_transpose.shape[3])
    img2_2d= img2_transpose.reshape(-1, img2_transpose.shape[3])
    psnr_val= psnr(img1_2d, img2_2d, data_range= img1_2d.max()- img1_2d.min())
    ssim_val= ssim(img1_2d, img2_2d, channel_axis= 1, data_range= img1_2d.max()- img1_2d.min())
    return psnr_val, ssim_val

## restore image
def restore_img(gt, pred, patch_size, stride, size):
    C, D, H, W= size
    pred_= np.zeros(size, dtype= np.float32)
    gt_= np.zeros(size, dtype= np.float32)
    overlap_count= np.zeros((1, D, H, W), dtype= np.float32)
    # compute indices
    D_indices= range(0, D- patch_size+ 1, stride)
    H_indices= range(0, H- patch_size+ 1, stride)
    W_indices= range(0, W- patch_size+ 1, stride)
    # meshgrid 
    D_grid, H_grid, W_grid= np.meshgrid(D_indices, H_indices, W_indices, indexing= 'ij')
    indices= np.vstack([D_grid.ravel(), H_grid.ravel(), W_grid.ravel()]).T
    for i, (xx, yy, zz) in enumerate(indices):
        pred_[:, xx: xx+patch_size, yy: yy+patch_size, zz: zz+patch_size]+= pred[i, :, :, :, :]
        overlap_count[0, xx: xx+patch_size, yy: yy+patch_size, zz: zz+patch_size]+= 1
    for i, (xx, yy, zz) in enumerate(indices):
        gt_[:, xx: xx+patch_size, yy: yy+patch_size, zz: zz+patch_size]+= gt[i, :, :, :, :]
    overlap_count+= overlap_count== 0
    pred_= pred_/ overlap_count
    gt_= gt_/ overlap_count
    return pred_, gt_