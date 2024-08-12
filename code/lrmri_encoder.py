import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from vqgan import Encoder, SamePadConv3d
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many

def default(a, b):
    if a:return a
    return b() if callable(b) else b

def exist(a):
    return False if a== None else True

def normalization(channels, num_ch= 32, _= None):
    _= [channels// num_ch, 1][channels<= num_ch or channels% num_ch!= 0] if not exist(_) else _
    return nn.GroupNorm(_, channels)

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale= 2):
        super().__init__()
        self.seq= nn.Sequential(Rearrange('b c (h s1) (w s2) (d s3) -> b (c s1 s2 s3) h w d', s1= scale, s2= scale, s3= scale), nn.Conv3d(in_ch* (scale** 3), out_ch, 1))
    def forward(self, x):
        return self.seq(x)
    
class PixelShuffle3D(nn.Module):
    '''This class is a 3d version of pixelshuffle.'''
    def __init__(self, scale):
        ''':param scale: upsample scale'''
        super().__init__()
        self.scale= scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width= input.size()
        nOut= channels // self.scale ** 3
        out_depth= in_depth * self.scale
        out_height= in_height * self.scale
        out_width= in_width * self.scale
        input_view= input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output= input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    
class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out= None, scale= 2):
        super().__init__()
        dim_out= default(dim_out, dim)
        conv= nn.Conv3d(dim, dim_out* (scale** 3), 1, padding_mode='replicate')
        self.net= nn.Sequential(conv, nn.SiLU(), PixelShuffle3D(2))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w, d= conv.weight.shape
        conv_weight= torch.empty(o // 4, i, h, w, d)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight= repeat(conv_weight, 'o ... -> (o 4) ...')
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)
    
class SelfAttention(nn.Module):
    # transformer version
    def __init__(self, d_embed= 128, n_head= 4):
        super().__init__()
        self.ln1, self.ln2= nn.LayerNorm(d_embed), nn.LayerNorm(d_embed)
        self.mha= nn.MultiheadAttention(d_embed, n_head, batch_first= True)
        self.ff= nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed)
        )
    def forward(self, x):
        x_ln1= self.ln1(x)
        x= self.mha(x_ln1, x_ln1, x_ln1)[0]+ x
        x_ln2= self.ln2(x)
        return self.ff(x_ln2)+ x
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size= 3):
        super().__init__()
        assert kernel_size% 2== 1, 'kernel_size must be odd.'
        self.norm= normalization(in_ch)
        self.act= nn.SiLU()
        self.proj= nn.Conv3d(in_ch, out_ch, kernel_size, padding= (kernel_size- 1)// 2)

    def forward(self, x):
        x= self.norm(x)
        x = self.act(x)
        return self.proj(x)
    
class CrossEmbedLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes= [3], stride= 1):
        super().__init__()
        assert stride== 1, 'It is under development.'
        kernel_sizes= sorted(kernel_sizes)
        num_scales= len(kernel_sizes)
        # calculate the dimension at each scale
        channel_scales= [int(out_ch / (2 ** i)) for i in range(1, num_scales)]
        channel_scales= [*channel_scales, out_ch- sum(channel_scales)]
        # conv
        self.convs= nn.ModuleList([])
        for ks, ch_scale in zip(kernel_sizes, channel_scales):
            assert ks% 2== 1, 'kernel_size must be odd.'
            self.convs.append(nn.Conv3d(in_ch, ch_scale, ks, stride, (ks- 1)// 2))

    def forward(self, x):
        fmaps= tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)
    
def window_partition(x, window_size= 8):
    B, C, F, H, W= x.shape
    x= x.view(B, C, F// window_size, window_size, H// window_size, window_size, W// window_size, window_size)
    windows= x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, window_size* window_size* window_size, C)
    return windows

def window_reverse(windows, window_size= 8, F= 32, H= 32, W= 32):
    B= int(windows.shape[0] / (F* H * W / window_size / window_size/ window_size))
    x= windows.view(B, F// window_size, H// window_size, W// window_size, window_size, window_size, window_size, -1)
    x= x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, -1, F, H, W)
    return x

class Conv3d(nn.Module):
    # mode 0, spatial; mode 1, temporal; mode 2 all.
    def __init__(self, in_ch, out_ch, kernel_size= 3, temporal_kernel_size= 3, mode= 2, padding_mode= 'zeros'):
        super().__init__()
        assert kernel_size% 2== 1, 'kernel_size shoud be odd.'
        self.mode= mode
        self.temporal_kernel_size= temporal_kernel_size
        self.spatial_conv= nn.Identity() if mode== 1 else nn.Conv2d(in_ch, out_ch, kernel_size= kernel_size, padding= (kernel_size- 1)// 2 if kernel_size> 1 else 0, padding_mode= padding_mode)
        temporal_in_ch= in_ch if mode== 1 else out_ch
        self.temporal_conv= nn.Identity() if mode== 0 else nn.Conv1d(temporal_in_ch, out_ch, kernel_size = temporal_kernel_size, padding_mode= padding_mode)

    def forward(self, x):
        b= x.shape[0]
        if self.mode!= 1:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            x = self.spatial_conv(x)
            x = rearrange(x, '(b f) c h w -> b c f h w', b = b)
        if self.mode!= 0:
            h, w= x.shape[-2:]
            x = rearrange(x, 'b c f h w -> (b h w) c f')
            # causal temporal convolution - time is causal in imagen-video
            if self.temporal_kernel_size> 1:x = F.pad(x, (self.temporal_kernel_size- 1, 0))
            x = self.temporal_conv(x)
            x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)
        return x
    
class SelfAttention(nn.Module):
    # transformer version
    def __init__(self, d_embed= 128, n_head= 4):
        super().__init__()
        self.ln1, self.ln2= LayerNorm(d_embed), LayerNorm(d_embed)
        self.mha= nn.MultiheadAttention(d_embed, n_head, batch_first= True)
        self.ff= nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed)
        )
    def forward(self, x):
        x_ln1= self.ln1(x)
        x= self.mha(x_ln1, x_ln1, x_ln1)[0]+ x
        x_ln2= self.ln2(x)
        return self.ff(x_ln2)+ x
    
class GMLPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size= 3):
        super().__init__()
        assert kernel_size% 2== 1, 'kernel_size must be odd.'
        self.pre_gate_conv= nn.Sequential(normalization(in_ch), nn.SiLU(), nn.Conv3d(in_ch, 2* out_ch, kernel_size, padding= (kernel_size- 1)// 2))
        self.gate_conv= nn.Sequential(normalization(out_ch), nn.SiLU(), nn.Conv3d(out_ch, out_ch, kernel_size, padding= (kernel_size- 1)// 2), nn.Sigmoid())
        self.channel_proj= nn.Conv3d(out_ch, out_ch, kernel_size, padding= (kernel_size- 1)// 2)
        self.res_conv= nn.Conv3d(in_ch, out_ch, kernel_size, padding= (kernel_size- 1)// 2)
    def forward(self, x):
        residual_long= x.clone()
        x= self.pre_gate_conv(x)
        x= x.chunk(2, dim= 1)
        x= x[0]* self.gate_conv(x[1])
        return self.channel_proj(x)+ self.res_conv(residual_long)
    
class RAB(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size= 3, n_head= 4, window_size= 8, use_attn= True):
        super().__init__()
        assert kernel_size% 2== 1, 'kernel_size must be odd.'
        self.window_size= window_size
        self.use_attn= use_attn
        self.seq= nn.Sequential(Block(in_ch, out_ch, kernel_size), Block(out_ch, out_ch, kernel_size))
        self.sa= SelfAttention(out_ch, n_head= n_head) if use_attn else nn.Identity()
        self.res_conv= nn.Conv3d(in_ch, out_ch, kernel_size= kernel_size, padding= (kernel_size- 1)// 2)

    def forward(self, x):
        b, c, f, h, w= x.shape
        residual_short= x.clone()
        # residual block
        x= self.seq(x)
        # attention block
        if self.use_attn:
            x= window_partition(x, self.window_size)
            x= self.sa(x)
            x= window_reverse(x, self.window_size, f, h, w)
        # residual block
        return x+ self.res_conv(residual_short)

class LREncoder(nn.Module):
    def __init__(self, in_ch, scale, hid_ch, out_ch, downsample= (2, 2, 2), kernel_sizes= [3, 5, 9, 13]):
        super().__init__()
        self.scale= scale
        # multi-scale conv
        self.pre_conv= CrossEmbedLayer(in_ch, hid_ch, kernel_sizes)
        # vqgan encoder
        self.encoder= Encoder(n_hiddens= hid_ch, image_channel= hid_ch, norm_type= 'group', downsample= downsample)
        # residual 
        self.residual4enc= nn.Conv3d(hid_ch, self.encoder.out_channels, 3, 2, 1)
        self.post_conv= nn.Sequential(normalization(self.encoder.out_channels), nn.SiLU(), nn.Conv3d(self.encoder.out_channels, out_ch, kernel_size= 3, padding= 1))

    def forward(self, x):
        # upsample
        x= self.pre_conv(F.interpolate(x, scale_factor= self.scale, mode= 'trilinear'))
        # long residual
        residual_short= x.clone()
        # encoder
        x= self.encoder(x)+ self.residual4enc(residual_short)
        # post conv+ residual
        return self.post_conv(x)