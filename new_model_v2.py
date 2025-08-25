import os
import math
from typing import Any, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random
import copy
from utils import WarmupCosineSchedule, CosineWDSchedule, grad_logger
from model_pretraining_v2 import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask
from configs import *
#-- use channels for model

use_channels_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", ]

class LitEEGPT(nn.Module):

    def __init__(self, models_configs, USE_LOSS_A=True, USE_LN=True, USE_SKIP=True):
        super().__init__()    
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LN     = USE_LN
        self.USE_SKIP   = USE_SKIP
        
        encoder = EEGTransformer(
            img_size=[12, 2250],
            patch_size=75,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])
        
        predictor = EEGTransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor'])
        
        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=75,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor'])
        
        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
            
        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.chans_id       = encoder.prepare_chan_ids(use_channels_names)
        
        self.loss_fn        = torch.nn.MSELoss()
        
    def make_masks(self, num_patchs, mC_x=4, p_n_y=0.5, p_c_y=0.2):
        
        C, N = num_patchs
        
        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random()>p_n_y:
                    mask_x.append(c_idx[:mC_x]) # [[m个导联], [m个导联], ..., [m个导联]] 
                    mask_y_bx.append(c_idx[mC_x:]) # [[n个导联], [n个导联], ..., [n个导联]] 
                else:
                    mask_y.append(c_idx) # [[全导联], [全导联], ..., [全导联]] n个元素
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y] # 一维数组
            if len(mask_y_bx)==0: continue
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0) # (m, 12)  (一维数组)
    
    def forward_target(self, x, mask_y):
        with torch.no_grad():
            h, fft1, fft2 = self.target_encoder(x, self.chans_id.to(x)) # B, N, embed_num, D
            # print(x.shape)
            # print(fft1.shape)
            # print(fft2.shape)
            h = F.layer_norm(h, (h.size(-1),))
            C, N = self.encoder.num_patches # C=12 N=30
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n) # bs, C, bc, N, bn
            # 将维度重新排列以使分块沿着通道轴和空间轴
            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n) # B, N, C, bc*bn
            fft1 = fft1.view(fft1.shape[0], C, N, block_size_c * block_size_n) # B, N, C, bc*bn
            fft2 = fft2.view(fft2.shape[0], C, N, block_size_c * block_size_n) # B, N, C, bc*bn
            y = apply_mask(mask_y.to(x.device), x)
            fft1 = apply_mask(mask_y.to(x.device), fft1)
            fft2 = apply_mask(mask_y.to(x.device), fft2)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y, fft1, fft2

    def forward_context(self, x, mask_x, mask_y):
        z, fft1, fft2 = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z = z
        r, rec_fft1, rec_fft2 = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r, fft1, fft2, rec_fft1, rec_fft2
    
    def forward(self, x):
        # print('开始创建mask')
        # print(self.encoder.num_patches)
        mask_x, mask_y = self.make_masks(self.encoder.num_patches) # mask_x 表示被mask的部分 mask_y表示没有被mask的部分 需要通过mask_x来预测出mask_y
        h, y, fft1_origin, fft2_origin = self.forward_target(x, mask_y)
        z, r, fft1, fft2, rec_fft1, rec_fft2 = self.forward_context(x, mask_x, mask_y)
        # print(fft1_origin.shape)
        # print(rec_fft1.shape)
        # print(r.shape)
        loss1 = self.loss_fn(h, z)
        loss2 = self.loss_fn(y, r)
        loss_amp = self.loss_fn(fft1_origin, rec_fft1)
        loss_angle = self.loss_fn(fft2_origin, rec_fft2)
        loss = loss1 + loss2 + loss_amp + loss_angle              
        return {"loss": loss,
                "loss1": loss1,
                "loss2": loss2,
                "loss_amp": loss_amp,
                "loss_angle": loss_angle}
    
    

#-- modeling
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
if __name__ == '__main__':
    def get_config(embed_dim=512, embed_num=1, depth=[6,6,6], num_heads=8):
        models_configs = {
                'encoder': {
                        'embed_dim': embed_dim,
                        'embed_num': embed_num,
                        'depth': depth[0],
                        'num_heads': num_heads,
                    },
                'predictor': {
                        'embed_dim': embed_dim,
                        'embed_num': embed_num,
                        'predictor_embed_dim': embed_dim,
                        'depth': depth[1],
                        'num_heads': num_heads,
                    },
                'reconstructor': {
                        'embed_dim': embed_dim,
                        'embed_num': embed_num,
                        'reconstructor_embed_dim': embed_dim,
                        'depth': depth[2],
                        'num_heads': num_heads,
                    },
        }
        return models_configs
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_configs = get_config()
    model = LitEEGPT(models_configs=model_configs)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    x = torch.randn((64, 12, 2250))
    result = model(x)
    print(result)