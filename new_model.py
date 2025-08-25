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
from modeling_pretraining import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask
from configs import *
#-- use channels for model

use_channels_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", ]

class LitEEGPT(nn.Module):

    def __init__(self, models_configs, USE_LOSS_A=False, USE_LN=True, USE_SKIP=True):
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
        
    def make_masks(self, num_patchs, mC_x=4, p_n_y=0.5, p_c_y=0.8):
        
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
        # 输出的x是二维数组()  第一个维度表示保留的时间步（N）的序号 每个时间步有50%的几率保留下来 
        # 第二个维度是当前时间步随机保留的导联（mC_x）
        # mask_y_bx表示如果当前时间步产生了保留，那么被mask掉的剩余导联有20%的几率不参与重建，仅保留当前时间步80%的导联参与重建
        # 因为如果当前时间步保留了，那么当前时间步的其它导联很容易被重建，导致重建任务简单
        # mask_y表示当前时间步没有发生保留，那么当前时间步的所有导联都可以参与重建
        # 因此最终输出的mask_x表示没有被mask的部分 mask_t表示需要被重建的部分
        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0) 
    
    def forward_target(self, x, mask_y):
        with torch.no_grad():
            h = self.target_encoder(x, self.chans_id.to(x)) # B, N, embed_num, D
            h = F.layer_norm(h, (h.size(-1),))
            C, N = self.encoder.num_patches # C=12 N=30
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n) # bs, C, bc, N, bn
            # 将维度重新排列以使分块沿着通道轴和空间轴
            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n) # B, N, C, bc*bn
            y = apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y

    def forward_context(self, x, mask_x, mask_y):
        z = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z = z
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r
    
    def forward(self, x):
        # print('开始创建mask')
        # print(self.encoder.num_patches)
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        print(mask_x.shape)
        print(mask_y.shape) # mask_x 表示没有mask的部分 mask_y表示被mask的部分 需要通过mask_x来预测出mask_y
        h, y = self.forward_target(x, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        loss1 = self.loss_fn(h, z)
        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss  = loss1 + loss2
        else:
            loss  = loss2                
        return {"loss": loss,
                "loss1": loss1,
                "loss2": loss2}
    
    

#-- modeling
def seed_torch(seed=2024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
