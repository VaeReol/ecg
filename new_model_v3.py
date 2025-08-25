import os
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random
import copy
from modeling_pretraining_new import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, apply_mask, ECGProjection
from configs import *
from augmentation import *
#-- use channels for model
# 根据"I", "II", "V1", "V2", "V3", "V4", "V5", "V6", 就可以推倒出全导联的信息
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
        projection = ECGProjection()
        target_encoder = copy.deepcopy(encoder)
        target_projection = copy.deepcopy(projection)
        for p in target_encoder.parameters():
            p.requires_grad = False
        for p in target_projection.parameters():
            p.requires_grad = False  
        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.projection = projection
        self.target_projection = target_projection
        self.chans_id       = encoder.prepare_chan_ids(use_channels_names)
        self.loss_fn        = torch.nn.MSELoss()

    def normal_contra_loss(self, z1, z2):
        '''
        Computes a contrastive loss for embeddings z1 and z2 based on the SimCLR framework.
        All diagonal pairs are considered positives, all others negatives.

        Args:
            z1 (torch.Tensor): Embeddings from view 1, shape [B, H].
            z2 (torch.Tensor): Embeddings from view 2, shape [B, H].
            id (torch.Tensor): Unused in this implementation, kept for compatibility.

        Returns:
            torch.Tensor: The computed contrastive loss.
        '''
        # Ensure all tensors are on the same device
        device = z1.device
        batch_size = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.T) / 0.2  # Divide by temperature
        
        # Create positive mask (diagonal elements)
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Compute exponential of similarity matrix
        sim_matrix_exp = torch.exp(sim_matrix)
        
        # For each positive pair (i,i), the denominator is sum_j exp(sim(i,j)) except i=j
        # This can be computed as sum of all elements in row i minus the diagonal element
        denom = sim_matrix_exp.sum(dim=1) - torch.diag(sim_matrix_exp)
        
        # Extract positive similarities
        pos_sim = sim_matrix_exp[pos_mask]
        
        # Compute InfoNCE loss: -log(exp(sim(i,i)) / (sum_j≠i exp(sim(i,j))))
        loss = -torch.mean(torch.log(pos_sim / denom))
        
        return loss

    def simclr_id_loss(self, z1, z2, id):
        '''
        Computes a contrastive loss for embeddings z1 and z2 based on the SimCLR framework and subject ID pairing.

        Args:
            z1 (torch.Tensor): Embeddings from view 1, shape [B, H].
            z2 (torch.Tensor): Embeddings from view 2, shape [B, H].
            id (torch.Tensor): Subject IDs corresponding to embeddings, shape [B].

        Returns:
            torch.Tensor: The computed contrastive loss.
        '''
        # Ensure all tensors are on the same device
        device = z1.device

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.T)
        sim_matrix_exp = torch.exp(sim_matrix / 0.2)

        # Convert IDs to a boolean matrix for positive pairs
        id_matrix = id.unsqueeze(1) == id.unsqueeze(0)  # Boolean matrix for matching IDs

        # Get upper and lower triangle indices
        rows1, cols1 = torch.triu_indices(id.size(0), id.size(0), offset=1, device=device)
        rows2, cols2 = torch.tril_indices(id.size(0), id.size(0), offset=-1, device=device)

        # Diagonal elements (positive pairs)
        diag_elements = torch.diag(sim_matrix_exp)
        triu_sum = sim_matrix_exp.sum(dim=1)
        tril_sum = sim_matrix_exp.sum(dim=0)

        # Loss terms for diagonal
        loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
        loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))
        loss = loss_diag1 + loss_diag2
        loss_terms = 2

        # Upper triangle positive pairs
        upper_mask = id_matrix[rows1, cols1].to(device)  # Ensure mask is on the correct device
        if upper_mask.any():
            selected_rows = rows1[upper_mask]
            selected_cols = cols1[upper_mask]
            triu_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
            loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[selected_rows]))
            loss += loss_triu
            loss_terms += 1

        # Lower triangle positive pairs
        lower_mask = id_matrix[rows2, cols2].to(device)  # Ensure mask is on the correct device
        if lower_mask.any():
            selected_rows = rows2[lower_mask]
            selected_cols = cols2[lower_mask]
            tril_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
            loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[selected_cols]))
            loss += loss_tril
            loss_terms += 1

        # Final loss normalization
        return loss / loss_terms if loss_terms > 0 else 0

    def aug_x(self, x):
        augmentation = ["frequency", "jitter", "mask", "channel"]
        aug_list = nn.ModuleList(
            [self.get_augmentation(aug) for aug in augmentation]
        )
        aug_idx = random.randint(0, len(augmentation) - 1)
        x_new_t = aug_list[aug_idx](x)
        return x_new_t

    def get_augmentation(self, augmentation):
        # 添加随机高斯噪声
        if augmentation.startswith("jitter"):
            if len(augmentation) == 6:
                return Jitter()
            return Jitter(float(augmentation[6:]))
        # 
        elif augmentation.startswith("drop"):
            if len(augmentation) == 4:
                return nn.Dropout(0.1)
            return nn.Dropout(float(augmentation[4:]))
        elif augmentation.startswith("flip"):
            if len(augmentation) == 4:
                return Flip()
            return Flip(float(augmentation[4:]))
        elif augmentation.startswith("shuffle"):
            if len(augmentation) == 7:
                return Shuffle()
            return Shuffle(float(augmentation[7:]))
        elif augmentation.startswith("frequency"):
            if len(augmentation) == 9:
                return FrequencyMask()
            return FrequencyMask(float(augmentation[9:]))
        elif augmentation.startswith("mask"):
            if len(augmentation) == 4:
                return TemporalMask()
            return TemporalMask(float(augmentation[4:]))
        elif augmentation.startswith("channel"):
            if len(augmentation) == 7:
                return ChannelMask()
            return ChannelMask(float(augmentation[7:]))
        elif augmentation == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown augmentation {augmentation}")
        
    def make_masks(self, num_patchs, mC_x=6, p_n_y=0.5, p_c_y=0.8):
        
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

    def make_random_masks(self, num_patches, mask_ratio=0.4):
        C, N = num_patches  # C 是类别数，N 是时间步数或批次大小


        # 生成空列表，用于存储 mask_x 和 mask_y
        mask_x = []
        mask_y = []

        # 遍历每个时间步
        for i in range(N):
            # 为当前时间步生成一个随机的掩码序号
            c_idx = torch.randperm(C) + i * C  # 每个时间步生成一个随机序号，并加上偏移

            # 计算保留的数量：根据mask_ratio选择要保留的部分
            num_to_keep = int(mask_ratio * C)  # 每个时间步保留40%（即C*0.4）
            
            # mask_x 保留的序号
            mask_x.append(c_idx[:num_to_keep])  # 保留的部分
            # mask_y 为剩下的丢弃的序号
            mask_y.append(c_idx[num_to_keep:])  # 丢弃的部分

        # 将 mask_x 和 mask_y 合并成一个张量返回
        mask_x_tensor = torch.stack(mask_x, dim=0)
        mask_y_tensor = torch.stack(mask_y, dim=0)

        # 将 mask_y 张量铺平成一维
        mask_y_flattened = mask_y_tensor.flatten()

        return mask_x_tensor, mask_y_flattened


    def make_masks_leads(self, num_patchs, mC_x=4, p_n_y=0.5, p_c_y=0.8):
        C, N = num_patchs
        
        while True:
            mask_x = []  # mN, mC
            mask_y = []
            mask_y_bx = []
            
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                
                # 检查前mC_x个索引中属于原始前6个通道的数量
                # 原始前6个通道的全局索引为 i*C 到 i*C+5
                orig_first_six = set(range(i*C, i*C+6))
                selected = c_idx[:mC_x]
                
                # 计算选中的索引中属于原始前6个通道的数量
                count = sum(1 for idx in selected if idx.item() in orig_first_six)
                
                # 如果超过2个，则重新生成排列，直到满足条件
                while count > 2:
                    c_idx = torch.randperm(C) + i*C
                    selected = c_idx[:mC_x]
                    count = sum(1 for idx in selected if idx.item() in orig_first_six)
                
                if random.random() > p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            
            if len(mask_x) == 0:
                continue
            if len(mask_y_bx) == 0:
                continue
            
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape) < p_c_y]
            
            if len(mask_y_bx) == 0:
                continue
            
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y + [mask_y_bx], dim=0)
        
    def forward_target(self, x, x_aug, mask_y):
        with torch.no_grad():
            h = self.target_encoder(x_aug, self.chans_id.to(x_aug)) # B, N, embed_num, D
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
        print(comb_z.shape)
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r
    
    def forward_con(self, z, h):
        # print(z.shape)
        latent = self.projection(z)
        with torch.no_grad():
            latent_aug = self.target_projection(h)
        return latent, latent_aug

    def forward(self, x, sample_id):
        # print('开始创建mask')
        # print(self.encoder.num_patches)
        x_aug = x.clone()
        x_aug = self.aug_x(x_aug)
        mask_x, mask_y = self.make_masks(self.encoder.num_patches)
        # print(mask_x.shape)
        # print(mask_y)
        # mask_x 表示没有mask的部分 mask_y表示被mask的部分 需要通过mask_x来预测出mask_y
        # print(mask_x.shape)
        # print(mask_y.shape)
        h, y = self.forward_target(x, x_aug, mask_y)
        z, r = self.forward_context(x, mask_x, mask_y)
        latent, latent_aug = self.forward_con(z, h)
        # loss1 = self.normal_contra_loss(latent, latent_aug)
        loss1 = self.normal_contra_loss(latent, latent_aug)
        loss2 = self.loss_fn(y, r)
        if self.USE_LOSS_A:
            loss  = 0.3 * loss1 + 0.7 * loss2
            return {"loss": loss,
                "loss1": loss1,
                "loss2": loss2}
        else:
            loss  = loss2                
            return {"loss": loss,
                    "loss1": loss2,
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
 
if __name__ == '__main__':
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def get_config(embed_dim=256, embed_num=1, depth=[10,10,10], num_heads=4): # 23977441
    
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
    x = torch.randn((64, 12, 2250))
    x_aug = nn.Dropout(0.1)(x)
    random_integers = torch.randint(0, 10, (64,))
    model_configs = get_config()
    model = LitEEGPT(models_configs=model_configs)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    result = model(x, random_integers)
    print(result['loss'])
    print(result['loss1'])
    print(result['loss2'])