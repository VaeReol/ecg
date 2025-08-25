import os
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random
import copy
from model_new import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor, ECGProjection, EEGTransformerDecoder
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
        
        decoder = EEGTransformerDecoder(
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
        self.decoder = decoder
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
        
    def forward_target(self, x_aug):
        with torch.no_grad():
            h = self.target_encoder(x_aug, self.chans_id.to(x_aug), need_mask=False) # B, N, embed_num, D
            h = F.layer_norm(h, (h.size(-1),))
            return h

    def forward_context(self, x):
        z, mask, ids_restore = self.encoder(x, self.chans_id.to(x))
        z_con = self.predictor(z, ids_restore)
        z_rec = self.decoder(z, ids_restore)
        return z_con, z_rec, mask
    
    def forward_con(self, z, h):
        # print(z.shape)
        latent = self.projection(z)
        with torch.no_grad():
            latent_aug = self.target_projection(h)
        return latent, latent_aug
    
    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = 75
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b n c p', p=p)
        return x
    
    def forward_loss(self, series, pred, mask):
        """
        series: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        mask: (batch_size, num_leads, n), 0 is keep, 1 is remove,
        """
        target = self.patchify(series) # （bs, 12, 30, 75)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (batch_size, num_leads, n), mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, x, sample_id):
        x_aug = x.clone()
        x_aug = self.aug_x(x_aug)
        aug_latent = self.forward_target(x_aug)
        z_con, z_rec, mask = self.forward_context(x)
        latent, latent_aug = self.forward_con(z_con, aug_latent)
        # loss1 = self.normal_contra_loss(latent, latent_aug)
        loss1 = self.normal_contra_loss(latent, latent_aug)
        loss2 = self.forward_loss(x, z_rec, mask)
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