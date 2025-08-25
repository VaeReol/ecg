import os
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random
import copy
from model_new import EEGTransformer, EEGTransformerPredictor, ECGProjection, EEGTransformerDecoder
from configs import *
from augmentation import *
from utils import AutomaticWeightedLoss
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
            **models_configs['decoder'])
        
        # reconstructor = EEGTransformerReconstructor(
        #     num_patches=encoder.num_patches,
        #     patch_size=75,
        #     mlp_ratio=4.0,
        #     drop_rate=0.0,
        #     attn_drop_rate=0.0,
        #     drop_path_rate=0.0,
        #     init_std=0.02,
        #     qkv_bias=True, 
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     **models_configs['reconstructor'])
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
        # self.reconstructor  = reconstructor
        self.projection = projection
        self.target_projection = target_projection
        self.chans_id       = encoder.prepare_chan_ids(use_channels_names)
        self.loss_fn        = torch.nn.MSELoss()
        self.awl = AutomaticWeightedLoss(2)

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
        labels = torch.arange(batch_size).long().to(device)
        labels = labels // 30
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


    def fft_aug(self, x, rate=0.5, dim=1):
    # Ensure that x and all other tensors are on the same device (GPU in this case)
        device = x.device  # Get the device of x
        
        # Perform FFT
        x_f = torch.fft.fft(x, dim=dim)

        # Create the mask on the same device as x
        m = torch.empty_like(x_f, dtype=torch.float32).uniform_() < rate
        m = m.to(device)  # Ensure that m is on the same device as x_f
        
        # Compute amplitude and dominant mask
        amp = abs(x_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m, dominant_mask)
        
        # Mask the real and imaginary parts
        freal = x_f.real.masked_fill(m, 0)
        fimag = x_f.imag.masked_fill(m, 0)

        # Shuffle the batch indices and create a new batch x2
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2 = x[b_idx]
        
        # Perform FFT on the shuffled batch x2
        x2_f = torch.fft.fft(x2, dim=dim)

        # Invert the mask for the second batch (x2)
        m = torch.bitwise_not(m)
        freal2 = x2_f.real.masked_fill(m, 0)
        fimag2 = x2_f.imag.masked_fill(m, 0)

        # Combine the real and imaginary parts from both batches
        freal += freal2
        fimag += fimag2

        # Reconstruct the complex FFT result
        x_f = torch.complex(freal, fimag)

        # Perform inverse FFT and take the absolute value
        x = torch.abs(torch.fft.ifft(x_f, dim=dim))
        
        return x


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
            h = self.target_encoder(x_aug, self.chans_id.to(x_aug), need_mask=False)
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
    
    def forward(self, x):
        x_aug = x.clone()
        x_aug = self.fft_aug(x_aug, dim=-1)
        aug_latent = self.forward_target(x_aug)
        z_con, z_rec, mask = self.forward_context(x)
        latent, latent_aug = self.forward_con(z_con, aug_latent)
        # loss1 = self.normal_contra_loss(latent, latent_aug)
        loss1 = self.normal_contra_loss(latent, latent_aug)
        loss2 = self.forward_loss(x, z_rec, mask)
        if self.USE_LOSS_A:
            loss  = self.awl(loss1, loss2)
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
    def get_config(embed_dim=256, embed_num=1, depth=[10,4,10], num_heads=4): # 23977441
    
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
            'decoder': {
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
    result = model(x)
    print(result['loss'])
    print(result['loss1'])
    print(result['loss2'])