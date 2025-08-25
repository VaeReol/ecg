class EEGTransformer(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        img_size=(12, 2250),
        patch_size=75,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,# PatchNormEmbed
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num
        
        self.num_heads = num_heads
        
        # --
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches # (58, 16)
        # --
        
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                is_causal=False, use_rope= False, return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
            
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            assert ch in CHANNEL_DICT
            chan_ids.append(CHANNEL_DICT[ch])
        return torch.tensor(chan_ids).unsqueeze_(0).long()
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        # x.shape B, C, T
        # mask_x.shape mN, mC
        # mask_t.shape mN
        
        # -- patchify x
        x = self.patch_embed(x) # (bs, T, channel, D)
        B, N, C, D = x.shape
        
        assert N==self.num_patches[1] and C==self.num_patches[0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
        
        if chan_ids is None:
            chan_ids = torch.arange(0,C)     
        chan_ids = chan_ids.to(x) # 使得channel_id和x具有相同的设备和数据类型
        
        # -- add channels positional embedding to x
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0) # (1,C) -> (1,1,C,D)
        
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)# (bs, mn*N, mc*C, D)
            B, N, C, D = x.shape
            
        
        x = x.flatten(0, 1) # B*N, C, D
        
        # -- concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x,summary_token], dim=1)  # B*N, C+embed_num, D
        
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B*N, C+embed_num, D
            if blk.return_attention==True: return x

        x = x[:, -summary_token.shape[1]:, :] # B*N, embed_num, D
        
        if self.norm is not None:
            x = self.norm(x) 

        
        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        # -- reshape back
            
        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)# B, mN, D        
        
        x = x.reshape((B, N, self.embed_num, -1)) # B, N, embed_num, D
        
        return x