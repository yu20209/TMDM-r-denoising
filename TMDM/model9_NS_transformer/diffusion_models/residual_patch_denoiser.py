import   进口 torch
import torch.nn as nn   进口火炬。Nn as   作为 Nn


class ResidualPatchDenoiser(nn.Module):
    """   """
    SimDiff-style patch    """Transformer denoiser for TMDM-r residual diffusion.

    This module predicts diffusion noise eps for residual r_t.该模块预测残余r_t的扩散噪声eps。

    Inputs:   输入:
        x:       historical input series, [B, seq_len, C]x：历史输入序列，[B, seq_len, C]
        y_base:  base forecast from NS-Transformer, [B, pred_len, C]y_base：来自NS-Transformer的基本预测，[B, pred_len, C]
        r_t:     noisy residual at diffusion step t, [B, pred_len, C]r_t：扩散步骤t的噪声残差，[B, pred_len, C]
        t:       diffusion step, [B]t：扩散阶跃，[B]

    Output:   输出:
        eps:     predicted noise, [B, pred_len, C]eps：预测噪声，[B, pred_len, C]

    Design:   设计:
        - Channel-independent: reshape [B, L, C] -> [B*C, L]-通道无关：重塑[B， L, C] -> [B*C, L]
        - Patch tokenization over temporal dimension
        - y_base patch embedding used as condition
        - diffusion timestep embedding added to tokens
        - Transformer encoder predicts noise patches
    """   """

    def __init__(self, args, num_timesteps):
        super().__init__()   .__init__ super () ()

        self.patch_len = getattr(args, "patch_len", 16)自我。Patch_len = getattr（args, " Patch_len ", 16）
        self.stride = getattr(args, "stride", 8)
        self.d_model = getattr(args, "d_model", 256)self .d_model = getattr（args，“d_model”，256）

        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.c_out = args.c_out   Self.c_out = args.c_out

        self.time_embed = nn.Embedding(num_timesteps + 1, self.d_model)自我。Time_embed = nn。嵌入（num_timesteps 1, self.d_model）

        # Shared patch embedding for residual and base forecast.#共享补丁嵌入残差和基础预测。
        # This keeps the first ablation simple and parameter-efficient.这使得第一次烧蚀简单且参数高效。
        self.patch_embed = nn.Linear(self.patch_len, self.d_model)自我。patch_embed = nn.Linear(self。patch_len self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(Encoder_layer = nn。TransformerEncoderLayer (
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,dim_feedforward =自我。D_model * 4；
            dropout=getattr(args, "dropout", 0.1),dropout=getattr(args, dropout, 0.1)，
            activation="gelu",   activation="gelu",
            batch_first=   """True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.head = nn.Linear(self.d_model, self.patch_len)自我。head = nn.Linear（self.）d_model self.patch_len)

    def patchify(self, x):   defpatchify (self, x)：
        """   """
        x: [B, L]   """
        return: [B, N, patch_len]return: [B， N, patch_len]
        """   """   """
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def unpatchify(self, patches, length):Def unpatchify(self, patches, length)：
        """   """
        Reconstruct sequence from   """ overlapping patches by averaging overlaps.

        patches: [B, N, patch_len]
        length:  output sequence length
        return:  [B, length]
        """   """
        B, N, P = patches.shape   """

        out = torch.zeros(B, length, device=patches.device, dtype=patches.dtype)   """Out =火炬。0 （B, length, device=patches.device, dtype=patches.dtype）
        count = torch.zeros(B, length, device=patches.device, dtype=patches.dtype)计数=火炬。0 （B, length, device=patches.device, dtype=patches.dtype）

        for i in range(N):   对于i在(N)范围内：
            start = i * self.strideStart = 1 * self.stride
            end = start + self.patch_len结束=开始self.patch_len
            out[:, start:end] += patches[:, i, :]Out [:, start:end] = patches[:, i,:]
            count[:, start:end] += 1.0Count [:, start:end] = 1.0

        return out / (count + 1e-6)退回/（第1e-6项）

    def forward(self, x, y_base, r_t, t):Def forward(self, x, y_base, r_t, t)：
        """   """
        x:       [B, seq_len, C]
        y_base:  [B, pred_len, C]   """
        r_t:     [B, pred_len, C]
        t:       [B]
        """   """
        B, L, C = r_t.shape

        # Channel-independent reshape:#频道无关的重塑：
        # [B, L, C] -> [B, C, L] -> [B*C, L][B， C, C] -> [B， C, L] -> [B*C, L]
        r = r_t.permute(0, 2, 1).reshape(B * C, L)R = r_t。permute(0,2,1). shaping （B * C， L）
        base = y_base.permute(0, 2, 1).reshape(B * C, L)Base = y_base。permute(0,2,1). shaping （B * C， L）

        # Patchify residual and base forecast.#修补剩余和基础预测。
        r_patches = self.patchify(r)          # [B*C, N, patch_len]
        base_patches = self.patchify(base)    # [B*C, N, patch_len]Base_patches = self。patchify(base) # [B*C， N, patch_len]

        r_tokens = self.patch_embed(r_patches)R_tokens = self.patch_embed（r_patches）
        base_tokens = self.patch_embed(base_patches)Base_tokens = self.patch_embed（base_patches）

        # Repeat timestep for each channel.
        # t: [B] -> [B*C] -> [B*C, 1, d_model]
        t_tokens = self.time_embed(t).repeat_interleave(C, dim=0).unsqueeze(1)

        # SimDiff-style simple conditioning:
        # noisy residual token + base forecast token + timestep token
        tokens = r_tokens + base_tokens + t_tokens

        hidden = self.encoder(tokens)

        eps_patches = self.head(hidden)       # [B*C, N, patch_len]
        eps = self.unpatchify(eps_patches, L) # [B*C, L]

        eps = eps.reshape(B, C, L).permute(0, 2, 1)  # [B, L, C]
        return eps
