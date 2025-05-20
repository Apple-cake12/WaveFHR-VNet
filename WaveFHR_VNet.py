from WaveFHR_VNet_parts import *




class ViTLayer1D(nn.Module):
    def __init__(self, dim=1024, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [B, C, L] -> [L, B, C]
        x = x.permute(2, 0, 1)

        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        # Back to original shape [L, B, C] -> [B, C, L]
        x = x.permute(1, 2, 0)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,wavelet, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channels = 64
        self.inc = (DoubleConv(n_channels, self.channels))
        self.down1 = (Down(self.channels, self.channels*2,wavelet))
        self.down2 = (Down(self.channels*2, self.channels*4,wavelet))
        self.down3 = (Down(self.channels*4, self.channels*8,wavelet))
        self.down4 = (Down(self.channels*8, self.channels*16,wavelet))
        self.vit = ViTLayer1D(dim=self.channels*16)

        self.CBAM_1 = CBAM(self.channels)
        self.CBAM_2 = CBAM(self.channels * 2)
        self.CBAM_3 = CBAM(self.channels * 4)
        self.CBAM_4 = CBAM(self.channels * 8)

        self.up1 = (Up(self.channels*16, self.channels*8,wavelet ))
        self.up2 = (Up(self.channels*8, self.channels*4 ,wavelet))
        self.up3 = (Up(self.channels*4, self.channels*2 ,wavelet))
        self.up4 = (Up(self.channels*2, self.channels,wavelet))
        self.outc = (OutConv(self.channels, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x1_down, x1_detail = self.down1(x1)
        x1_detail = self.CBAM_1(x1_detail)

        x2_down, x2_detail = self.down2(x1_down)
        x2_detail = self.CBAM_2(x2_detail)

        x3_down, x3_detail = self.down3(x2_down)
        x3_detail = self.CBAM_3(x3_detail)

        x4_down, x4_detail = self.down4(x3_down)
        x4_detail = self.CBAM_4(x4_detail)

        x5 = self.vit(x4_down)

        x = self.up1(x5, x4_detail)
        x = self.up2(x, x3_detail)
        x = self.up3(x, x2_detail)
        x = self.up4(x, x1_detail)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)