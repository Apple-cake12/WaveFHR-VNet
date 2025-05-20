""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,wavelet):
        super().__init__()
        self.dwt = DWT_1D(wavelet)
        self.conv = DoubleConv(in_channels, out_channels)
        self.enhanced_block = Interaction_Block(in_channels,in_channels)
    def forward(self, x):
        approx, detail = self.dwt(x)
        H_detail = self.enhanced_block(approx, detail)
        out = self.conv(approx)
        return out, H_detail


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,wavelet):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.Idwt = IDWT_1D(wavelet)
        self.conv = DoubleConv(out_channels, out_channels)
        self.conv_channel = nn.Conv1d(in_channels, out_channels,kernel_size=1)
    def forward(self, x_down, x_detail):
        x_down = self.conv_channel(x_down)
        recur_out = self.Idwt(x_down,x_detail)
        out = self.conv(recur_out)

        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#离散小波变换 来代替下采样
class DWT_1D(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        #初始化小波滤波器（分解阶段）
        self.coeffs = pywt.Wavelet(self.wavelet).filter_bank
        self.dec_lo = torch.Tensor(self.coeffs[0]).view(1,1,-1)#低通滤波器
        self.dec_hi = torch.Tensor(self.coeffs[1]).view(1,1,-1)#高通滤波器
        self.register_buffer('dec_lo_filt',self.dec_lo) #注册为Buffer，支持设备切换
        self.register_buffer('dec_hi_filt', self.dec_hi)

    def forward(self,x):
        pad = len(self.coeffs[0])//2 - 1
        #低通滤波+下采样（近似系数）  repeat(x.size(1),1,1 代表每个通道都有自己的滤波器（固定的）
        approx = F.conv1d(x,self.dec_lo_filt.repeat(x.size(1),1,1),
                          stride=2,padding=pad, groups=x.size(1))
        #高通滤波+下采样（细节系数）
        detail = F.conv1d(x, self.dec_hi_filt.repeat(x.size(1),1,1),
                          stride=2,padding=pad, groups=x.size(1))
        return approx,detail #输出长度减半 各通道独立处理

#逆小波变换 来代替上采样恢复信息
class IDWT_1D(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        #初始化小波滤波器（重构阶段）
        self.coeffs = pywt.Wavelet(wavelet).filter_bank
        self.rec_lo = torch.Tensor(self.coeffs[2]).view(1,1,-1) #低通重构滤波器
        self.rec_hi = torch.Tensor(self.coeffs[3]).view(1,1,-1) #高通重构滤波器
        self.register_buffer('rec_lo_filt', self.rec_lo)
        self.register_buffer('rec_hi_filt', self.rec_hi)

    def forward(self, approx, detail):#B C N
        # 填充滤波器长度相关的边界
        pad = (self.rec_lo_filt.size(-1)-1) // 2
        # 通过卷积重构信号
        x_lo = F.conv_transpose1d(approx, self.rec_lo_filt.repeat(approx.size(1), 1, 1),
                        stride=2,padding=pad, groups=approx.size(1))
        x_hi = F.conv_transpose1d(detail, self.rec_hi_filt.repeat(detail.size(1), 1, 1),
                        stride=2,padding=pad,groups=detail.size(1))
        return x_lo + x_hi  # 合并低频和高频分量

#信息交流器
class Interaction_Block(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel * 2 , out_channels=in_channel * 2, kernel_size=31, padding=15),
            nn.BatchNorm1d(in_channel*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=in_channel * 2 , out_channels=in_channel * 2, kernel_size=31, padding=15),
            nn.BatchNorm1d(in_channel * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=in_channel * 2 , out_channels= out_channel, kernel_size=31, padding=15),
            nn.Sigmoid()#掩码矩阵
        )

    def forward(self,x_approx,x_detail):
        x_cat = torch.cat((x_approx,x_detail), dim=1)
        #生成掩码矩阵
        x_mask = self.Conv(x_cat)
        #抑制噪声 增强高频信息
        H_detail = x_detail * x_mask
        #残差结构
        H_detail = H_detail + x_detail
        return H_detail


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=31, stride=1, padding=15)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv1d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out