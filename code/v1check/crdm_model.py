#!/usr/bin/env python3
"""
OTFS扩散模型网络架构

基于UNet的时延-多普勒域信道估计模型
支持条件生成（速度+SNR）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码（用于时间步嵌入）"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ComplexConv2d(nn.Module):
    """复数卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复数卷积: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        
        Args:
            x_real: 实部 [batch, channels, H, W]
            x_imag: 虚部 [batch, channels, H, W]
        
        Returns:
            out_real, out_imag: 输出的实部和虚部
        """
        # 实部: ac - bd
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        # 虚部: ad + bc
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """复数批归一化"""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bn_real(x_real), self.bn_imag(x_imag)


class ResBlock(nn.Module):
    """残差块（支持复数）"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        
        # 时间步嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 第一个卷积块
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        
        # 第二个卷积块
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = ComplexConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.residual_conv = None
        
        # 注意力（可选）
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, 
                time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_real, x_imag: 输入的实部和虚部 [batch, channels, H, W]
            time_emb: 时间步嵌入 [batch, time_emb_dim]
        """
        # 保存输入用于残差连接
        residual_real, residual_imag = x_real, x_imag
        
        # 第一个卷积
        h_real, h_imag = self.conv1(x_real, x_imag)
        h_real, h_imag = self.bn1(h_real, h_imag)
        h_real, h_imag = F.silu(h_real), F.silu(h_imag)
        
        # 添加时间步信息
        time_emb_proj = self.time_mlp(time_emb)[:, :, None, None]
        h_real = h_real + time_emb_proj
        h_imag = h_imag + time_emb_proj
        
        # 第二个卷积
        h_real, h_imag = self.conv2(h_real, h_imag)
        h_real, h_imag = self.bn2(h_real, h_imag)
        h_real = self.dropout(h_real)
        h_imag = self.dropout(h_imag)
        
        # 残差连接
        if self.residual_conv is not None:
            residual_real, residual_imag = self.residual_conv(residual_real, residual_imag)
        
        out_real = h_real + residual_real
        out_imag = h_imag + residual_imag
        out_real, out_imag = F.silu(out_real), F.silu(out_imag)
        
        # 自注意力（可选）
        if self.use_attention:
            out_real, out_imag = self.attention(out_real, out_imag)
        
        return out_real, out_imag


class SelfAttention(nn.Module):
    """自注意力层（支持复数）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Query, Key, Value投影（实部和虚部分别处理）
        self.q_real = nn.Conv2d(channels, channels, 1)
        self.k_real = nn.Conv2d(channels, channels, 1)
        self.v_real = nn.Conv2d(channels, channels, 1)
        
        self.q_imag = nn.Conv2d(channels, channels, 1)
        self.k_imag = nn.Conv2d(channels, channels, 1)
        self.v_imag = nn.Conv2d(channels, channels, 1)
        
        # 输出投影
        self.proj_out_real = nn.Conv2d(channels, channels, 1)
        self.proj_out_imag = nn.Conv2d(channels, channels, 1)
        
        self.scale = channels ** -0.5
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_real.shape
        
        # 计算Q, K, V（实部）
        q_r = self.q_real(x_real).view(B, C, H*W).permute(0, 2, 1)
        k_r = self.k_real(x_real).view(B, C, H*W)
        v_r = self.v_real(x_real).view(B, C, H*W).permute(0, 2, 1)
        
        # 注意力权重
        attn = torch.bmm(q_r, k_r) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out_real = torch.bmm(attn, v_r).permute(0, 2, 1).view(B, C, H, W)
        out_real = self.proj_out_real(out_real)
        
        # 虚部：用自己的Q,K计算独立注意力权重（修复bug：原代码q_i,k_i从未被使用）
        q_i = self.q_imag(x_imag).view(B, C, H*W).permute(0, 2, 1)
        k_i = self.k_imag(x_imag).view(B, C, H*W)
        v_i = self.v_imag(x_imag).view(B, C, H*W).permute(0, 2, 1)
        
        attn_i = torch.bmm(q_i, k_i) * self.scale
        attn_i = F.softmax(attn_i, dim=-1)
        out_imag = torch.bmm(attn_i, v_i).permute(0, 2, 1).view(B, C, H, W)
        out_imag = self.proj_out_imag(out_imag)
        
        # 残差连接
        return x_real + out_real, x_imag + out_imag


class Downsample(nn.Module):
    """下采样层（复数版本）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv_real = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.conv_imag = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_real(x_real), self.conv_imag(x_imag)


class Upsample(nn.Module):
    """上采样层（复数版本）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv_real = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_imag = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_real = F.interpolate(x_real, scale_factor=2, mode='nearest')
        x_imag = F.interpolate(x_imag, scale_factor=2, mode='nearest')
        return self.conv_real(x_real), self.conv_imag(x_imag)


# ========== 实数UNet组件（消融实验Baseline用） ==========

class RealResBlock(nn.Module):
    """实数残差块（消融实验用）"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = None
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = RealSelfAttention(out_channels)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.silu(h)
        
        time_emb_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb_proj
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.dropout(h)
        
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        out = F.silu(h + residual)
        
        if self.use_attention:
            out = self.attention(out)
        
        return out


class RealSelfAttention(nn.Module):
    """实数自注意力层（消融实验用）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, C, H*W).permute(0, 2, 1)
        k = self.k(x).view(B, C, H*W)
        v = self.v(x).view(B, C, H*W).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj_out(out)


class RealDownsample(nn.Module):
    """实数下采样层（消融实验用）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RealUpsample(nn.Module):
    """实数上采样层（消融实验用）"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class OTFSDiffusionModel(nn.Module):
    """OTFS扩散模型（UNet架构）
    
    消融实验支持:
    - use_complex_unet: True=复数UNet, False=实数UNet
    - use_conditioning: True=速度-SNR条件嵌入, False=无条件
    """
    
    def __init__(self, 
                 in_channels: int = 8,  # 4 rx × 2 (实部+虚部)
                 model_channels: int = 128,
                 channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (8, 16),
                 dropout: float = 0.1,
                 time_embed_dim: int = 512,
                 num_classes: int = 50,
                 use_class_condition: bool = True,
                 use_explicit_snr: bool = True,
                 use_sinusoidal_snr_embed: bool = True,
                 # === 消融实验开关 ===
                 use_complex_unet: bool = True,
                 use_conditioning: bool = True,
                 use_speed_cond: bool = True,
                 use_snr_cond: bool = True,
                 speed_norm_min: float = 10.0,
                 speed_norm_max: float = 120.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.use_class_condition = use_class_condition
        self.use_explicit_snr = use_explicit_snr
        self.use_sinusoidal_snr_embed = use_sinusoidal_snr_embed
        # 消融实验开关
        self.use_complex_unet = use_complex_unet
        self.use_conditioning = use_conditioning
        # 【审稿rev1】单独条件消融开关（默认都开；嵌入层照常创建，参数量不变）
        self.use_speed_cond = use_speed_cond
        self.use_snr_cond = use_snr_cond
        # 【R1-4】速度归一化区间：默认 10/120（= 原 (v−10)/110，逐字节一致）；Prong B 可置 5/200
        self.speed_norm_min = speed_norm_min
        self.speed_norm_max = speed_norm_max

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 条件嵌入（速度+SNR） - 只在use_conditioning=True时创建
        if use_conditioning and use_class_condition:
            if use_explicit_snr:
                # 【方案2】显式SNR嵌入：速度和SNR分离编码
                self.speed_embed = nn.Sequential(
                    nn.Linear(1, time_embed_dim // 2),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim // 2, time_embed_dim // 2),
                )
                if use_sinusoidal_snr_embed:
                    # 【方案C】SNR正弦位置编码嵌入
                    self.snr_sinusoidal = SinusoidalPositionEmbeddings(time_embed_dim // 2)
                    self.snr_embed = nn.Sequential(
                        nn.Linear(time_embed_dim // 2, time_embed_dim // 2),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim // 2, time_embed_dim // 2),
                    )
                else:
                    self.snr_embed = nn.Sequential(
                        nn.Linear(1, time_embed_dim // 2),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim // 2, time_embed_dim // 2),
                    )
            else:
                self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # 输入投影 - 根据是否使用复数UNet选择不同结构
        reshaped_channels = (in_channels // 2) * 4  # 4 * 4 = 16
        
        if use_complex_unet:
            # 复数UNet: 实部和虚部分别处理
            self.input_conv_real = nn.Conv2d(reshaped_channels, model_channels, 3, padding=1)
            self.input_conv_imag = nn.Conv2d(reshaped_channels, model_channels, 3, padding=1)
        else:
            # 实数UNet: 实部和虚部拼接为32通道
            self.input_conv = nn.Conv2d(reshaped_channels * 2, model_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            
            for _ in range(num_res_blocks):
                use_attn = (now_channels in attention_resolutions)
                if use_complex_unet:
                    self.down_blocks.append(
                        ResBlock(now_channels, out_channels, time_embed_dim, dropout, use_attn)
                    )
                else:
                    self.down_blocks.append(
                        RealResBlock(now_channels, out_channels, time_embed_dim, dropout, use_attn)
                    )
                now_channels = out_channels
                channels.append(now_channels)
            
            if level != len(channel_mult) - 1:
                if use_complex_unet:
                    self.down_blocks.append(Downsample(now_channels))
                else:
                    self.down_blocks.append(RealDownsample(now_channels))
                channels.append(now_channels)
        
        # 中间块
        if use_complex_unet:
            self.middle_block1 = ResBlock(now_channels, now_channels, time_embed_dim, dropout, use_attention=True)
            self.middle_block2 = ResBlock(now_channels, now_channels, time_embed_dim, dropout, use_attention=False)
        else:
            self.middle_block1 = RealResBlock(now_channels, now_channels, time_embed_dim, dropout, use_attention=True)
            self.middle_block2 = RealResBlock(now_channels, now_channels, time_embed_dim, dropout, use_attention=False)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                ich = channels.pop()
                use_attn = (ich in attention_resolutions)
                if use_complex_unet:
                    self.up_blocks.append(
                        ResBlock(now_channels + ich, out_channels, time_embed_dim, dropout, use_attn)
                    )
                else:
                    self.up_blocks.append(
                        RealResBlock(now_channels + ich, out_channels, time_embed_dim, dropout, use_attn)
                    )
                now_channels = out_channels
            
            if level != 0:
                if use_complex_unet:
                    self.up_blocks.append(Upsample(now_channels))
                else:
                    self.up_blocks.append(RealUpsample(now_channels))
        
        # 输出投影
        reshaped_out_channels = (in_channels // 2) * 4  # 4 * 4 = 16
        if use_complex_unet:
            self.output_conv_real = nn.Conv2d(model_channels, reshaped_out_channels, 3, padding=1)
            self.output_conv_imag = nn.Conv2d(model_channels, reshaped_out_channels, 3, padding=1)
        else:
            self.output_conv = nn.Conv2d(model_channels, reshaped_out_channels * 2, 3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                class_labels: Optional[torch.Tensor] = None,
                speed_kmh: Optional[torch.Tensor] = None,
                snr_db: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, 8, 4, 64, 128] 实数格式（4实部+4虚部，时延-多普勒域）
            timesteps: 时间步 [batch]
            class_labels: 类别标签 [batch] (可选，仅use_explicit_snr=False时使用)
            speed_kmh: 速度 [batch] (km/h，仅use_explicit_snr=True时使用)
            snr_db: SNR [batch] (dB，仅use_explicit_snr=True时使用)
        
        Returns:
            out: 输出 [batch, 8, 4, 64, 128] 预测的噪声或干净信号
        """
        batch_size, n_channels, n_tx, n_doppler, n_delay = x.shape
        n_rx = n_channels // 2  # = 4
        
        # 分离实部和虚部：[batch, 8, 4, 64, 128] -> 两个 [batch, 4, 4, 64, 128]
        x_real = x[:, :n_rx, :, :, :]
        x_imag = x[:, n_rx:, :, :, :]
        
        # 重塑为 [batch, n_rx*n_tx, n_doppler, n_delay]
        x_real = x_real.reshape(batch_size, n_rx * n_tx, n_doppler, n_delay)
        x_imag = x_imag.reshape(batch_size, n_rx * n_tx, n_doppler, n_delay)
        
        # 时间步嵌入
        time_emb = self.time_embed(timesteps)
        
        # 条件嵌入 - 只在use_conditioning=True时添加
        if self.use_conditioning and self.use_class_condition:
            if self.use_explicit_snr:
                if speed_kmh is not None and snr_db is not None:
                    speed_norm = (speed_kmh - self.speed_norm_min) / (self.speed_norm_max - self.speed_norm_min)
                    if getattr(self, 'speed_clamp', True):   # 默认True=截断到[0,1](原行为);置False则不截断(好奇版)
                        speed_norm = torch.clamp(speed_norm, 0.0, 1.0)
                    speed_norm = speed_norm.unsqueeze(-1).to(dtype=torch.float32)
                    speed_emb = self.speed_embed(speed_norm)
                    
                    if self.use_sinusoidal_snr_embed:
                        snr_scaled = (snr_db / 10.0 * 1000.0).to(dtype=torch.float32)
                        snr_scaled = torch.clamp(snr_scaled, 0.0, 1000.0)
                        snr_sinusoidal = self.snr_sinusoidal(snr_scaled)
                        snr_emb = self.snr_embed(snr_sinusoidal)
                    else:
                        snr_norm = snr_db / 10.0
                        snr_norm = torch.clamp(snr_norm, 0.0, 1.0)
                        snr_norm = snr_norm.unsqueeze(-1).to(dtype=torch.float32)
                        snr_emb = self.snr_embed(snr_norm)

                    # 【审稿rev1】单独条件消融：置零被关闭那一路的嵌入向量（另一路照常，参数量不变）
                    if not self.use_speed_cond:
                        speed_emb = torch.zeros_like(speed_emb)
                    if not self.use_snr_cond:
                        snr_emb = torch.zeros_like(snr_emb)
                    cond_emb = torch.cat([speed_emb, snr_emb], dim=-1)
                    time_emb = time_emb + cond_emb
            else:
                if class_labels is not None:
                    class_emb = self.class_embed(class_labels)
                    time_emb = time_emb + class_emb
        
        # ========== 复数UNet前向传播 ==========
        if self.use_complex_unet:
            h_real = self.input_conv_real(x_real)
            h_imag = self.input_conv_imag(x_imag)
            
            skip_connections_real = [h_real]
            skip_connections_imag = [h_imag]
            
            for module in self.down_blocks:
                if isinstance(module, ResBlock):
                    h_real, h_imag = module(h_real, h_imag, time_emb)
                else:  # Downsample
                    h_real, h_imag = module(h_real, h_imag)
                skip_connections_real.append(h_real)
                skip_connections_imag.append(h_imag)
            
            h_real, h_imag = self.middle_block1(h_real, h_imag, time_emb)
            h_real, h_imag = self.middle_block2(h_real, h_imag, time_emb)
            
            for module in self.up_blocks:
                if isinstance(module, ResBlock):
                    skip_real = skip_connections_real.pop()
                    skip_imag = skip_connections_imag.pop()
                    h_real = torch.cat([h_real, skip_real], dim=1)
                    h_imag = torch.cat([h_imag, skip_imag], dim=1)
                    h_real, h_imag = module(h_real, h_imag, time_emb)
                else:  # Upsample
                    h_real, h_imag = module(h_real, h_imag)
            
            out_real = self.output_conv_real(h_real)
            out_imag = self.output_conv_imag(h_imag)
            
            out_real = out_real.view(batch_size, n_rx, n_tx, n_doppler, n_delay)
            out_imag = out_imag.view(batch_size, n_rx, n_tx, n_doppler, n_delay)
            out = torch.cat([out_real, out_imag], dim=1)
        
        # ========== 实数UNet前向传播 ==========
        else:
            # 实部和虚部拼接为32通道
            h = torch.cat([x_real, x_imag], dim=1)  # [batch, 32, 64, 128]
            h = self.input_conv(h)
            
            skip_connections = [h]
            
            for module in self.down_blocks:
                if isinstance(module, RealResBlock):
                    h = module(h, time_emb)
                else:  # RealDownsample
                    h = module(h)
                skip_connections.append(h)
            
            h = self.middle_block1(h, time_emb)
            h = self.middle_block2(h, time_emb)
            
            for module in self.up_blocks:
                if isinstance(module, RealResBlock):
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = module(h, time_emb)
                else:  # RealUpsample
                    h = module(h)
            
            out = self.output_conv(h)  # [batch, 32, 64, 128]
            
            # 拆分为实部和虚部，重塑为输出格式
            out_real = out[:, :16, :, :]
            out_imag = out[:, 16:, :, :]
            out_real = out_real.view(batch_size, n_rx, n_tx, n_doppler, n_delay)
            out_imag = out_imag.view(batch_size, n_rx, n_tx, n_doppler, n_delay)
            out = torch.cat([out_real, out_imag], dim=1)
        
        return out


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*80)
    print("OTFS扩散模型测试")
    print("="*80)
    
    # 创建模型
    model = OTFSDiffusionModel(
        in_channels=8,  # 4x4 MIMO -> 16 channels, 分实部虚部
        model_channels=64,  # 减小以便测试
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(16,),
        use_class_condition=True,
        num_classes=50
    )
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 4, 4, 64, 128, dtype=torch.cfloat)
    timesteps = torch.randint(0, 1000, (batch_size,))
    class_labels = torch.randint(0, 50, (batch_size,))
    
    print(f"\n输入形状: {x.shape}")
    print(f"时间步: {timesteps}")
    print(f"类别标签: {class_labels}")
    
    # 前向传播
    with torch.no_grad():
        out = model(x, timesteps, class_labels)
    
    print(f"\n输出形状: {out.shape}")
    print(f"输出数据类型: {out.dtype}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n✓ 模型测试完成！")
