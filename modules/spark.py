#nguồn tham khảo: https://github.com/keyu-tian/SparK/blob/main/pretrain/spark.py
from pprint import pformat
from typing import List

import torch
import torch.nn as nn
from torchvision.models import *
from modules.config import System_Config as cfg
import torch.nn.functional as F


class Encoder_EfficientNet(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_EfficientNet, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.model = self.backbone.features

    def forward(self, x):
        return self.model(x)

class Encoder_ResNet(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_ResNet, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

class Encoder_MobileNet(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_MobileNet, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        self.model = self.backbone.features

    def forward(self, x):
        return self.model(x)
    
class Encoder_DenseNet(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_DenseNet, self).__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)
        self.model = self.backbone.features

    def forward(self, x):
        return self.model(x)

class Encoder_GoogleNet(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_GoogleNet, self).__init__()
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        self.backbone = googlenet(weights=weights)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.conv3(x)
        x = self.backbone.maxpool2(x)
        x = self.backbone.inception3a(x)
        x = self.backbone.inception3b(x)
        x = self.backbone.maxpool3(x)
        x = self.backbone.inception4a(x)
        x = self.backbone.inception4b(x)
        x = self.backbone.inception4c(x)
        x = self.backbone.inception4d(x)
        x = self.backbone.inception4e(x)
        x = self.backbone.maxpool4(x)
        x = self.backbone.inception5a(x)
        x = self.backbone.inception5b(x)
        return x
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # Về 3 kênh màu (RGB)
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.decoder(x)
    
class SparK(nn.Module):
    def __init__(self,
                mask_ratio = cfg.SPARK_CONFIG["MASK_RATIO"],
                patch_size = cfg.SPARK_CONFIG["PATCH_SIZE"],
                decoder_channels = cfg.SPARK_CONFIG["DECODER_CHANNELS"],
                pretrained = cfg.PRETRAIN_CONFIG["pretrained"],
                densify_norm = 'bn',
                sbn=False):
        super().__init__()
        
        input_size = cfg.IMG_SIZE
        self.fmap_h, self.fmap_w = input_size // patch_size, input_size // patch_size
        self.mask_ratio = mask_ratio #tỉ lệ che
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        self.decoder_channels = decoder_channels

        self.backbone = cfg.PRETRAIN_CONFIG["BACKBONE"]
        if self.backbone == "EfficientNet":
            self.encoder = Encoder_EfficientNet(pretrained=pretrained)
            enc_channels = 1280
        elif self.backbone == "ResNet":
            self.encoder = Encoder_ResNet(pretrained=pretrained)
            enc_channels = 512
        elif self.backbone == "MobileNet":
            self.encoder = Encoder_MobileNet(pretrained=pretrained)
            enc_channels = 1280
        elif self.backbone == "DenseNet":
            self.encoder = Encoder_DenseNet(pretrained=pretrained)
            enc_channels = 1024
        elif self.backbone == "GoogleNet":
            self.encoder = Encoder_GoogleNet(pretrained=pretrained)
            enc_channels = 1024
        else:
            raise ValueError(
                f"'{self.backbone}'. Chỉ các backbone: EfficientNet, ResNet, DenseNet, MobileNet, GoogleNet"
            )

        self.sbn = sbn
        self.densify_norm_str = densify_norm.lower()
        # Projection layer đưa về 256 channels để khớp với decoder
        self.densify_proj = nn.Conv2d(enc_channels, self.decoder_channels, kernel_size=1)
        if densify_norm.lower() == 'bn':
            self.densify_norm = nn.BatchNorm2d(self.decoder_channels)
        else: 
            self.densify_norm = nn.GroupNorm(1, self.decoder_channels)
        # Mask token học được
        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_channels, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        #decoder khôi phục ảnh 
        self.decoder = Decoder(in_channels=self.decoder_channels, out_channels=3)

    def mask(self, B: int, device):
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, device=device).argsort(dim=1) #trộn ngẫu nhiên
        idx = idx[:, :self.len_keep]  # Lấy danh sách các ô không che
        
        # Tạo mask: True = Giữ lại, False = Bị che
        mask = torch.zeros(B, h * w, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=idx, value=True)
        
        # Trả về dạng float (1.0 và 0.0) kích thước (B, 1, h, w)
        return mask.view(B, 1, h, w).float()
    
    def forward(self, x):
        # Tạo mask và che ảnh gốc
        mask_7x7 = self.mask(x.shape[0], x.device) 
        mask_224 = F.interpolate(mask_7x7, size=cfg.IMG_SIZE, mode='nearest') # phóng to mask từ 7x7 lên kích thước ảnh gốc
        x_masked = x * mask_224
        
        f = self.encoder(x_masked)
        #Lấp đầy vùng trống
        f = self.densify_proj(f)
        f = self.densify_norm(f)
        # Lấp đầy: Nếu mask == 0, thay bằng mask_token
        f_densified = f * mask_7x7 + self.mask_token * (1 - mask_7x7)
        #Khôi phục ảnh bằng Decoder
        recon_img = self.decoder(f_densified)
        
        return recon_img