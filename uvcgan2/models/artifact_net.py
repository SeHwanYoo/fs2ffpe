"""
DeepThaw: ArtifactNet
=====================

저장 위치: uvcgan2/models/artifact_net.py

Learnable artifact generator for Progressive Artifact Injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtifactNet(nn.Module):
    """
    Learnable artifact generator.
    
    Ice crystal, blur, stain variation 등의 artifact를 학습해서 생성.
    Generator를 adversarial하게 훈련시키는 역할.
    """
    
    def __init__(
        self, 
        in_channels: int = 3,
        hidden_channels: int = 64,
        z_dim: int = 128,
    ):
        super().__init__()
        
        self.z_dim = z_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Noise projection
        self.noise_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_channels * 4 * 16 * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # Output heads
        self.to_artifact_map = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
            nn.Sigmoid(),  # 0~1 범위
        )
        
        self.to_artifact_color = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, 3, 1, 1),
            nn.Tanh(),  # -1~1 범위
        )
        
    def forward(self, image, z=None, strength=1.0):
        """
        Args:
            image: (B, 3, H, W) 입력 이미지
            z: (B, z_dim) noise (None이면 random)
            strength: artifact 강도 (0~1)
            
        Returns:
            corrupted: (B, 3, H, W) artifact 추가된 이미지
            artifact_map: (B, 1, H, W) artifact 위치
        """
        B, C, H, W = image.shape
        device = image.device
        
        if z is None:
            z = torch.randn(B, self.z_dim, device=device)
        
        # Encode
        feat = self.encoder(image)
        
        # Noise injection
        z_spatial = self.noise_proj(z).view(B, -1, 16, 16)
        z_spatial = F.interpolate(
            z_spatial, size=feat.shape[2:], 
            mode='bilinear', align_corners=False
        )
        feat = torch.cat([feat, z_spatial], dim=1)
        
        # Decode
        feat = self.decoder(feat)
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Outputs
        artifact_map = self.to_artifact_map(feat) * strength
        artifact_color = self.to_artifact_color(feat)
        
        # Blend: 원본과 artifact color를 artifact_map에 따라 섞음
        corrupted = image * (1 - artifact_map) + artifact_color * artifact_map
        corrupted = torch.clamp(corrupted, -1, 1)
        
        return corrupted, artifact_map
