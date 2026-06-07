"""
DeepThaw Loss Functions
=======================

저장 위치: uvcgan2/torch/deepthaw_loss.py

사용법:
    from uvcgan2.torch.deepthaw_loss import SelfChallengingLoss, ProgressiveScheduler
"""

import math
import torch
import torch.nn.functional as F


class SelfChallengingLoss:
    """
    Self-Challenging Loss.
    
    Discriminator가 "가짜 같다"고 하는 영역에 더 높은 weight 부여.
    Generator가 자기 약점을 집중 보완하게 만듦.
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        challenge_weight: float = 5.0,
        temperature: float = 1.0,
    ):
        self.base_weight = base_weight
        self.challenge_weight = challenge_weight
        self.temperature = temperature
    
    def compute_weight_map(self, discriminator, fake_image):
        """Discriminator output으로 difficulty weight map 계산."""
        B, C, H, W = fake_image.shape
        
        with torch.no_grad():
            d_output = discriminator(fake_image)
            
            # Multi-scale discriminator면 마지막 것 사용
            if isinstance(d_output, (list, tuple)):
                d_output = d_output[-1]
            
            # Sigmoid로 0~1 확률
            confidence = torch.sigmoid(d_output / self.temperature)
            
            # Difficulty = 1 - confidence
            difficulty = 1.0 - confidence
            
            # 이미지 크기로 resize
            if difficulty.shape[2:] != (H, W):
                difficulty = F.interpolate(
                    difficulty, size=(H, W),
                    mode='bilinear', align_corners=False
                )
            
            # Channel 맞추기
            if difficulty.shape[1] != 1:
                difficulty = difficulty.mean(dim=1, keepdim=True)
            
            # Weight 계산
            weight = self.base_weight + self.challenge_weight * difficulty
            weight = weight / (weight.mean() + 1e-8) * self.base_weight
        
        return weight
    
    def __call__(self, fake_image, real_image, discriminator):
        """Weighted L1 loss 계산."""
        weight_map = self.compute_weight_map(discriminator, fake_image)
        l1_diff = torch.abs(fake_image - real_image)
        weighted_loss = (l1_diff * weight_map).mean()
        return weighted_loss, weight_map


class ProgressiveScheduler:
    """Artifact strength를 epoch에 따라 조절."""
    
    def __init__(
        self,
        warmup_epochs: int = 10,
        rampup_epochs: int = 50,
        max_strength: float = 0.7,
    ):
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.max_strength = max_strength
    
    def get_strength(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        
        progress = (epoch - self.warmup_epochs) / self.rampup_epochs
        progress = min(1.0, max(0.0, progress))
        
        # Cosine schedule
        strength = 0.5 * (1 - math.cos(math.pi * progress))
        return strength * self.max_strength
