"""
DeepThaw CycleGAN - Progressive Artifact Injection + Self-Challenging
======================================================================

UVCGAN2의 CycleGAN을 확장하여 DeepThaw의 핵심 기능 추가:
1. Progressive Artifact Injection (PAI)
2. Self-Challenging Loss
3. Pathology-Aware Training

사용법:
    기존 uvcgan2/cgan/cyclegan.py의 CycleGAN 클래스 대신 이 클래스 사용
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import os
import sys

# 기존 UVCGAN2 import (경로는 실제 구조에 맞게 조정)
# from uvcgan2.cgan.cyclegan import CycleGAN as BaseCycleGAN

from ..models.artifact_net import ArtifactNet, ProgressiveArtifactScheduler
from ..losses.deepthaw_loss import SelfChallengingLoss, DeepThawLoss


class DeepThawCycleGAN:
    """
    DeepThaw CycleGAN Trainer.
    
    기존 UVCGAN2 CycleGAN에 다음 기능 추가:
    - Progressive Artifact Injection
    - Self-Challenging Loss
    
    이 클래스를 기존 CycleGAN 학습 루프에 통합하거나,
    backward_generators() 부분만 참고해서 수정하면 됨.
    """
    
    def __init__(
        self,
        # 기존 CycleGAN 컴포넌트들
        generator_ab: nn.Module,
        generator_ba: nn.Module,
        discriminator_a: nn.Module,
        discriminator_b: nn.Module,
        # DeepThaw 설정
        use_artifact_injection: bool = True,
        use_self_challenging: bool = True,
        artifact_warmup_epochs: int = 10,
        artifact_rampup_epochs: int = 50,
        artifact_max_strength: float = 0.7,
        challenge_weight: float = 5.0,
        # Loss weights
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lambda_artifact: float = 5.0,
        # Device
        device: str = 'cuda',
    ):
        self.device = device
        
        # Generators & Discriminators
        self.gen_ab = generator_ab.to(device)
        self.gen_ba = generator_ba.to(device)
        self.disc_a = discriminator_a.to(device)
        self.disc_b = discriminator_b.to(device)
        
        # DeepThaw 컴포넌트
        self.use_artifact_injection = use_artifact_injection
        self.use_self_challenging = use_self_challenging
        
        # Artifact Network
        if use_artifact_injection:
            self.artifact_net = ArtifactNet(
                in_channels=3,
                hidden_channels=64,
                z_dim=128,
            ).to(device)
            
            self.artifact_scheduler = ProgressiveArtifactScheduler(
                warmup_epochs=artifact_warmup_epochs,
                rampup_epochs=artifact_rampup_epochs,
                max_strength=artifact_max_strength,
                schedule_type='cosine',
            )
            
            # Artifact optimizer (별도로 학습)
            self.artifact_optimizer = torch.optim.Adam(
                self.artifact_net.parameters(),
                lr=1e-4,
                betas=(0.5, 0.999)
            )
        else:
            self.artifact_net = None
            self.artifact_scheduler = None
            
        # Self-Challenging Loss
        if use_self_challenging:
            self.self_challenging = SelfChallengingLoss(
                base_weight=1.0,
                challenge_weight=challenge_weight,
            )
        else:
            self.self_challenging = None
            
        # Loss weights
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_artifact = lambda_artifact
        
        # Standard losses
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Current epoch (외부에서 설정)
        self.current_epoch = 0
        
        # Logging
        self.loss_dict = {}
        
    def set_epoch(self, epoch: int):
        """현재 epoch 설정 (artifact strength 계산용)."""
        self.current_epoch = epoch
        
    def get_artifact_strength(self) -> float:
        """현재 epoch의 artifact strength 반환."""
        if self.artifact_scheduler is None:
            return 0.0
        return self.artifact_scheduler.get_strength(self.current_epoch)
    
    def forward_generators(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generator forward pass.
        
        Args:
            real_a: Domain A 이미지 (FS)
            real_b: Domain B 이미지 (FFPE)
            
        Returns:
            Dict containing all generated images
        """
        # Standard CycleGAN forward
        fake_b = self.gen_ab(real_a)  # FS → FFPE
        fake_a = self.gen_ba(real_b)  # FFPE → FS
        reco_a = self.gen_ba(fake_b)  # FS → FFPE → FS
        reco_b = self.gen_ab(fake_a)  # FFPE → FS → FFPE
        
        outputs = {
            'real_a': real_a,
            'real_b': real_b,
            'fake_a': fake_a,
            'fake_b': fake_b,
            'reco_a': reco_a,
            'reco_b': reco_b,
        }
        
        # Progressive Artifact Injection
        if self.use_artifact_injection and self.artifact_net is not None:
            strength = self.get_artifact_strength()
            
            if strength > 0:
                with torch.no_grad():
                    # Synthetic FS 생성
                    synthetic_fs = self.gen_ba(real_b).detach()
                
                # Artifact 주입
                corrupted_fs, artifact_map = self.artifact_net(
                    synthetic_fs,
                    strength=strength
                )
                
                # Corrupted FS → FFPE 복구
                restored_ffpe = self.gen_ab(corrupted_fs)
                
                outputs['synthetic_fs'] = synthetic_fs
                outputs['corrupted_fs'] = corrupted_fs
                outputs['restored_ffpe'] = restored_ffpe
                outputs['artifact_map'] = artifact_map
                outputs['artifact_strength'] = strength
        
        return outputs
    
    def compute_generator_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Generator loss 계산.
        
        Returns:
            Dict containing all loss components
        """
        losses = {}
        
        real_a = outputs['real_a']
        real_b = outputs['real_b']
        fake_a = outputs['fake_a']
        fake_b = outputs['fake_b']
        reco_a = outputs['reco_a']
        reco_b = outputs['reco_b']
        
        # ============================================================
        # 1. GAN Loss
        # ============================================================
        pred_fake_b = self.disc_b(fake_b)
        pred_fake_a = self.disc_a(fake_a)
        
        # Handle multi-scale discriminator output
        if isinstance(pred_fake_b, list):
            loss_gan_ab = sum(self.criterion_gan(p, torch.ones_like(p)) for p in pred_fake_b)
            loss_gan_ba = sum(self.criterion_gan(p, torch.ones_like(p)) for p in pred_fake_a)
        else:
            loss_gan_ab = self.criterion_gan(pred_fake_b, torch.ones_like(pred_fake_b))
            loss_gan_ba = self.criterion_gan(pred_fake_a, torch.ones_like(pred_fake_a))
        
        losses['gan_ab'] = loss_gan_ab
        losses['gan_ba'] = loss_gan_ba
        
        # ============================================================
        # 2. Cycle Consistency Loss (with Self-Challenging)
        # ============================================================
        if self.use_self_challenging and self.self_challenging is not None:
            # Self-Challenging: 어려운 영역에 더 높은 weight
            loss_cycle_a, weight_map_a = self.self_challenging(
                reco_a, real_a, self.disc_a
            )
            loss_cycle_b, weight_map_b = self.self_challenging(
                reco_b, real_b, self.disc_b
            )
            loss_cycle_a = self.lambda_cycle * loss_cycle_a
            loss_cycle_b = self.lambda_cycle * loss_cycle_b
            
            # Weight map 저장 (visualization용)
            outputs['weight_map_a'] = weight_map_a
            outputs['weight_map_b'] = weight_map_b
        else:
            # Standard cycle loss
            loss_cycle_a = self.lambda_cycle * self.criterion_cycle(reco_a, real_a)
            loss_cycle_b = self.lambda_cycle * self.criterion_cycle(reco_b, real_b)
        
        losses['cycle_a'] = loss_cycle_a
        losses['cycle_b'] = loss_cycle_b
        
        # ============================================================
        # 3. Identity Loss (optional)
        # ============================================================
        if self.lambda_identity > 0:
            idt_a = self.gen_ba(real_a)  # G_BA(A) should be A
            idt_b = self.gen_ab(real_b)  # G_AB(B) should be B
            loss_idt_a = self.lambda_identity * self.criterion_identity(idt_a, real_a)
            loss_idt_b = self.lambda_identity * self.criterion_identity(idt_b, real_b)
            losses['idt_a'] = loss_idt_a
            losses['idt_b'] = loss_idt_b
        
        # ============================================================
        # 4. Progressive Artifact Injection Loss 🔥
        # ============================================================
        if 'restored_ffpe' in outputs:
            restored_ffpe = outputs['restored_ffpe']
            loss_artifact = self.lambda_artifact * self.criterion_cycle(
                restored_ffpe, real_b
            )
            losses['artifact'] = loss_artifact
            
            # Artifact strength logging
            losses['artifact_strength'] = outputs.get('artifact_strength', 0.0)
        
        # ============================================================
        # Total Loss
        # ============================================================
        total_loss = (
            losses['gan_ab'] + losses['gan_ba'] +
            losses['cycle_a'] + losses['cycle_b']
        )
        
        if 'idt_a' in losses:
            total_loss = total_loss + losses['idt_a'] + losses['idt_b']
            
        if 'artifact' in losses:
            total_loss = total_loss + losses['artifact']
        
        losses['total'] = total_loss
        
        return losses
    
    def compute_discriminator_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Discriminator loss 계산."""
        losses = {}
        
        real_a = outputs['real_a']
        real_b = outputs['real_b']
        fake_a = outputs['fake_a'].detach()
        fake_b = outputs['fake_b'].detach()
        
        # Discriminator A
        pred_real_a = self.disc_a(real_a)
        pred_fake_a = self.disc_a(fake_a)
        
        if isinstance(pred_real_a, list):
            loss_d_a = sum(
                self.criterion_gan(pr, torch.ones_like(pr)) +
                self.criterion_gan(pf, torch.zeros_like(pf))
                for pr, pf in zip(pred_real_a, pred_fake_a)
            ) * 0.5
        else:
            loss_d_a = (
                self.criterion_gan(pred_real_a, torch.ones_like(pred_real_a)) +
                self.criterion_gan(pred_fake_a, torch.zeros_like(pred_fake_a))
            ) * 0.5
        
        # Discriminator B
        pred_real_b = self.disc_b(real_b)
        pred_fake_b = self.disc_b(fake_b)
        
        if isinstance(pred_real_b, list):
            loss_d_b = sum(
                self.criterion_gan(pr, torch.ones_like(pr)) +
                self.criterion_gan(pf, torch.zeros_like(pf))
                for pr, pf in zip(pred_real_b, pred_fake_b)
            ) * 0.5
        else:
            loss_d_b = (
                self.criterion_gan(pred_real_b, torch.ones_like(pred_real_b)) +
                self.criterion_gan(pred_fake_b, torch.zeros_like(pred_fake_b))
            ) * 0.5
        
        losses['disc_a'] = loss_d_a
        losses['disc_b'] = loss_d_b
        losses['total'] = loss_d_a + loss_d_b
        
        return losses
    
    def compute_artifact_net_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        ArtifactNet loss 계산.
        
        ArtifactNet의 목표: Generator가 복구하기 어려운 artifact 생성
        → Generator loss가 높아지는 방향으로 학습 (adversarial)
        """
        if not self.use_artifact_injection or 'restored_ffpe' not in outputs:
            return None
        
        restored_ffpe = outputs['restored_ffpe']
        real_b = outputs['real_b']
        
        # ArtifactNet은 Generator가 실패하길 원함
        # = restored_ffpe와 real_b의 차이가 크길 원함
        # = -L1(restored, real) 최대화 = L1(restored, real) 최소화의 반대
        
        # 하지만 너무 강한 artifact는 의미없으니 regularization 추가
        artifact_map = outputs.get('artifact_map', None)
        
        # Adversarial loss: Generator가 못 복구하게
        recon_error = torch.abs(restored_ffpe - real_b).mean()
        loss_adversarial = -recon_error  # maximize reconstruction error
        
        # Regularization: artifact가 너무 강하면 안 됨
        if artifact_map is not None:
            loss_reg = artifact_map.mean() * 0.1  # artifact 크기 제한
        else:
            loss_reg = 0.0
        
        loss_artifact_net = loss_adversarial + loss_reg
        
        return loss_artifact_net
    
    def train_step(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Returns:
            Dict of loss values for logging
        """
        real_a = real_a.to(self.device)
        real_b = real_b.to(self.device)
        
        # ============================================================
        # 1. Generator Update
        # ============================================================
        optimizer_g.zero_grad()
        
        outputs = self.forward_generators(real_a, real_b)
        g_losses = self.compute_generator_loss(outputs)
        
        g_losses['total'].backward()
        optimizer_g.step()
        
        # ============================================================
        # 2. Discriminator Update
        # ============================================================
        optimizer_d.zero_grad()
        
        d_losses = self.compute_discriminator_loss(outputs)
        
        d_losses['total'].backward()
        optimizer_d.step()
        
        # ============================================================
        # 3. ArtifactNet Update (if enabled)
        # ============================================================
        if self.use_artifact_injection and self.artifact_net is not None:
            strength = self.get_artifact_strength()
            
            if strength > 0:
                self.artifact_optimizer.zero_grad()
                
                # Re-forward for artifact net
                outputs = self.forward_generators(real_a, real_b)
                artifact_loss = self.compute_artifact_net_loss(outputs)
                
                if artifact_loss is not None:
                    artifact_loss.backward()
                    self.artifact_optimizer.step()
                    g_losses['artifact_net'] = artifact_loss.item()
        
        # ============================================================
        # Logging
        # ============================================================
        log_dict = {
            'g_total': g_losses['total'].item(),
            'g_gan_ab': g_losses['gan_ab'].item(),
            'g_gan_ba': g_losses['gan_ba'].item(),
            'g_cycle_a': g_losses['cycle_a'].item(),
            'g_cycle_b': g_losses['cycle_b'].item(),
            'd_total': d_losses['total'].item(),
            'd_a': d_losses['disc_a'].item(),
            'd_b': d_losses['disc_b'].item(),
        }
        
        if 'idt_a' in g_losses:
            log_dict['g_idt_a'] = g_losses['idt_a'].item()
            log_dict['g_idt_b'] = g_losses['idt_b'].item()
            
        if 'artifact' in g_losses:
            log_dict['g_artifact'] = g_losses['artifact'].item()
            log_dict['artifact_strength'] = g_losses['artifact_strength']
            
        if 'artifact_net' in g_losses:
            log_dict['artifact_net'] = g_losses['artifact_net']
        
        return log_dict


# ============================================================
# UVCGAN2 통합을 위한 Mixin
# ============================================================

def patch_cyclegan_backward_generators(cyclegan_instance, epoch: int):
    """
    기존 UVCGAN2 CycleGAN의 backward_generators를 패치하는 헬퍼 함수.
    
    사용법:
        # cyclegan.py의 backward_generators 안에서
        patch_cyclegan_backward_generators(self, current_epoch)
    """
    # 이 함수는 실제 uvcgan2 코드에 통합할 때 참고용
    pass
