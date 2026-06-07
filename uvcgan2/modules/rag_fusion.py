"""
RAGFusionModule v3 — C-RAG: Contrastive Retrieval-Augmented Generation

Modes:
  'feature'      — L1 in UNI feature space (기존, UNI loss와 겹침)
  'pixel'        — L1 in pixel space (semantic과 분리)
  'hybrid'       — pixel + feature 결합
  'contrastive'  — ★ C-RAG: pull morphology + push stain
                    Pull: UNI(fake) → UNI(ref)  (구조 보존)
                    Push: stain(fake) ↛ stain(ref) (stain 다양성)
                    → implicit stain normalization 유지하면서 morphology guidance

Usage:
    # C-RAG (recommended)
    self.rag = RAGFusionModule(cache_dir='.', rag_mode='contrastive',
                                ffpe_image_dir='/path/to/trainB',
                                shared_uni_model=self.uni_loss_fn.uni,
                                stain_push_weight=0.5)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms


class RAGFusionModule(nn.Module):

    VALID_MODES = ('feature', 'pixel', 'hybrid', 'contrastive')

    def __init__(self, cache_dir, k=5, feat_dim=1024,
                 rag_mode='feature', shared_uni_model=None,
                 ffpe_image_dir=None,
                 pixel_weight=1.0, feature_weight=1.0,
                 stain_push_weight=0.5):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.cache_dir = cache_dir
        self.rag_mode = rag_mode
        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        self.stain_push_weight = stain_push_weight

        assert rag_mode in self.VALID_MODES, \
            f"rag_mode must be one of {self.VALID_MODES}, got '{rag_mode}'"

        # Mode flags
        self.has_lookup = False

        # Load cache
        self._load_cache(cache_dir)

        # FFPE image directory (for pixel/hybrid/contrastive modes)
        self.ffpe_image_dir = ffpe_image_dir
        if rag_mode in ('pixel', 'hybrid', 'contrastive') and ffpe_image_dir is None:
            # Try to find from meta
            meta_path = os.path.join(cache_dir, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.ffpe_image_dir = meta.get('ffpe_dir')
            if self.ffpe_image_dir is None:
                print("  [RAG] WARNING: pixel/hybrid/contrastive mode needs ffpe_image_dir")

        # Image transform for loading reference FFPE images
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Match generator output range [-1, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Load UNI — shared from UNIPerceptualLoss or load new
        self.uni_model = None
        if shared_uni_model is not None:
            self.uni_model = shared_uni_model
            self.uni_mean = [0.485, 0.456, 0.406]
            self.uni_std = [0.229, 0.224, 0.225]
            print("  [RAG] UNI model shared from UNIPerceptualLoss")
        elif rag_mode in ('feature', 'hybrid', 'contrastive'):
            self._load_uni()

        print(f"  [RAG] Mode: {rag_mode}")
        if rag_mode == 'hybrid':
            print(f"  [RAG] pixel_weight={pixel_weight}, feature_weight={feature_weight}")
        if rag_mode == 'contrastive':
            print(f"  [RAG] stain_push_weight={stain_push_weight}")

    # ==============================================================
    # Loading
    # ==============================================================

    def _load_cache(self, cache_dir):
        """Load precomputed lookup. Supports both flat and split format.
        
        Flat:  cache_dir/rag_lookup.pt, cache_dir/ffpe_features.npy
        Split: cache_dir/train/rag_lookup.pt, cache_dir/train/ffpe_features.npy
        """
        # --- Lookup ---
        lookup_path = os.path.join(cache_dir, 'rag_lookup.pt')
        if not os.path.exists(lookup_path):
            lookup_path = os.path.join(cache_dir, 'train', 'rag_lookup.pt')

        if os.path.exists(lookup_path):
            print(f"  [RAG] Loading lookup: {lookup_path}")
            lookup = torch.load(lookup_path, map_location='cpu', weights_only=False)
            self.fs_name2idx = lookup['fs_name2idx']
            self.topk_indices = lookup['topk_indices']
            self.ffpe_filenames = lookup['ffpe_filenames']
            self.k = lookup.get('k', self.k)
            print(f"  [RAG] Lookup: {len(self.fs_name2idx)} FS -> "
                  f"{len(self.ffpe_filenames)} FFPE, k={self.k}")
            self.has_lookup = True

        # --- FFPE features ---
        feat_path = os.path.join(cache_dir, 'ffpe_features.npy')
        if not os.path.exists(feat_path):
            feat_path = os.path.join(cache_dir, 'train', 'ffpe_features.npy')

        if os.path.exists(feat_path):
            self.ref_features = np.load(feat_path)
            self.feat_dim = self.ref_features.shape[1]
            print(f"  [RAG] FFPE features: {self.ref_features.shape}")
        else:
            self.ref_features = None
            if self.rag_mode in ('feature', 'hybrid', 'contrastive'):
                print("  [RAG] WARNING: no ffpe_features.npy found")

    def _load_uni(self):
        """Frozen UNI encoder."""
        import timm
        self.uni_model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True,
            init_values=1e-5, dynamic_img_size=True
        )
        self.uni_model.eval()
        for p in self.uni_model.parameters():
            p.requires_grad = False
        self.uni_mean = [0.485, 0.456, 0.406]
        self.uni_std = [0.229, 0.224, 0.225]
        print("  [RAG] UNI encoder loaded (frozen)")

    # ==============================================================
    # Preprocessing / Encoding
    # ==============================================================

    def _preprocess_uni(self, images):
        """Training tensor -> UNI input."""
        if images.min() < 0:
            images = (images + 1) / 2
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=224, mode='bilinear', align_corners=False)
        mean = torch.tensor(self.uni_mean, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.uni_std, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    def _encode(self, images, grad=True):
        """(B, 3, H, W) -> (B, D) L2-normalized UNI features."""
        x = self._preprocess_uni(images)
        if next(self.uni_model.parameters()).device != x.device:
            self.uni_model = self.uni_model.to(x.device)
        if grad:
            feat = self.uni_model(x)
        else:
            with torch.no_grad():
                feat = self.uni_model(x)
        return F.normalize(feat, dim=1)

    # ==============================================================
    # Retrieval
    # ==============================================================

    def _get_matched_indices(self, fs_names):
        """fs_names -> (B, k) matched FFPE indices."""
        B = len(fs_names)
        indices = np.zeros((B, self.k), dtype=np.int64)
        valid = np.ones(B, dtype=bool)
        for i, name in enumerate(fs_names):
            idx = self.fs_name2idx.get(name)
            if idx is None:
                stem = name.split('/')[-1]
                idx = self.fs_name2idx.get(stem)
            if idx is not None:
                indices[i] = self.topk_indices[idx]
            else:
                valid[i] = False
        return indices, valid

    def _retrieve_features(self, fs_names, device):
        """Precomputed -> (B, k, D) ref features."""
        indices, valid = self._get_matched_indices(fs_names)
        B = len(fs_names)
        ref_feats = np.zeros((B, self.k, self.feat_dim), dtype=np.float32)
        for i in range(B):
            if valid[i]:
                ref_feats[i] = self.ref_features[indices[i]]
        return (torch.from_numpy(ref_feats).to(device),
                torch.from_numpy(valid).to(device))

    def _retrieve_online(self, query_features):
        """Brute-force retrieval -> (B, k, D)."""
        query_np = query_features.detach().cpu().numpy()
        sim = query_np @ self.ref_features.T
        topk_idx = np.argsort(-sim, axis=1)[:, :self.k]
        ref_feats = self.ref_features[topk_idx]
        return torch.from_numpy(ref_feats).to(query_features.device)

    def _load_matched_images(self, fs_names, device, target_size=None):
        """
        Precomputed lookup -> top-1 matched FFPE image 로드.
        
        Returns: (B, 3, H, W) tensor on device, [-1, 1] range
        """
        indices, valid = self._get_matched_indices(fs_names)
        B = len(fs_names)
        
        # Determine target size from first valid or default 256
        h, w = target_size if target_size else (256, 256)
        ref_imgs = torch.zeros(B, 3, h, w, device=device)
        
        ffpe_base = Path(self.ffpe_image_dir) if self.ffpe_image_dir else None
        
        for i in range(B):
            if not valid[i] or ffpe_base is None:
                continue
            
            # top-1 match
            ffpe_name = self.ffpe_filenames[indices[i, 0]]
            
            # Find image file
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                p = ffpe_base / f"{ffpe_name}{ext}"
                if p.exists():
                    img_path = p
                    break
            # Try without extension (maybe name already has ext)
            if img_path is None:
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    # Search recursively
                    candidates = list(ffpe_base.rglob(f"*{ffpe_name.split('/')[-1]}*"))
                    if candidates:
                        img_path = candidates[0]
                        break

            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Resize to match fake_b size
                    transform = transforms.Compose([
                        transforms.Resize((h, w)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                    ref_imgs[i] = transform(img).to(device)
                except Exception:
                    pass  # zero tensor as fallback

        return ref_imgs, torch.from_numpy(valid).to(device)

    # ==============================================================
    # Loss Functions
    # ==============================================================

    def _loss_feature(self, fake_b, real_a, fs_names):
        """기존: L1 in UNI feature space."""
        fake_feat = self._encode(fake_b, grad=True)

        if fs_names is not None and self.has_lookup:
            ref_feats, valid = self._retrieve_features(fs_names, fake_feat.device)
            if not valid.any():
                query_feat = self._encode(real_a, grad=False)
                ref_feats = self._retrieve_online(query_feat)
            elif not valid.all():
                missing = ~valid
                query_feat = self._encode(real_a[missing], grad=False)
                ref_feats[missing] = self._retrieve_online(query_feat)
        else:
            query_feat = self._encode(real_a, grad=False)
            ref_feats = self._retrieve_online(query_feat)

        ref_mean = F.normalize(ref_feats.mean(dim=1), dim=1)
        return F.l1_loss(fake_feat, ref_mean)

    def _loss_pixel(self, fake_b, real_a, fs_names):
        """방법 1: Pixel-level L1 with matched FFPE image."""
        if not self.has_lookup or self.ffpe_image_dir is None:
            return torch.tensor(0.0, device=fake_b.device)

        h, w = fake_b.shape[2], fake_b.shape[3]
        ref_imgs, valid = self._load_matched_images(
            fs_names, fake_b.device, target_size=(h, w))

        if not valid.any():
            return torch.tensor(0.0, device=fake_b.device)

        # Only compute loss for valid matches
        if valid.all():
            return F.l1_loss(fake_b, ref_imgs.detach())
        else:
            return F.l1_loss(fake_b[valid], ref_imgs[valid].detach())

    def _loss_hybrid(self, fake_b, real_a, fs_names):
        """방법 3: Pixel + Feature 결합."""
        loss_f = self._loss_feature(fake_b, real_a, fs_names)
        loss_p = self._loss_pixel(fake_b, real_a, fs_names)
        return self.feature_weight * loss_f + self.pixel_weight * loss_p

    @staticmethod
    def _stain_descriptor(img):
        """
        Image → stain representation (grad-friendly).
        
        Stain = color statistics per channel.
        (B, 3, H, W) → (B, 9): [mean_R, mean_G, mean_B, std_R, std_G, std_B,
                                  skew_R, skew_G, skew_B]
        
        [-1,1] range 이미지를 [0,1]로 변환 후 계산.
        """
        # Ensure [0, 1]
        if img.min() < 0:
            img = (img + 1) / 2

        B, C = img.shape[0], img.shape[1]
        flat = img.view(B, C, -1)  # (B, 3, H*W)

        mean = flat.mean(dim=2)  # (B, 3)
        std = flat.std(dim=2)    # (B, 3)

        # Skewness: captures stain distribution asymmetry
        centered = flat - mean.unsqueeze(2)
        skew = (centered ** 3).mean(dim=2) / (std ** 3 + 1e-8)  # (B, 3)

        return torch.cat([mean, std, skew], dim=1)  # (B, 9)

    def _loss_contrastive(self, fake_b, real_a, fs_names):
        """
        C-RAG: Contrastive Retrieval-Augmented Generation.
        
        Pull: UNI(fake) → UNI(ref)  — morphology 보존
        Push: stain(fake) ↛ stain(ref) — stain 다양성 강제
        
        L = L_pull + λ_push * L_push
        
        직관: "레퍼런스의 구조만 배우고, stain은 네 맘대로 해"
        → generator가 canonical stain을 스스로 학습
        → implicit stain normalization 유지
        """
        # ---- Pull: morphology 보존 (기존 feature loss와 동일) ----
        fake_feat = self._encode(fake_b, grad=True)  # (B, D)

        if fs_names is not None and self.has_lookup:
            ref_feats, valid = self._retrieve_features(fs_names, fake_feat.device)
            if not valid.any():
                query_feat = self._encode(real_a, grad=False)
                ref_feats = self._retrieve_online(query_feat)
            elif not valid.all():
                missing = ~valid
                query_feat = self._encode(real_a[missing], grad=False)
                ref_feats[missing] = self._retrieve_online(query_feat)
        else:
            query_feat = self._encode(real_a, grad=False)
            ref_feats = self._retrieve_online(query_feat)

        ref_mean = F.normalize(ref_feats.mean(dim=1), dim=1)  # (B, D)
        L_pull = F.l1_loss(fake_feat, ref_mean)

        # ---- Push: stain 다양성 (reference stain과 달라야 함) ----
        if not self.has_lookup or self.ffpe_image_dir is None:
            return L_pull  # fallback: pull only

        h, w = fake_b.shape[2], fake_b.shape[3]
        ref_imgs, img_valid = self._load_matched_images(
            fs_names, fake_b.device, target_size=(h, w))

        if not img_valid.any():
            return L_pull  # fallback: pull only

        # Stain descriptors
        fake_stain = self._stain_descriptor(fake_b)       # (B, 9)
        ref_stain = self._stain_descriptor(ref_imgs)       # (B, 9)

        # Push: negative L1 → fake stain should differ from ref stain
        # margin 추가: stain 차이가 margin 이상이면 더 이상 push 안 함
        margin = 0.1
        stain_dist = F.l1_loss(fake_stain, ref_stain.detach(), reduction='none')  # (B, 9)
        stain_dist = stain_dist.mean(dim=1)  # (B,)

        # Hinge: margin 이하인 것만 push (이미 충분히 다르면 무시)
        L_push = F.relu(margin - stain_dist).mean()

        return L_pull + self.stain_push_weight * L_push

    # ==============================================================
    # Main
    # ==============================================================

    def compute_loss(self, fake_b, real_a, fs_names=None):
        """
        Args:
            fake_b:   (B, 3, H, W) generated FFPE — grad flows
            real_a:   (B, 3, H, W) input FS
            fs_names: list[str] or None
        Returns:
            loss: scalar tensor
        """
        if self.rag_mode == 'feature':
            return self._loss_feature(fake_b, real_a, fs_names)
        elif self.rag_mode == 'pixel':
            return self._loss_pixel(fake_b, real_a, fs_names)
        elif self.rag_mode == 'hybrid':
            return self._loss_hybrid(fake_b, real_a, fs_names)
        elif self.rag_mode == 'contrastive':
            return self._loss_contrastive(fake_b, real_a, fs_names)
        else:
            raise ValueError(f"Unknown rag_mode: {self.rag_mode}")

    def forward(self, fake_b, real_a, fs_names=None):
        return self.compute_loss(fake_b, real_a, fs_names)