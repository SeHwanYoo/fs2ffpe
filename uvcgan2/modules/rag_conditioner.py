"""
RAGConditioner — Retrieved FFPE를 Generator 입력에 주입

NLP RAG처럼 검색 결과를 모델 입력에 넣는 진짜 RAG.

동작:
  1. real_a (FS) → UNI encode (no grad, shared model)
  2. FFPE feature DB에서 top-1 nearest neighbor 검색
  3. 매칭된 real FFPE 이미지 로드
  4. FuseNet으로 [FS, ref_FFPE] → fused 3ch input 생성
  5. Generator가 fused input으로 fake FFPE 생성

    FS ──┐
         ├─ FuseNet ─→ fused (3ch) ─→ Generator ─→ fake FFPE
    ref ─┘

FuseNet은 learnable — "reference에서 뭘 가져올지" 학습.
Generator 아키텍처 변경 없음 (3ch 그대로).

Usage:
    conditioner = RAGConditioner(
        ffpe_features_path='rag_cache/ffpe_features.npy',
        ffpe_filenames_path='rag_cache/ffpe_filenames.npy',
        ffpe_image_dir='/path/to/trainB',
        uni_model=shared_uni,  # UNIPerceptualLoss에서 공유
    )

    # Training step:
    ref_img = conditioner.retrieve(real_a)       # (B, 3, H, W)
    fused = conditioner.fuse(real_a, ref_img)     # (B, 3, H, W)
    fake_b = generator(fused)                     # generator 변경 없음
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from functools import lru_cache


class FuseNet(nn.Module):
    """
    [FS, ref_FFPE] (6ch) → fused (3ch).
    
    Residual 구조: fused = FS + alpha * blend(FS, ref)
    → 학습 초반에는 alpha≈0이라 거의 FS 그대로
    → 학습이 진행되면 reference에서 유용한 정보를 blend
    """
    def __init__(self, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, mid_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_ch, 3, 3, padding=1),
            nn.Tanh(),
        )
        # Alpha: 0에서 시작 → reference 영향 점진적 증가
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, fs, ref):
        """
        fs:  (B, 3, H, W) FS image
        ref: (B, 3, H, W) retrieved FFPE image
        Returns: (B, 3, H, W) fused input for generator
        """
        combined = torch.cat([fs, ref], dim=1)  # (B, 6, H, W)
        blend = self.net(combined)               # (B, 3, H, W)
        alpha = torch.sigmoid(self.alpha)        # [0, 1]
        return fs + alpha * blend                # residual


class RAGConditioner(nn.Module):
    """
    Complete RAG conditioning pipeline.
    
    Supports two cache formats:
      (A) Split format (precompute_rag_matches_uni_split_v2.py):
          cache_dir/train/rag_lookup.pt (contains ffpe_paths with absolute paths)
          cache_dir/train/ffpe_features.npy
      
      (B) Old format (precompute_rag_matches_uni.py):
          cache_dir/ffpe_features.npy
          cache_dir/ffpe_filenames.npy
          + ffpe_image_dir 필요
    """

    def __init__(self, cache_dir, uni_model, image_size=256,
                 fuse_mid_ch=32, ffpe_image_dir=None):
        super().__init__()

        cache_dir = Path(cache_dir)

        # --- Try split format first (train/rag_lookup.pt) ---
        lookup_path = cache_dir / 'rag_lookup.pt'
        if not lookup_path.exists():
            lookup_path = cache_dir / 'train' / 'rag_lookup.pt'

        self.ffpe_paths = None  # absolute paths (split format)
        self.ffpe_filenames = None
        self.ffpe_image_dir = Path(ffpe_image_dir) if ffpe_image_dir else None

        if lookup_path.exists():
            lookup = torch.load(lookup_path, map_location='cpu', weights_only=False)
            # Split format has ffpe_paths (absolute)
            if 'ffpe_paths' in lookup:
                self.ffpe_paths = lookup['ffpe_paths']
                print(f"  [RAG-Cond] Using absolute paths from lookup ({len(self.ffpe_paths)} DX)")
            if 'ffpe_filenames' in lookup:
                self.ffpe_filenames = lookup['ffpe_filenames']

        # --- FFPE features ---
        feat_path = cache_dir / 'ffpe_features.npy'
        if not feat_path.exists():
            feat_path = cache_dir / 'train' / 'ffpe_features.npy'
        if not feat_path.exists():
            raise FileNotFoundError(f"No ffpe_features.npy found in {cache_dir}")

        self.ffpe_features = np.load(feat_path)
        print(f"  [RAG-Cond] FFPE DB: {self.ffpe_features.shape[0]} patches, dim={self.ffpe_features.shape[1]}")

        # --- Filenames fallback (old format) ---
        if self.ffpe_filenames is None:
            fname_path = cache_dir / 'ffpe_filenames.npy'
            if not fname_path.exists():
                fname_path = cache_dir / 'train' / 'ffpe_filenames.npy'
            if fname_path.exists():
                self.ffpe_filenames = np.load(fname_path, allow_pickle=True).tolist()

        self.image_size = image_size
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # --- Shared UNI (frozen, from UNIPerceptualLoss) ---
        self.uni_model = uni_model  # 외부에서 주입, frozen
        self.uni_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.uni_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # --- FuseNet (learnable) ---
        self.fuse_net = FuseNet(mid_ch=fuse_mid_ch)

        # --- Image cache (LRU) ---
        self._img_cache = {}
        self._cache_max = 5000

        print(f"  [RAG-Cond] FuseNet: 6ch → {fuse_mid_ch}ch → 3ch (residual)")
        print(f"  [RAG-Cond] Ready")

    def _preprocess_uni(self, images):
        """Training tensor → UNI input."""
        if images.min() < 0:
            images = (images + 1) / 2
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=224,
                                   mode='bilinear', align_corners=False)
        mean = self.uni_mean.to(images.device)
        std = self.uni_std.to(images.device)
        return (images - mean) / std

    @torch.no_grad()
    def _encode_uni(self, images):
        """(B, 3, H, W) → (B, 1024) L2-normalized."""
        x = self._preprocess_uni(images)
        if next(self.uni_model.parameters()).device != x.device:
            self.uni_model = self.uni_model.to(x.device)
        feat = self.uni_model(x)
        return F.normalize(feat, dim=1)

    @torch.no_grad()
    def _search_topk(self, query_feat, k=1):
        """
        (B, D) query → (B, k) FFPE indices.
        Brute force cosine (features already L2-normalized).
        """
        q = query_feat.cpu().numpy()           # (B, D)
        sim = q @ self.ffpe_features.T         # (B, N)
        topk = np.argsort(-sim, axis=1)[:, :k]  # (B, k)
        return topk

    def _load_image(self, idx_or_name, device):
        """FFPE 이미지 로드 (캐시 사용).
        
        Split format: ffpe_paths[idx]로 절대경로 직접 로드
        Old format: ffpe_image_dir + name + extension 조합
        """
        cache_key = str(idx_or_name)
        if cache_key in self._img_cache:
            return self._img_cache[cache_key].to(device)

        img_tensor = torch.zeros(3, self.image_size, self.image_size)
        loaded = False

        # Method 1: 절대경로 (split format)
        if self.ffpe_paths is not None and isinstance(idx_or_name, int):
            abs_path = self.ffpe_paths[idx_or_name]
            if os.path.exists(abs_path):
                try:
                    img = Image.open(abs_path).convert('RGB')
                    img_tensor = self.img_transform(img)
                    loaded = True
                except Exception:
                    pass

        # Method 2: 파일명 + image_dir (old format)
        if not loaded and self.ffpe_image_dir is not None:
            ffpe_name = self.ffpe_filenames[idx_or_name] if isinstance(idx_or_name, int) else idx_or_name
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '']:
                p = self.ffpe_image_dir / f"{ffpe_name}{ext}"
                if p.exists():
                    try:
                        img = Image.open(p).convert('RGB')
                        img_tensor = self.img_transform(img)
                        loaded = True
                    except Exception:
                        pass
                    break

        # Cache
        if len(self._img_cache) < self._cache_max:
            self._img_cache[cache_key] = img_tensor

        return img_tensor.to(device)

    def retrieve(self, real_a):
        """
        FS batch → matched ref FFPE images.
        
        Args:
            real_a: (B, 3, H, W) FS images
        Returns:
            ref_imgs: (B, 3, H, W) matched FFPE images, same device
        """
        # 1. Encode FS with UNI
        query_feat = self._encode_uni(real_a)  # (B, 1024)

        # 2. Search top-1
        top1_indices = self._search_topk(query_feat, k=1)  # (B, 1)

        # 3. Load images
        B = real_a.shape[0]
        ref_imgs = torch.zeros_like(real_a)

        for i in range(B):
            idx = int(top1_indices[i, 0])
            ref_imgs[i] = self._load_image(idx, real_a.device)

        return ref_imgs

    def fuse(self, real_a, ref_img):
        """
        FS + ref FFPE → fused generator input.
        
        Args:
            real_a: (B, 3, H, W) FS image
            ref_img: (B, 3, H, W) retrieved ref FFPE
        Returns:
            fused: (B, 3, H, W) — generator에 넣을 입력
        """
        # Resize ref to match real_a if needed
        if ref_img.shape[2:] != real_a.shape[2:]:
            ref_img = F.interpolate(ref_img, size=real_a.shape[2:],
                                    mode='bilinear', align_corners=False)
        return self.fuse_net(real_a, ref_img)

    def forward(self, real_a):
        """retrieve + fuse in one call."""
        ref_img = self.retrieve(real_a)
        return self.fuse(real_a, ref_img), ref_img
