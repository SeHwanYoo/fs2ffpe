"""
UVCGAN2-DeepThaw: UVCGAN2 + UNI Perceptual Loss
===================================================

저장 위치: uvcgan2/cgan/uvcgan2_deepthaw.py

UVCGAN2를 상속. backward_gen()만 override.
원본 uvcgan2.py 절대 안 건드림.

왜 UNI인가:
  - 고정 stain matrix (Ruifrok) 기반 decomposition은 실패함
    (FS의 OD 분포가 FFPE와 달라서 분해 자체가 틀림)
  - UNI는 병리 이미지 100M+ 학습 → stain variation에 invariant하면서 structural change에 sensitive한 feature
  - Feature space에서 loss 걸면 "조직 구조는 보존, 스타일은 변환" 자연스럽게 됨

Loss 구성:
  1. UNI Content Loss: cos_sim(UNI(fake_FFPE), UNI(FS_input))
     → 같은 조직이면 UNI feature가 비슷해야 함
  2. UNI Distribution Loss: mean/std matching of UNI features
     → fake FFPE의 feature 분포가 real FFPE와 같아야 함
  3. Self-Challenging (optional): 어려운 영역 가중
  4. RAG (optional): reference FFPE 일관성
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .uvcgan2 import UVCGAN2, queued_forward
from .named_dict import NamedDict


# ============================================================
# 1. UNI Perceptual Loss
# ============================================================
class UNIPerceptualLoss(nn.Module):
    """
    UNI foundation model을 frozen feature extractor로 사용.

    Content Loss:
        UNI(fake_FFPE)와 UNI(FS_input)의 cosine similarity.
        UNI는 조직 형태(morphology)를 캡처하면서
        stain variation에 invariant한 feature를 추출 (S2 실험 검증 완료).

    Distribution Loss:
        UNI(fake_FFPE)의 batch-level mean/std를 UNI(real_FFPE)와 매칭.
        → fake FFPE가 real FFPE의 feature 분포에 들어가도록.

    Gradient flow:
        UNI params: frozen (no grad)
        fake_FFPE → UNI → loss: grad flows to generator
        FS_input, real_FFPE → UNI: no grad (detached)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device, lambda_content=1.0, lambda_distrib=1.0):
        super().__init__()
        self.lambda_content = lambda_content
        self.lambda_distrib = lambda_distrib

        # Load & freeze UNI
        self.uni = self._load_uni(device)
        self.uni.eval()
        for p in self.uni.parameters():
            p.requires_grad_(False)

        # ImageNet normalization
        self.register_buffer(
            'img_mean',
            torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'img_std',
            torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

        print(f"  [UNI] Loaded & frozen. "
              f"content={lambda_content}, distrib={lambda_distrib}")

    @staticmethod
    def _load_uni(device):
        """UNI ViT-L/16 (1024-dim) 로드."""
        HF_TOKEN = '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'
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN)
        except: pass

        import timm
        model = timm.create_model(
            "hf_hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True
        )
        model = model.to(device)
        print("  [UNI] Loaded via timm (HF hub)")
        return model

    def _to_01(self, img):
        """[-1,1] or [0,1] → [0,1]."""
        if img.min() < -0.5:
            img = (img + 1.0) / 2.0
        return img.clamp(0.0, 1.0)

    def _preprocess(self, img):
        """Generator output → UNI input format."""
        x = self._to_01(img)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear',
                              align_corners=False)
        x = (x - self.img_mean.to(x.device)) / self.img_std.to(x.device)
        return x

    def _encode(self, img, grad=False):
        """UNI feature extraction → (B, 1024) L2-normalized."""
        x = self._preprocess(img)
        if grad:
            feat = self.uni(x)
        else:
            with torch.no_grad():
                feat = self.uni(x)
        return F.normalize(feat, dim=1)

    def forward(self, fs_input, fake_ffpe, real_ffpe):
        """
        Args:
            fs_input:   FS 이미지 (B, 3, H, W)
            fake_ffpe:  생성된 FFPE (B, 3, H, W) - grad ON
            real_ffpe:  진짜 FFPE (B, 3, H, W)

        Returns: (total, content_loss, distrib_loss)
        """
        fs_feat = self._encode(fs_input, grad=False)        # (B, D)
        fake_feat = self._encode(fake_ffpe, grad=True)       # (B, D) ← grad ON
        real_feat = self._encode(real_ffpe, grad=False)       # (B, D)

        # 1. Content: 같은 조직 → 비슷한 UNI feature
        cos_sim = F.cosine_similarity(fake_feat, fs_feat, dim=1)  # (B,)
        content_loss = (1.0 - cos_sim).mean()

        # 2. Distribution: fake FFPE feature 분포 → real FFPE feature 분포
        distrib_loss = F.l1_loss(
            fake_feat.mean(dim=0), real_feat.mean(dim=0)
        ) + F.l1_loss(
            fake_feat.std(dim=0), real_feat.std(dim=0)
        )

        total = self.lambda_content * content_loss \
            + self.lambda_distrib * distrib_loss

        return total, content_loss, distrib_loss


# ============================================================
# 2. Self-Challenging Weight
# ============================================================
class SelfChallengingWeight(nn.Module):
    """Cycle loss에 per-pixel difficulty weight."""

    def __init__(self, challenge_weight=2.0, max_ratio=5.0):
        super().__init__()
        self.challenge_weight = challenge_weight
        self.max_ratio = max_ratio

    def forward(self, reco, real):
        """Weighted L1 loss. Returns scalar."""
        with torch.no_grad():
            diff = torch.abs(reco - real).mean(dim=1, keepdim=True)
            d_min = diff.amin(dim=[2, 3], keepdim=True)
            d_max = diff.amax(dim=[2, 3], keepdim=True)
            diff_norm = (diff - d_min) / (d_max - d_min + 1e-8)
            weight = 1.0 + self.challenge_weight * diff_norm
            weight = weight.clamp(1.0, self.max_ratio)
            weight = weight / weight.mean()

        return (weight * torch.abs(reco - real)).mean()


# ============================================================
# 3. UVCGAN2-DeepThaw Model
# ============================================================
class UVCGAN2DeepThaw(UVCGAN2):
    """
    UVCGAN2 + UNI perceptual loss + optional SC/RAG.

    원본 UVCGAN2 기능 100% 유지 (EMA, batch head, queue, spectr_norm).
    backward_gen()만 override.
    """

    def __init__(
        self, savedir, config, is_train, device,
        # === UVCGAN2 원본 args ===
        head_config=None,
        lambda_a=5.0,
        lambda_b=5.0,
        lambda_idt=0.5,
        lambda_consist=0,
        head_queue_size=3,
        avg_momentum=0.9999,
        consistency=None,
        # === UNI Perceptual Loss ===
        use_uni_loss=False,
        lambda_uni_content=1.0,
        lambda_uni_distrib=1.0,
        # === Self-Challenging ===
        use_self_challenging=False,
        challenge_weight=2.0,
        max_weight_ratio=5.0,
        # === RAG ===
        use_rag=False,
        rag_cache_dir=None,
        rag_k_neighbors=5,
        lambda_rag=1.0,
        rag_mode='feature',
        ffpe_image_dir=None,
        rag_pixel_weight=1.0,
        rag_feature_weight=1.0,
        stain_push_weight=0.5,
        # === Misc ===
        xai_log_every=500,
    ):
        # Flags (super().__init__이 _setup_losses 호출하므로 먼저 설정)
        self.use_uni_loss = use_uni_loss
        self.use_self_challenging = use_self_challenging
        self.use_rag = use_rag

        self.lambda_uni_content = lambda_uni_content
        self.lambda_uni_distrib = lambda_uni_distrib
        self.lambda_rag = lambda_rag
        self.xai_log_every = xai_log_every
        self._step_count = 0

        # 모듈 None 초기화 (super().__init__ 전에)
        self.uni_loss_fn = None
        self.sc_weight_fn = None
        self.rag_module = None
        self.rag_conditioner = None  # input-mode RAG
        self._original_real_a = None

        # UVCGAN2 초기화 (EMA, head, queue 전부)
        super().__init__(
            savedir=savedir,
            config=config,
            is_train=is_train,
            device=device,
            head_config=head_config,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            lambda_idt=lambda_idt,
            lambda_consist=lambda_consist,
            head_queue_size=head_queue_size,
            avg_momentum=avg_momentum,
            consistency=consistency,
        )

        # DeepThaw 모듈 생성 (super 이후)
        if self.is_train:
            self._init_modules(
                device, use_uni_loss, use_self_challenging, use_rag,
                lambda_uni_content, lambda_uni_distrib,
                challenge_weight, max_weight_ratio,
                rag_cache_dir, rag_k_neighbors,
                rag_mode, ffpe_image_dir,
                rag_pixel_weight, rag_feature_weight,
                stain_push_weight,
            )

    def _init_modules(
        self, device, use_uni, use_sc, use_rag,
        lc_content, lc_distrib,
        cw, mwr,
        rag_dir, rag_k,
        rag_mode='feature', ffpe_image_dir=None,
        rag_pixel_weight=1.0, rag_feature_weight=1.0,
        stain_push_weight=0.5,
    ):
        """DeepThaw 모듈 생성."""
        active = []

        if use_uni:
            try:
                self.uni_loss_fn = UNIPerceptualLoss(
                    device=device,
                    lambda_content=lc_content,
                    lambda_distrib=lc_distrib,
                )
                active.append(f"UNI(content={lc_content}, distrib={lc_distrib})")
            except RuntimeError as e:
                print(f"  [DeepThaw] UNI disabled: {e}")
                self.uni_loss_fn = None

        if use_sc:
            self.sc_weight_fn = SelfChallengingWeight(
                challenge_weight=cw, max_ratio=mwr,
            ).to(device)
            active.append(f"SC(w={cw}, max={mwr})")

        if use_rag:
            if rag_mode == 'input':
                # === Input-mode RAG: ref FFPE를 generator 입력에 주입 ===
                try:
                    from uvcgan2.modules.rag_conditioner import RAGConditioner
                    if rag_dir is not None:
                        shared_uni = self.uni_loss_fn.uni if self.uni_loss_fn else None
                        self.rag_conditioner = RAGConditioner(
                            cache_dir=rag_dir,
                            uni_model=shared_uni,
                            ffpe_image_dir=ffpe_image_dir,
                        ).to(device)
                        active.append(f"RAG-Input(conditioner)")
                        # FuseNet params → generator optimizer에 추가
                        self._fuse_params_pending = True
                    else:
                        print("  [DeepThaw] RAG-Input needs rag_cache_dir")
                except ImportError as e:
                    print(f"  [DeepThaw] RAG-Input disabled: {e}")
            else:
                # === Loss-mode RAG: 기존 방식 ===
                try:
                    from uvcgan2.modules.rag_fusion import RAGFusionModule
                    if rag_dir is not None:
                        shared_uni = self.uni_loss_fn.uni if self.uni_loss_fn else None
                        self.rag_module = RAGFusionModule(
                            cache_dir=rag_dir, k=rag_k,
                            rag_mode=rag_mode,
                            shared_uni_model=shared_uni,
                            ffpe_image_dir=ffpe_image_dir,
                            pixel_weight=rag_pixel_weight,
                            feature_weight=rag_feature_weight,
                            stain_push_weight=stain_push_weight,
                        ).to(device)
                        active.append(f"RAG(k={rag_k}, mode={rag_mode})")
                    else:
                        print("  [DeepThaw] RAG disabled: no cache dir")
                except ImportError:
                    print("  [DeepThaw] RAG disabled: module not found")

        if active:
            print(f"  [DeepThaw] Active: {', '.join(active)}")
        else:
            print("  [DeepThaw] WARNING: no modules active")

    def _setup_losses(self, config):
        """원본 + DeepThaw loss names."""
        names = ['gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b']

        if self.is_train and self.lambda_idt > 0:
            names += ['idt_a', 'idt_b']
        if self.is_train and config.gradient_penalty is not None:
            names += ['gp_a', 'gp_b']
        if self.consist_model is not None:
            names += ['consist_a', 'consist_b']

        # DeepThaw
        if self.use_uni_loss:
            names += ['dt_uni_content', 'dt_uni_distrib']
        if self.use_self_challenging:
            names += ['dt_sc_a', 'dt_sc_b']
        if self.use_rag:
            names += ['dt_rag']

        return NamedDict(*names)

    # ==============================================================
    # RAG Input Conditioning
    # ==============================================================

    def forward(self):
        """Override: RAG-Input mode에서 fused input을 generator에 전달."""
        if self.rag_conditioner is not None and self.images.real_a is not None:
            # FuseNet params를 gen optimizer에 추가 (최초 1회)
            if getattr(self, '_fuse_params_pending', False):
                try:
                    for p in self.rag_conditioner.fuse_net.parameters():
                        self.optimizers.gen.param_groups[0]['params'].append(p)
                    print("  [RAG-Input] FuseNet params added to gen optimizer")
                except Exception as e:
                    print(f"  [RAG-Input] optimizer hookup failed: {e}")
                self._fuse_params_pending = False

            # 1. Retrieve ref FFPE + fuse
            fused, ref_img = self.rag_conditioner(self.images.real_a)

            # 2. Store original real_a (for cycle/UNI losses)
            self._original_real_a = self.images.real_a

            # 3. Swap: generator sees fused input
            self.images.real_a = fused

        # 4. Original UVCGAN2 forward (gen_ab, gen_ba, etc.)
        super().forward()

        # 5. Restore original real_a (for loss computation)
        if self._original_real_a is not None:
            self.images.real_a = self._original_real_a
            self._original_real_a = None

    def backward_gen(self, direction):
        """UVCGAN2 backward_gen + UNI/SC/RAG losses."""

        if direction == 'ab':
            # ---- 원본 UVCGAN2 ----
            (self.losses.gen_ab, self.losses.cycle_a, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_b,
                    self.images.real_a, self.images.fake_b, self.images.reco_a,
                    self.queues.fake_b, self.lambda_a
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )
                loss += self.losses.consist_a

            # ---- UNI Perceptual Loss ----
            if self.uni_loss_fn is not None and self.images.real_b is not None:
                uni_total, c_loss, d_loss = self.uni_loss_fn(
                    fs_input=self.images.real_a,
                    fake_ffpe=self.images.fake_b,
                    real_ffpe=self.images.real_b,
                )
                loss = loss + uni_total
                self.losses.dt_uni_content = c_loss
                self.losses.dt_uni_distrib = d_loss

            # ---- Self-Challenging (cycle_a 교체) ----
            if self.sc_weight_fn is not None:
                sc_loss = self.sc_weight_fn(
                    self.images.reco_a, self.images.real_a
                )
                loss = loss - self.losses.cycle_a + self.lambda_a * sc_loss
                self.losses.dt_sc_a = self.lambda_a * sc_loss

            # ---- RAG ----
            if self.rag_module is not None:
                try:
                    rag_loss = self.rag_module.compute_loss(
                        self.images.fake_b, self.images.real_a
                    )
                    loss = loss + self.lambda_rag * rag_loss
                    self.losses.dt_rag = self.lambda_rag * rag_loss
                except Exception:
                    pass

        elif direction == 'ba':
            # ---- 원본 UVCGAN2 ----
            (self.losses.gen_ba, self.losses.cycle_b, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_a,
                    self.images.real_b, self.images.fake_a, self.images.reco_b,
                    self.queues.fake_a, self.lambda_b
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )
                loss += self.losses.consist_b

            # ---- SC (ba 방향) ----
            if self.sc_weight_fn is not None:
                sc_loss = self.sc_weight_fn(
                    self.images.reco_b, self.images.real_b
                )
                loss = loss - self.losses.cycle_b + self.lambda_b * sc_loss
                self.losses.dt_sc_b = self.lambda_b * sc_loss

        elif direction == 'aa':
            (self.losses.idt_a, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a, self.lambda_a
                )

        elif direction == 'bb':
            (self.losses.idt_b, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_b, self.images.idt_b, self.lambda_b
                )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

        loss.backward()
        self._step_count += 1

    def get_current_losses(self):
        """Override: DeepThaw losses가 None일 수 있음."""
        result = {}
        try:
            items = self.losses.items()
        except AttributeError:
            items = vars(self.losses).items()
        for k, v in items:
            if v is None:
                continue
            result[k] = float(v)
        return result
