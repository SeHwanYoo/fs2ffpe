"""
DeepThaw 증명 실험 패키지 (교수 피드백 기반)
==============================================

FM 후보 6개:
  B (Histology FM): CONCH, GigaPath, UNI
  A (General FM):   EfficientNet-B0, ViT-B/16
  C (Baseline):     Random ViT (no pretrain)

실험 구성:
  S1: FM별 retrieval quality 비교
  S2: Nuisance invariance vs signal sensitivity
  E1: 5-way RAG ablation (학습 후 결과 비교)
  E4: Wrong-RAG stress test
  B1-B4: Data bias 분석
  N1-N4: Noise vs information 구분

사용법:
  # FM 선택 검증 (학습 필요 없음, 바로 실행)
  python proof_experiments.py fm_validate \
    --fs_dir /home/sehwan001/datasets/linked_v2/FS2FFPE/trainA \
    --ffpe_dir /home/sehwan001/datasets/linked_v2/FS2FFPE/trainB

  # RAG ablation 결과 비교 (학습 완료 후)
  python proof_experiments.py rag_ablation \
    --csv_file idh_gbm_aggregated.csv \
    --result_dirs full=/path no_rag=/path random_rag=/path baseline=/path

  # Bias 분석
  python proof_experiments.py bias_audit \
    --csv_file idh_gbm_aggregated.csv \
    --fs_dir /home/sehwan001/datasets/linked_v2/FS2FFPE/trainA --ffpe_dir /home/sehwan001/datasets/linked_v2/FS2FFPE/trainB

  # Noise vs Information
  python proof_experiments.py noise_vs_info \
    --csv_file idh_gbm_aggregated.csv \
    --fs_dir /home/sehwan001/datasets/linked_v2/FS2FFPE/trainA --fake_ffpe_dir /path/to/generated

  # 전부
  python proof_experiments.py all --csv_file idh_gbm_aggregated.csv --fs_dir /home/sehwan001/datasets/linked_v2/FS2FFPE/trainA --fake_ffpe_dir /path/to/generated
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIL_EPOCHS = 30
MIL_LR = 2e-4
MY_HF_TOKEN = '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'


# ================================================================
# 0. FM 로더 (6개 후보)
# ================================================================
# 모든 로더는 (model, preprocess, feat_dim, encode_fn) 리턴
# encode_fn(model, batch_tensor) → (B, D) normalized features

def _hf_login():
    try:
        from huggingface_hub import login
        login(token=MY_HF_TOKEN)
    except: pass

def _imagenet_preprocess():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _generic_encode(model, batch):
    """일반 모델: model(batch) → normalize."""
    feat = model(batch)
    return F.normalize(feat, dim=1)

def _conch_encode(model, batch):
    """CONCH: encode_image with proj_contrast=False."""
    feat = model.encode_image(batch, proj_contrast=False, normalize=True)
    return feat


def load_conch(device):
    """B: CONCH ViT-B/16 (병리 특화, 512-dim)"""
    custom_lib_path = '/home/users/astar/ares/yoosehwa/scratch/my_libs'
    if os.path.exists(custom_lib_path) and custom_lib_path not in sys.path:
        sys.path.insert(0, custom_lib_path)
    _hf_login()
    from conch.open_clip_custom import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16', "hf_hub:MahmoodLab/CONCH",
        device=device, hf_auth_token=MY_HF_TOKEN
    )
    model.eval()
    return model, preprocess, 512, _conch_encode

def load_gigapath(device):
    """B: GigaPath tile encoder (병리 특화, 1536-dim)"""
    _hf_login()
    import timm
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 1536, _generic_encode

def load_uni(device):
    """B: UNI ViT-L/16 (병리 특화, 1024-dim)"""
    _hf_login()
    import timm
    model = timm.create_model(
        "hf_hub:MahmoodLab/uni", pretrained=True,
        init_values=1e-5, dynamic_img_size=True
    )
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 1024, _generic_encode

def load_efficientnet(device):
    """A: EfficientNet-B0 ImageNet (일반 비전, 1280-dim)"""
    import torchvision.models as models
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier = nn.Identity()
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 1280, _generic_encode

def load_vit(device):
    """A: ViT-B/16 ImageNet (일반 비전, 768-dim)"""
    import torchvision.models as models
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = nn.Identity()
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 768, _generic_encode

def load_random_vit(device):
    """C: Random ViT-B/16 (no pretrain, 768-dim)"""
    import torchvision.models as models
    model = models.vit_b_16(weights=None)
    model.heads = nn.Identity()
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 768, _generic_encode


# FM 카탈로그
FM_CATALOG = {
    "CONCH":        ("B: Histology FM",  load_conch,        "병리 특화 512d"),
    "GigaPath":     ("B: Histology FM",  load_gigapath,     "WSI 특화 1536d"),
    "UNI":          ("B: Histology FM",  load_uni,          "병리 특화 1024d"),
    "EfficientNet": ("A: General FM",    load_efficientnet, "ImageNet 1280d"),
    "ViT-B/16":     ("A: General FM",    load_vit,          "ImageNet 768d"),
    "Random ViT":   ("C: Baseline",      load_random_vit,   "No pretrain 768d"),
}


def load_fm(name, device):
    """FM을 이름으로 로드. Returns (model, preprocess, feat_dim, encode_fn)"""
    if name not in FM_CATALOG:
        raise ValueError(f"Unknown FM: {name}. Choose from: {list(FM_CATALOG.keys())}")
    cat, loader, desc = FM_CATALOG[name]
    print(f"  Loading {name} [{cat}] ({desc})...")
    model, preprocess, feat_dim, encode_fn = loader(device)
    print(f"  ✅ {name} loaded. feat_dim={feat_dim}")
    return model, preprocess, feat_dim, encode_fn


# ================================================================
# 1. Feature Extraction (통합)
# ================================================================
class PatchDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img), os.path.basename(self.files[idx])
        except:
            return torch.zeros(3, 224, 224), "error"


def fname_to_slide_id(fname):
    """파일명 → TCGA slide ID (TCGA-XX-XXXX)"""
    name = fname.replace("fake_B_", "").replace("fake_", "").replace("real_A_", "").replace("real_", "")
    parts = name.split('-')
    return "-".join(parts[:3]) if len(parts) >= 3 and parts[0] == "TCGA" else name.split('_')[0]


def extract_patch_features(img_dir, model, preprocess, encode_fn, device,
                           max_patches=5000, batch_size=256):
    """패치별 feature 추출. Returns {filename: (D,) tensor}"""
    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))][:max_patches]
    if not files: return {}

    loader = DataLoader(PatchDataset(files, preprocess), batch_size=batch_size, num_workers=8)
    result = {}
    with torch.no_grad():
        for imgs, fnames in tqdm(loader, desc="Patch features", leave=False):
            feats = encode_fn(model, imgs.to(device)).cpu()
            for i, f in enumerate(fnames):
                if f != "error":
                    result[f] = feats[i]
    return result


def extract_slide_features(img_dir, model, preprocess, encode_fn, device,
                           max_per_slide=100, batch_size=256):
    """Slide별 feature 추출. Returns {slide_id: (N, D) tensor}"""
    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    if not files: return {}

    loader = DataLoader(PatchDataset(files, preprocess), batch_size=batch_size, num_workers=8)
    slide_feats = defaultdict(list)
    with torch.no_grad():
        for imgs, fnames in tqdm(loader, desc="Slide features", leave=False):
            feats = encode_fn(model, imgs.to(device)).cpu()
            for i, f in enumerate(fnames):
                if f == "error": continue
                sid = fname_to_slide_id(f)
                slide_feats[sid].append(feats[i])

    return {k: torch.stack(v[:max_per_slide]) for k, v in slide_feats.items() if v}


# ================================================================
# 2. MIL 분류기 (idh_gbm_aggregated.csv 형식)
# ================================================================
class GatedAttention(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        L, D, K = 128, 64, 1
        self.fe = nn.Sequential(nn.Linear(input_dim, L), nn.ReLU())
        self.av = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.au = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.w = nn.Linear(D, K)
        self.clf = nn.Sequential(nn.Linear(L * K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        f = self.fe(x)
        A = self.w(self.av(f) * self.au(f)).T
        A = F.softmax(A, dim=1)
        return self.clf(A @ f), A


class BagDataset(Dataset):
    def __init__(self, features_dict, df, feat_dim=512):
        self.features_dict = features_dict
        self.df = df
        self.feat_dim = feat_dim
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid, label = row['slide_id'], int(row['label'])
        if pid in self.features_dict:
            return self.features_dict[pid], label
        return torch.zeros(1, self.feat_dim), label


def run_mil_classification(features_dict, df_labels, feat_dim=512, n_runs=5):
    """MIL 분류 n_runs 반복 → mean±std. df_labels 컬럼: slide_id, label, split"""
    available = list(features_dict.keys())
    df = df_labels[df_labels['slide_id'].isin(available)].reset_index(drop=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'].isin(['test', 'val'])].reset_index(drop=True)
    if len(train_df) < 5 or len(test_df) < 3:
        print(f"    ⚠️  Not enough data: train={len(train_df)}, test={len(test_df)}")
        return None

    all_res = []
    for run in range(n_runs):
        model = GatedAttention(feat_dim).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=MIL_LR)
        crit = nn.BCELoss()

        model.train()
        train_loader = DataLoader(BagDataset(features_dict, train_df, feat_dim),
                                  batch_size=1, shuffle=True)
        for _ in range(MIL_EPOCHS):
            for data, label in train_loader:
                if data.shape[1] == 0: continue
                opt.zero_grad()
                prob, _ = model(data.to(DEVICE))
                crit(prob.view(-1), label.to(DEVICE).float().view(-1)).backward()
                opt.step()

        model.eval()
        yt, yp = [], []
        test_loader = DataLoader(BagDataset(features_dict, test_df, feat_dim),
                                 batch_size=1, shuffle=False)
        with torch.no_grad():
            for data, label in test_loader:
                if data.shape[1] == 0: continue
                prob, _ = model(data.to(DEVICE))
                yp.append(prob.item()); yt.append(label.item())

        if len(np.unique(yt)) < 2: continue
        try: auc = roc_auc_score(yt, yp)
        except: auc = 0.5
        fpr, tpr, thr = roc_curve(yt, yp)
        bt = thr[np.argmax(tpr - fpr)]
        if bt < 0.1 or bt > 0.9: bt = 0.5
        pred = [1 if p >= bt else 0 for p in yp]
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        all_res.append({
            'AUC': auc, 'Accuracy': accuracy_score(yt, pred),
            'F1': f1_score(yt, pred, zero_division=0),
            'Sensitivity': recall_score(yt, pred, zero_division=0),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        })

    if not all_res: return None
    return {k: (np.mean([r[k] for r in all_res]), np.std([r[k] for r in all_res])) for k in all_res[0]}


def print_metrics_table(results_dict):
    print(f"\n  {'Method':<25} | {'AUC':<18} | {'ACC':<18} | {'F1':<18}")
    print(f"  {'-'*80}")
    for name, res in results_dict.items():
        if res is None:
            print(f"  {name:<25} | {'FAILED':<18}")
        else:
            print(f"  {name:<25} | {res['AUC'][0]:.4f}±{res['AUC'][1]:.4f}  "
                  f"| {res['Accuracy'][0]:.4f}±{res['Accuracy'][1]:.4f}  "
                  f"| {res['F1'][0]:.4f}±{res['F1'][1]:.4f}")
    print()


# ================================================================
# S1: Retrieval Quality — FM 6개 비교
# ================================================================
def exp_s1_retrieval_quality(fs_dir, ffpe_dir, top_k=5):
    """
    FS query → FFPE bank에서 top-k 검색.
    "같은 TCGA patient의 FFPE를 찾는 비율" = same-case retrieval rate.
    FM 6개 전부 비교.
    """
    print("\n" + "=" * 70)
    print("S1: Retrieval Quality — FM 후보별 검색 성능 비교")
    print("=" * 70)
    print(f"  FS query → FFPE bank top-{top_k} 검색")
    print(f"  Metric: same-case hit rate (같은 환자 FFPE 찾는 비율)\n")

    fm_results = {}

    for fm_name, (cat, loader, desc) in FM_CATALOG.items():
        print(f"\n--- [{cat}] {fm_name} ({desc}) ---")
        try:
            model, preprocess, feat_dim, encode_fn = loader(DEVICE)
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            fm_results[fm_name] = None
            continue

        fs_feats = extract_patch_features(
            fs_dir, model, preprocess, encode_fn, DEVICE, max_patches=3000)
        ffpe_feats = extract_patch_features(
            ffpe_dir, model, preprocess, encode_fn, DEVICE, max_patches=3000)

        if len(fs_feats) < 10 or len(ffpe_feats) < 10:
            print(f"  Not enough patches (FS={len(fs_feats)}, FFPE={len(ffpe_feats)})")
            fm_results[fm_name] = None
            del model; torch.cuda.empty_cache(); continue

        fs_names = list(fs_feats.keys())
        ffpe_names = list(ffpe_feats.keys())
        fs_mat = torch.stack([fs_feats[n] for n in fs_names])
        ffpe_mat = torch.stack([ffpe_feats[n] for n in ffpe_names])
        fs_slides = [fname_to_slide_id(n) for n in fs_names]
        ffpe_slides = [fname_to_slide_id(n) for n in ffpe_names]

        sim = fs_mat @ ffpe_mat.T
        topk_idx = sim.topk(top_k, dim=1).indices

        hits = 0
        prec_at_k = []
        for i, fs_slide in enumerate(fs_slides):
            retrieved = [ffpe_slides[j] for j in topk_idx[i].tolist()]
            n_correct = sum(1 for s in retrieved if s == fs_slide)
            prec_at_k.append(n_correct / top_k)
            if n_correct > 0: hits += 1

        hit_rate = hits / len(fs_slides)
        mean_prec = np.mean(prec_at_k)

        fm_results[fm_name] = {
            'hit_rate': hit_rate, 'precision_at_k': mean_prec,
            'n_queries': len(fs_slides), 'category': cat,
        }
        print(f"  Hit Rate@{top_k}: {hit_rate:.4f} ({hits}/{len(fs_slides)})")
        print(f"  Precision@{top_k}: {mean_prec:.4f}")

        del model, fs_feats, ffpe_feats, fs_mat, ffpe_mat
        torch.cuda.empty_cache()

    # 비교 테이블
    print(f"\n  {'='*70}")
    print(f"  {'FM':<20} {'Category':<20} {'Hit@k':<12} {'Prec@k':<12}")
    print(f"  {'-'*70}")
    for name in FM_CATALOG:
        r = fm_results.get(name)
        if r:
            print(f"  {name:<20} {r['category']:<20} {r['hit_rate']:.4f}       {r['precision_at_k']:.4f}")
        else:
            print(f"  {name:<20} {'—':<20} {'FAILED':<12}")
    print(f"  {'='*70}")

    # 판정
    hist_fms = {n: r for n, r in fm_results.items() if r and 'Histology' in r['category']}
    gen_fms = {n: r for n, r in fm_results.items() if r and 'General' in r['category']}
    if hist_fms and gen_fms:
        best_h = max(hist_fms.items(), key=lambda x: x[1]['hit_rate'])
        best_g = max(gen_fms.items(), key=lambda x: x[1]['hit_rate'])
        gap = best_h[1]['hit_rate'] - best_g[1]['hit_rate']
        print(f"\n  Best Histology: {best_h[0]} ({best_h[1]['hit_rate']:.4f})")
        print(f"  Best General:   {best_g[0]} ({best_g[1]['hit_rate']:.4f})")
        print(f"  Gap: {gap:+.4f} → {'Histology FM 우세 ✓' if gap > 0.05 else '차이 미미'}")

    return fm_results


# ================================================================
# S2: Nuisance Invariance vs Signal Sensitivity — FM 6개
# ================================================================
def exp_s2_invariance_sensitivity(img_dir, n_samples=200):
    """좋은 FM = nuisance(stain)에 불변, signal(morphology)에 민감."""
    print("\n" + "=" * 70)
    print("S2: Nuisance Invariance vs Signal Sensitivity — FM 6개 비교")
    print("=" * 70)

    import torchvision.transforms as T
    from torchvision.transforms import functional as TF

    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))][:n_samples]
    if not files:
        print("  No images found"); return {}

    nuisance_augs = {
        'color_jitter': T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        'brightness':   T.ColorJitter(brightness=0.5),
        'blur':         T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
    }
    signal_augs = {
        'elastic':    T.ElasticTransform(alpha=50.0),
        'heavy_crop': T.RandomResizedCrop(224, scale=(0.3, 0.5)),
        'rotation':   lambda img: TF.rotate(img, 90),
    }

    results = {}
    for fm_name, (cat, loader, desc) in FM_CATALOG.items():
        print(f"\n--- [{cat}] {fm_name} ---")
        try:
            model, preprocess, feat_dim, encode_fn = loader(DEVICE)
        except Exception as e:
            print(f"  ❌ {e}"); continue

        nuis_sims = defaultdict(list)
        sig_sims = defaultdict(list)

        for fpath in tqdm(files, desc=fm_name, leave=False):
            try: img = Image.open(fpath).convert('RGB')
            except: continue
            with torch.no_grad():
                orig = encode_fn(model, preprocess(img).unsqueeze(0).to(DEVICE))
            for an, af in nuisance_augs.items():
                with torch.no_grad():
                    aug = encode_fn(model, preprocess(af(img)).unsqueeze(0).to(DEVICE))
                nuis_sims[an].append(F.cosine_similarity(orig, aug).item())
            for an, af in signal_augs.items():
                try:
                    with torch.no_grad():
                        aug = encode_fn(model, preprocess(af(img)).unsqueeze(0).to(DEVICE))
                    sig_sims[an].append(F.cosine_similarity(orig, aug).item())
                except: pass

        na = [np.mean(v) for v in nuis_sims.values()] if nuis_sims else [0]
        sa = [np.mean(v) for v in sig_sims.values()] if sig_sims else [0]
        gap = np.mean(na) - np.mean(sa)

        print(f"  Nuisance (↑): {np.mean(na):.4f}")
        for k, v in nuis_sims.items(): print(f"    {k:<20}: {np.mean(v):.4f}±{np.std(v):.4f}")
        print(f"  Signal (↓):   {np.mean(sa):.4f}")
        for k, v in sig_sims.items(): print(f"    {k:<20}: {np.mean(v):.4f}±{np.std(v):.4f}")
        print(f"  Gap (↑):      {gap:.4f}")

        results[fm_name] = {'nuisance': np.mean(na), 'signal': np.mean(sa), 'gap': gap, 'category': cat}
        del model; torch.cuda.empty_cache()

    print(f"\n  {'='*70}")
    print(f"  {'FM':<20} {'Nuisance(↑)':<14} {'Signal(↓)':<14} {'Gap(↑)':<10}")
    print(f"  {'-'*70}")
    for name in FM_CATALOG:
        r = results.get(name)
        if r: print(f"  {name:<20} {r['nuisance']:.4f}        {r['signal']:.4f}        {r['gap']:.4f}")
    print(f"  {'='*70}")
    return results


# ================================================================
# E1: 5-way RAG Ablation
# ================================================================
def exp_e1_rag_ablation(csv_file, result_dirs, n_runs=5):
    """학습 완료된 각 preset 결과 비교."""
    print("\n" + "=" * 70)
    print("E1: 5-way RAG Ablation")
    print("=" * 70)

    df = pd.read_csv(csv_file)
    model, preprocess, feat_dim, encode_fn = load_fm("CONCH", DEVICE)
    results = {}

    for name, path in result_dirs.items():
        if not os.path.exists(path):
            print(f"  ⚠️ {name}: {path} not found"); continue
        print(f"\n  📦 [{name}] {path}")
        feats = extract_slide_features(path, model, preprocess, encode_fn, DEVICE)
        if feats:
            print(f"    {len(feats)} slides, MIL {n_runs} runs...")
            results[name] = run_mil_classification(feats, df, feat_dim, n_runs)

    print_metrics_table(results)
    for a, b, msg in [('full','no_rag','RAG기여'), ('full','random_rag','검색품질'),
                       ('full','baseline','FM+RAG전체')]:
        if a in results and b in results and results[a] and results[b]:
            d = results[a]['AUC'][0] - results[b]['AUC'][0]
            print(f"  {a} vs {b}: ΔAUC={d:+.4f} → {'✓ '+msg if d>0.02 else '✗ 차이없음'}")
    return results


# ================================================================
# E4: Wrong-RAG
# ================================================================
def exp_e4_wrong_rag(csv_file, correct_dir, wrong_dir, n_runs=5):
    print("\n" + "=" * 70)
    print("E4: Wrong-RAG Stress Test")
    print("=" * 70)
    df = pd.read_csv(csv_file)
    model, preprocess, feat_dim, encode_fn = load_fm("CONCH", DEVICE)
    results = {}
    for name, path in [("correct_rag", correct_dir), ("wrong_rag", wrong_dir)]:
        if not os.path.exists(path): continue
        feats = extract_slide_features(path, model, preprocess, encode_fn, DEVICE)
        if feats: results[name] = run_mil_classification(feats, df, feat_dim, n_runs)
    print_metrics_table(results)
    if 'correct_rag' in results and 'wrong_rag' in results and results['correct_rag'] and results['wrong_rag']:
        d = results['correct_rag']['AUC'][0] - results['wrong_rag']['AUC'][0]
        print(f"  → Δ={d:.3f} {'= RAG 작동 ✓' if d > 0.05 else '= ⚠️ RAG 무시?'}")
    return results


# ================================================================
# B1-B4: Bias 분석
# ================================================================
def exp_b1_data_audit(csv_file):
    """데이터 구성 분석."""
    print("\n" + "=" * 70)
    print("B1: Data Composition Audit")
    print("=" * 70)
    df = pd.read_csv(csv_file)
    print(f"\n  Total: {len(df)} slides | Columns: {list(df.columns)}")

    print(f"\n  Split × Label:")
    for split in sorted(df['split'].unique()):
        sub = df[df['split'] == split]
        l0, l1 = (sub['label']==0).sum(), (sub['label']==1).sum()
        print(f"    {split:<8}: {len(sub)} (label0={l0}, label1={l1}, pos_rate={l1/(l0+l1)*100:.1f}%)")

    df['tcga_site'] = df['slide_id'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) and x.startswith('TCGA') and len(x.split('-'))>1 else '?')
    print(f"\n  TCGA sites: {df['tcga_site'].nunique()}")
    sl = df.groupby('tcga_site').agg(n=('label','size'), pos=('label','sum'),
        splits=('split', lambda x: ','.join(sorted(x.unique())))).sort_values('n', ascending=False)
    print(f"  {'Site':<8} {'N':<6} {'Pos':<5} {'Splits'}")
    for s, r in sl.head(15).iterrows():
        print(f"  {s:<8} {r['n']:<6} {int(r['pos']):<5} {r['splits']}")

    df['patient'] = df['slide_id'].apply(
        lambda x: "-".join(x.split('-')[:3]) if isinstance(x, str) and x.startswith('TCGA') else x)
    tr = set(df[df['split']=='train']['patient'])
    te = set(df[df['split'].isin(['test','val'])]['patient'])
    leak = tr & te
    print(f"\n  {'⚠️ LEAK: '+str(len(leak))+' patients!' if leak else '✅ No patient leakage'}")
    return df

def exp_b2_confounder(csv_file):
    """Site → Label confounding."""
    print("\n" + "=" * 70)
    print("B2: Confounder Check")
    print("=" * 70)
    df = pd.read_csv(csv_file)
    df['tcga_site'] = df['slide_id'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) and x.startswith('TCGA') and len(x.split('-'))>1 else '?')
    if df['tcga_site'].nunique() < 2: return
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    site_oh = np.eye(df['tcga_site'].nunique())[le.fit_transform(df['tcga_site'])]
    scores = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                             site_oh, df['label'].values, cv=min(5,len(df)//3), scoring='roc_auc')
    print(f"\n  Site → Label AUC: {scores.mean():.4f}±{scores.std():.4f}")
    if scores.mean() > 0.7: print(f"  ⚠️ STRONG confounding!")
    elif scores.mean() > 0.6: print(f"  ⚠️ Moderate confounding")
    else: print(f"  ✅ Low confounding")

def exp_b3_worst_group(csv_file, img_dir):
    """Site별 성능 분해."""
    print("\n" + "=" * 70)
    print("B3: Worst-Group Evaluation")
    print("=" * 70)
    df = pd.read_csv(csv_file)
    model, preprocess, feat_dim, encode_fn = load_fm("CONCH", DEVICE)
    feats = extract_slide_features(img_dir, model, preprocess, encode_fn, DEVICE)
    if not feats: return
    overall = run_mil_classification(feats, df, feat_dim, n_runs=3)
    if overall: print(f"\n  Overall AUC: {overall['AUC'][0]:.4f}±{overall['AUC'][1]:.4f}")
    df['tcga_site'] = df['slide_id'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) and x.startswith('TCGA') and len(x.split('-'))>1 else '?')
    test_df = df[df['split'].isin(['test','val'])]
    print(f"\n  Test sites:")
    for site, sub in test_df.groupby('tcga_site'):
        print(f"    {site}: {len(sub)} ({sub['label'].sum()} pos)")
    return overall

def exp_b4_rag_bias(fs_dir, ffpe_dir, top_k=5):
    """검색 top-k site 쏠림."""
    print("\n" + "=" * 70)
    print("B4: RAG Bias")
    print("=" * 70)
    model, preprocess, feat_dim, encode_fn = load_fm("CONCH", DEVICE)
    fs_f = extract_patch_features(fs_dir, model, preprocess, encode_fn, DEVICE, 2000)
    ff_f = extract_patch_features(ffpe_dir, model, preprocess, encode_fn, DEVICE, 2000)
    if len(fs_f)<10 or len(ff_f)<10: return
    fs_n, ff_n = list(fs_f.keys()), list(ff_f.keys())
    fs_m, ff_m = torch.stack([fs_f[n] for n in fs_n]), torch.stack([ff_f[n] for n in ff_n])
    def get_site(fn):
        s = fname_to_slide_id(fn).split('-')
        return s[1] if len(s)>1 else '?'
    ff_sites = [get_site(n) for n in ff_n]
    topk = (fs_m @ ff_m.T).topk(top_k, dim=1).indices
    ents = []
    for i in range(len(fs_n)):
        sites = [ff_sites[j] for j in topk[i].tolist()]
        c = defaultdict(int)
        for s in sites: c[s]+=1
        p = np.array(list(c.values()))/top_k
        ents.append(-np.sum(p*np.log(p+1e-10)))
    me = np.log(min(top_k, len(set(ff_sites))))
    br = 1.0 - np.mean(ents)/(me+1e-10)
    print(f"\n  Site entropy: {np.mean(ents):.4f}±{np.std(ents):.4f}")
    print(f"  Bias ratio: {br:.4f} {'⚠️ HIGH' if br>0.7 else '⚠️ Moderate' if br>0.4 else '✅ Low'}")
    return {'entropy': np.mean(ents), 'bias_ratio': br}


# ================================================================
# N1-N4: Noise vs Information
# ================================================================
def exp_n1_artifact_scores(fs_dir, n_samples=500):
    """FS artifact 강도 정량화."""
    print("\n" + "=" * 70)
    print("N1: Artifact Quantification")
    print("=" * 70)
    files = sorted(glob.glob(os.path.join(fs_dir, "*")))
    files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))][:n_samples]
    scores = []
    for fp in tqdm(files, desc="Artifact scoring"):
        try:
            img = np.array(Image.open(fp).convert('RGB')).astype(np.float32)/255.0
            gray = 0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
            fft = np.fft.fftshift(np.fft.fft2(gray)); mag = np.abs(fft)
            h,w = gray.shape; cy,cx = h//2,w//2
            rl,rh = min(h,w)//8, min(h,w)//3
            Y,X = np.ogrid[:h,:w]; dist = np.sqrt((Y-cy)**2+(X-cx)**2)
            hm = (dist>rl)&(dist<rh); te = (mag**2).sum()
            ice = (mag[hm]**2).sum()/(te+1e-8)
            gx,gy = np.diff(gray,axis=1), np.diff(gray,axis=0)
            sharp = (gx**2).mean()+(gy**2).mean()
            tissue = gray<0.85
            snu = np.std([img[:,:,c][tissue].std() for c in range(3)]) if tissue.sum()>100 else 0
            wr = (gray>0.9).mean()
            scores.append({'name':os.path.basename(fp), 'slide_id':fname_to_slide_id(os.path.basename(fp)),
                           'ice_crystal_score':ice, 'sharpness':sharp, 'stain_nonunif':snu, 'white_ratio':wr})
        except: continue
    df = pd.DataFrame(scores)
    print(f"\n  {len(df)} patches")
    for c in ['ice_crystal_score','sharpness','stain_nonunif','white_ratio']:
        print(f"  {c:<22}: {df[c].mean():.4f}±{df[c].std():.4f}")
    return df

def exp_n2_artifact_label(artifact_df, csv_file):
    """Artifact → IDH. Site 통제."""
    print("\n" + "=" * 70)
    print("N2: Artifact → Label (Confounder Control)")
    print("=" * 70)
    dl = pd.read_csv(csv_file)
    ss = artifact_df.groupby('slide_id').agg({
        'ice_crystal_score':'mean','sharpness':'mean','stain_nonunif':'mean','white_ratio':'mean'
    }).reset_index()
    m = pd.merge(ss, dl, on='slide_id', how='inner')
    if len(m)<10 or len(np.unique(m['label']))<2: print("  Not enough data"); return
    fc = ['ice_crystal_score','sharpness','stain_nonunif','white_ratio']
    X, y = m[fc].values, m['label'].values
    sr = cross_val_score(LogisticRegression(max_iter=1000,random_state=42), X, y, cv=min(5,len(m)//3), scoring='roc_auc')
    print(f"\n  Artifact→Label (raw): AUC={sr.mean():.4f}±{sr.std():.4f}")
    m['site'] = m['slide_id'].apply(lambda x: x.split('-')[1] if x.startswith('TCGA') and len(x.split('-'))>1 else '?')
    if m['site'].nunique()>1:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder(); so = np.eye(m['site'].nunique())[le.fit_transform(m['site'])]
        ss_ = cross_val_score(LogisticRegression(max_iter=1000,random_state=42), so, y, cv=min(5,len(m)//3), scoring='roc_auc')
        sb = cross_val_score(LogisticRegression(max_iter=1000,random_state=42), np.hstack([X,so]), y, cv=min(5,len(m)//3), scoring='roc_auc')
        print(f"  Site only:            AUC={ss_.mean():.4f}±{ss_.std():.4f}")
        print(f"  Artifact+Site:        AUC={sb.mean():.4f}±{sb.std():.4f}")
        inc = sb.mean()-ss_.mean()
        print(f"  Incremental: ΔAUC={inc:+.4f} {'⚠️ Bio signal!' if inc>0.05 else '→ Spurious' if sr.mean()>0.6 and inc<0.02 else '→ Weak'}")

def exp_n3_intervention(csv_file, fs_dir, fake_dir, n_runs=3):
    """fake_FFPE + α×residual → IDH."""
    print("\n" + "=" * 70)
    print("N3: Intervention Test")
    print("=" * 70)
    idir = './intervention_images'; os.makedirs(idir, exist_ok=True)
    fs_files = sorted([f for f in glob.glob(os.path.join(fs_dir,"*"))
                       if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))])
    fmap = {}
    for f in glob.glob(os.path.join(fake_dir,"*")):
        bn=os.path.basename(f); c=bn.replace("fake_B_","").replace("fake_","")
        fmap[c]=f; fmap[bn]=f

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for a in alphas:
        ad = os.path.join(idir, f"a{a:.2f}"); os.makedirs(ad, exist_ok=True)
        cnt = 0
        for fp in fs_files[:5000]:
            bn=os.path.basename(fp); c=bn.replace("real_A_","").replace("real_","")
            fk = fmap.get(c) or fmap.get(bn)
            if not fk: continue
            try:
                fi = np.array(Image.open(fp).convert('RGB')).astype(np.float32)/255.0
                ki = np.array(Image.open(fk).convert('RGB')).astype(np.float32)/255.0
                if fi.shape!=ki.shape: continue
                inj = np.clip(ki + a*(fi-ki), 0, 1)
                Image.fromarray((inj*255).astype(np.uint8)).save(os.path.join(ad,c))
                cnt += 1
            except: continue
        print(f"  α={a:.2f}: {cnt} images")

    df = pd.read_csv(csv_file)
    model, preprocess, feat_dim, encode_fn = load_fm("CONCH", DEVICE)
    results = {}
    for a in alphas:
        ad = os.path.join(idir, f"a{a:.2f}")
        feats = extract_slide_features(ad, model, preprocess, encode_fn, DEVICE)
        if feats: results[f"α={a:.2f}"] = run_mil_classification(feats, df, feat_dim, n_runs)
    print_metrics_table(results)

    if "α=0.00" in results and "α=1.00" in results and results["α=0.00"] and results["α=1.00"]:
        a0, a1 = results["α=0.00"]['AUC'][0], results["α=1.00"]['AUC'][0]
        d = a1-a0
        print(f"\n  α=0 (clean): {a0:.4f} → α=1 (FS): {a1:.4f} | Δ={d:+.4f}")
        print(f"  {'→ Bio signal!' if d>0.03 else '→ Noise' if d<-0.03 else '→ Neutral'}")
    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='DeepThaw Proof Experiments')
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('fm_validate')
    p.add_argument('--fs_dir', required=True)
    p.add_argument('--ffpe_dir', required=True)
    p.add_argument('--n_samples', type=int, default=200)

    p = sub.add_parser('rag_ablation')
    p.add_argument('--csv_file', required=True)
    p.add_argument('--result_dirs', nargs='+', required=True, help='name=path')
    p.add_argument('--n_runs', type=int, default=5)

    p = sub.add_parser('wrong_rag')
    p.add_argument('--csv_file', required=True)
    p.add_argument('--correct_dir', required=True)
    p.add_argument('--wrong_dir', required=True)

    p = sub.add_parser('bias_audit')
    p.add_argument('--csv_file', required=True)
    p.add_argument('--fs_dir', default=None)
    p.add_argument('--ffpe_dir', default=None)
    p.add_argument('--fake_ffpe_dir', default=None)

    p = sub.add_parser('noise_vs_info')
    p.add_argument('--csv_file', required=True)
    p.add_argument('--fs_dir', required=True)
    p.add_argument('--fake_ffpe_dir', required=True)

    p = sub.add_parser('all')
    p.add_argument('--csv_file', required=True)
    p.add_argument('--fs_dir', required=True)
    p.add_argument('--ffpe_dir', required=True)
    p.add_argument('--fake_ffpe_dir', required=True)
    p.add_argument('--result_dirs', nargs='*', default=[])

    args = parser.parse_args()

    if args.command == 'fm_validate':
        exp_s1_retrieval_quality(args.fs_dir, args.ffpe_dir)
        exp_s2_invariance_sensitivity(args.fs_dir, args.n_samples)

    elif args.command == 'rag_ablation':
        dirs = dict(i.split('=',1) for i in args.result_dirs)
        exp_e1_rag_ablation(args.csv_file, dirs, args.n_runs)

    elif args.command == 'wrong_rag':
        exp_e4_wrong_rag(args.csv_file, args.correct_dir, args.wrong_dir)

    elif args.command == 'bias_audit':
        exp_b1_data_audit(args.csv_file)
        exp_b2_confounder(args.csv_file)
        if args.fs_dir and args.ffpe_dir: exp_b4_rag_bias(args.fs_dir, args.ffpe_dir)
        if args.fake_ffpe_dir: exp_b3_worst_group(args.csv_file, args.fake_ffpe_dir)

    elif args.command == 'noise_vs_info':
        adf = exp_n1_artifact_scores(args.fs_dir)
        if adf is not None and len(adf)>0: exp_n2_artifact_label(adf, args.csv_file)
        exp_n3_intervention(args.csv_file, args.fs_dir, args.fake_ffpe_dir)

    elif args.command == 'all':
        exp_s1_retrieval_quality(args.fs_dir, args.ffpe_dir)
        exp_s2_invariance_sensitivity(args.fs_dir)
        exp_b1_data_audit(args.csv_file)
        exp_b2_confounder(args.csv_file)
        exp_b4_rag_bias(args.fs_dir, args.ffpe_dir)
        if args.fake_ffpe_dir:
            exp_b3_worst_group(args.csv_file, args.fake_ffpe_dir)
        adf = exp_n1_artifact_scores(args.fs_dir)
        if adf is not None and len(adf)>0: exp_n2_artifact_label(adf, args.csv_file)
        if args.fake_ffpe_dir:
            exp_n3_intervention(args.csv_file, args.fs_dir, args.fake_ffpe_dir)
        if args.result_dirs:
            dirs = dict(i.split('=',1) for i in args.result_dirs)
            exp_e1_rag_ablation(args.csv_file, dirs)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()