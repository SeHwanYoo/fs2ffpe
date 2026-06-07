"""
DeepThaw FM Selection Validation v2 (논문용)
==============================================

Why UNI? — 정량적 근거 + 시각화

실험:
  S1: Cross-Domain Alignment Score (slide-level)
      → FM이 FS↔FFPE 간 같은 조직을 얼마나 가깝게 매핑하나?
  S2: Nuisance Invariance vs Signal Sensitivity
      → FM이 stain variation(무시할 것)과 structural change(감지할 것)을 구분하나?

출력 (figures/):
  fig_augmentation_gallery.png  — 논문 Fig. augmentation 예시
  fig_S1_alignment.png          — Cross-domain alignment score bar chart
  fig_S1_similarity_matrix.png  — FM별 case×case similarity heatmap
  fig_S2_invariance_gap.png     — Nuisance vs Signal bar chart
  fig_S2_per_augmentation.png   — 개별 augmentation별 cosine similarity
  fig_feature_tsne.png          — t-SNE: FS vs FFPE feature space (top 3 FM)
  fig_summary_table.png         — 최종 비교 테이블 (figure로)
  results_S1.csv                — S1 raw numbers
  results_S2.csv                — S2 raw numbers

사용법:
  python proof_experiments_v2.py \
    --fs-dir /path/to/FS/patches \
    --ffpe-dir /path/to/FFPE/patches \
    --output-dir figures \
    --n-samples 200

  # 특정 실험만
  python proof_experiments_v2.py --exp s1 --fs-dir ... --ffpe-dir ...
  python proof_experiments_v2.py --exp s2 --fs-dir ...
  python proof_experiments_v2.py --exp all --fs-dir ... --ffpe-dir ...
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
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# matplotlib 설정 (headless server)
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# 논문 스타일
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MY_HF_TOKEN = os.getenv("HF_TOKEN")

# Color scheme for paper
FM_COLORS = {
    'UNI':          '#2196F3',  # blue (our choice)
    'CONCH':        '#4CAF50',  # green
    'GigaPath':     '#FF9800',  # orange
    'EfficientNet': '#9C27B0',  # purple
    'ViT-B/16':     '#F44336',  # red
    'Resnet':   '#9E9E9E',  # gray
}

FM_CATEGORIES = {
    'UNI':          'Pathology FM',
    'CONCH':        'Pathology FM',
    'GigaPath':     'Pathology FM',
    'EfficientNet': 'General FM',
    'ViT-B/16':     'General FM',
    'Resnet':   'General FM',
}

# FM 순서 (논문 표시 순서)
FM_ORDER = ['UNI', 'CONCH', 'GigaPath', 'EfficientNet', 'ViT-B/16', 'Resnet']


# ================================================================
# 0. FM 로더 (6개)
# ================================================================
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
    feat = model(batch)
    return F.normalize(feat, dim=1)

def _conch_encode(model, batch):
    feat = model.encode_image(batch, proj_contrast=False, normalize=True)
    return feat

def load_conch(device):
    custom_lib = '/home/users/astar/ares/yoosehwa/scratch/my_libs'
    if os.path.exists(custom_lib) and custom_lib not in sys.path:
        sys.path.insert(0, custom_lib)
    _hf_login()
    from conch.open_clip_custom import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16', "hf_hub:MahmoodLab/CONCH",
        device=device, hf_auth_token=MY_HF_TOKEN)
    model.eval()
    return model, preprocess, 512, _conch_encode

def load_gigapath(device):
    _hf_login()
    import timm
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 1536, _generic_encode

def load_uni(device):
    _hf_login()
    import timm
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI", pretrained=True,
        init_values=1e-5, dynamic_img_size=True)
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 1024, _generic_encode

def load_efficientnet(device):
    import torchvision.models as models
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier = nn.Identity()
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 1280, _generic_encode

def load_vit(device):
    import torchvision.models as models
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = nn.Identity()
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 768, _generic_encode

def load_random_vit(device):
    import torchvision.models as models
    # model = models.vit_b_16(weights=None)
    # model.heads = nn.Identity()
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Identity()
    model = model.to(device).eval()
    return model, _imagenet_preprocess(), 2048, _generic_encode


FM_LOADERS = {
    'UNI':          load_uni,
    'CONCH':        load_conch,
    'GigaPath':     load_gigapath,
    'EfficientNet': load_efficientnet,
    'ViT-B/16':     load_vit,
    'Resnet':   load_random_vit,
}


# ================================================================
# 1. Data Loading
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


def fname_to_case_id(fname):
    """TCGA-XX-XXXX-01Z-00-DX1_x_y.png → TCGA-XX-XXXX"""
    name = fname.replace("fake_B_", "").replace("fake_", "").replace("real_A_", "").replace("real_", "")
    parts = name.split('-')
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return name.split('_')[0]


def get_image_files(img_dir, max_files=None):
    """이미지 파일 리스트. Case별 균등 샘플링."""
    import random as _random

    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    if max_files and len(files) > max_files:
        # Case별 균등 샘플링 (sorted()[:N] → 앞쪽 case만 잡히는 문제 방지)
        case_files = defaultdict(list)
        for f in files:
            cid = fname_to_case_id(os.path.basename(f))
            case_files[cid].append(f)

        # Round-robin: 각 case에서 골고루 뽑기
        n_cases = len(case_files)
        per_case = max(1, max_files // n_cases)
        sampled = []
        rng = _random.Random(42)
        for cid, cfiles in case_files.items():
            rng.shuffle(cfiles)
            sampled.extend(cfiles[:per_case])

        # 아직 부족하면 남은 것에서 추가
        if len(sampled) < max_files:
            remaining = [f for f in files if f not in set(sampled)]
            rng.shuffle(remaining)
            sampled.extend(remaining[:max_files - len(sampled)])

        files = sampled[:max_files]
        print(f"  Sampled {len(files)} patches from {n_cases} cases (balanced)")

    return files


def extract_features_by_case(img_dir, model, preprocess, encode_fn, device,
                              max_patches=5000, batch_size=256):
    """
    Case별 feature 추출.
    Returns: {case_id: {'features': (N, D) tensor, 'mean': (D,) tensor, 'files': [str]}}
    """
    files = get_image_files(img_dir, max_patches)
    if not files:
        return {}

    loader = DataLoader(PatchDataset(files, preprocess),
                        batch_size=batch_size, num_workers=8, pin_memory=True)

    case_data = defaultdict(lambda: {'features': [], 'files': []})
    with torch.no_grad():
        for imgs, fnames in tqdm(loader, desc="  Extracting", leave=False):
            feats = encode_fn(model, imgs.to(device)).cpu()
            for i, f in enumerate(fnames):
                if f == "error": continue
                cid = fname_to_case_id(f)
                case_data[cid]['features'].append(feats[i])
                case_data[cid]['files'].append(f)

    # Stack and compute means
    result = {}
    for cid, data in case_data.items():
        feat_stack = torch.stack(data['features'])
        result[cid] = {
            'features': feat_stack,
            'mean': F.normalize(feat_stack.mean(dim=0, keepdim=True), dim=1).squeeze(0),
            'files': data['files'],
            'n_patches': len(data['features']),
        }
    return result


# ================================================================
# S1: Cross-Domain Alignment Score (REDESIGNED)
# ================================================================
def exp_s1_alignment(fs_dir, ffpe_dir, output_dir, max_patches=5000):
    """
    Slide-level cross-domain alignment.

    각 FM이 FS↔FFPE 간 "같은 조직"을 feature space에서 얼마나 가깝게 매핑하는지 측정.

    Metrics:
      1. Intra-case similarity: cos_sim(mean_FS_i, mean_FFPE_i) — 같은 case
      2. Inter-case similarity: cos_sim(mean_FS_i, mean_FFPE_j) — 다른 case
      3. Alignment score = intra - inter (높을수록 좋음)
      4. Retrieval Rank: mean reciprocal rank (MRR) at slide level
    """
    print("\n" + "=" * 70)
    print("S1: Cross-Domain Alignment Score (Slide-Level)")
    print("=" * 70)

    results = {}

    for fm_name in FM_ORDER:
        loader_fn = FM_LOADERS[fm_name]
        cat = FM_CATEGORIES[fm_name]
        print(f"\n--- [{cat}] {fm_name} ---")

        try:
            model, preprocess, feat_dim, encode_fn = loader_fn(DEVICE)
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            results[fm_name] = None
            continue

        # Extract features grouped by case
        print("  FS features...")
        fs_cases = extract_features_by_case(
            fs_dir, model, preprocess, encode_fn, DEVICE, max_patches)
        print("  FFPE features...")
        ffpe_cases = extract_features_by_case(
            ffpe_dir, model, preprocess, encode_fn, DEVICE, max_patches)

        # Find common cases
        common = sorted(set(fs_cases.keys()) & set(ffpe_cases.keys()))
        print(f"  Common cases: {len(common)} (FS={len(fs_cases)}, FFPE={len(ffpe_cases)})")

        if len(common) < 3:
            print("  ⚠️ Not enough common cases")
            results[fm_name] = None
            del model; torch.cuda.empty_cache()
            continue

        # Slide-level mean features
        fs_means = torch.stack([fs_cases[c]['mean'] for c in common])     # (N_cases, D)
        ffpe_means = torch.stack([ffpe_cases[c]['mean'] for c in common]) # (N_cases, D)

        # Cross-domain similarity matrix: (N_cases, N_cases)
        sim_matrix = (fs_means @ ffpe_means.T).numpy()  # sim[i,j] = cos(FS_i, FFPE_j)
        n = len(common)

        # 1. Intra-case similarity (diagonal)
        intra_sims = np.diag(sim_matrix)
        mean_intra = np.mean(intra_sims)

        # 2. Inter-case similarity (off-diagonal)
        mask = ~np.eye(n, dtype=bool)
        inter_sims = sim_matrix[mask]
        mean_inter = np.mean(inter_sims)

        # 3. Alignment score
        alignment = mean_intra - mean_inter

        # 4. Mean Reciprocal Rank (MRR)
        #    For each FS_i, rank FFPE candidates by similarity. Where does correct FFPE_i land?
        mrr_values = []
        for i in range(n):
            row = sim_matrix[i]
            rank = np.sum(row >= row[i])  # rank 1 = best
            mrr_values.append(1.0 / rank)
        mrr = np.mean(mrr_values)

        # 5. Top-1 accuracy (slide-level)
        top1_correct = np.sum(np.argmax(sim_matrix, axis=1) == np.arange(n))
        top1_acc = top1_correct / n

        results[fm_name] = {
            'intra': mean_intra,
            'inter': mean_inter,
            'alignment': alignment,
            'mrr': mrr,
            'top1_acc': top1_acc,
            'n_cases': n,
            'sim_matrix': sim_matrix,
            'case_ids': common,
            'intra_sims': intra_sims,
        }

        print(f"  Intra-case sim:  {mean_intra:.4f}")
        print(f"  Inter-case sim:  {mean_inter:.4f}")
        print(f"  Alignment score: {alignment:.4f}")
        print(f"  MRR:             {mrr:.4f}")
        print(f"  Top-1 accuracy:  {top1_acc:.4f} ({top1_correct}/{n})")

        del model, fs_cases, ffpe_cases
        torch.cuda.empty_cache()

    # ---- Summary Table ----
    print(f"\n  {'='*80}")
    print(f"  {'FM':<15} {'Cat':<14} {'Intra↑':<10} {'Inter↓':<10} {'Align↑':<10} {'MRR↑':<8} {'Top1↑':<8}")
    print(f"  {'-'*80}")
    for fm in FM_ORDER:
        r = results.get(fm)
        if r:
            print(f"  {fm:<15} {FM_CATEGORIES[fm]:<14} {r['intra']:.4f}    "
                  f"{r['inter']:.4f}    {r['alignment']:.4f}    "
                  f"{r['mrr']:.4f}  {r['top1_acc']:.4f}")
    print(f"  {'='*80}")

    # ---- Figures ----
    _plot_s1_alignment_bar(results, output_dir)
    _plot_s1_similarity_matrices(results, output_dir)

    # ---- CSV ----
    rows = []
    for fm in FM_ORDER:
        r = results.get(fm)
        if r:
            rows.append({
                'FM': fm, 'Category': FM_CATEGORIES[fm],
                'Intra_sim': r['intra'], 'Inter_sim': r['inter'],
                'Alignment': r['alignment'], 'MRR': r['mrr'],
                'Top1_acc': r['top1_acc'], 'N_cases': r['n_cases'],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'results_S1.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return results


def _plot_s1_alignment_bar(results, output_dir):
    """S1: Alignment score bar chart."""
    fms = [fm for fm in FM_ORDER if results.get(fm)]
    if not fms:
        print("  ⚠️ No valid results for alignment bar plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Alignment Score
    ax = axes[0]
    vals = [results[fm]['alignment'] for fm in fms]
    colors = [FM_COLORS[fm] for fm in fms]
    bars = ax.bar(range(len(fms)), vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(fms)))
    ax.set_xticklabels(fms, rotation=35, ha='right')
    ax.set_ylabel('Alignment Score')
    ax.set_title('(a) Cross-Domain Alignment')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # Highlight best
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)

    # (b) MRR
    ax = axes[1]
    vals = [results[fm]['mrr'] for fm in fms]
    bars = ax.bar(range(len(fms)), vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(fms)))
    ax.set_xticklabels(fms, rotation=35, ha='right')
    ax.set_ylabel('Mean Reciprocal Rank')
    ax.set_title('(b) Slide-Level Retrieval (MRR)')
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)

    # (c) Intra vs Inter
    ax = axes[2]
    x = np.arange(len(fms))
    w = 0.35
    intra = [results[fm]['intra'] for fm in fms]
    inter = [results[fm]['inter'] for fm in fms]
    ax.bar(x - w/2, intra, w, label='Intra-case (↑)', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, inter, w, label='Inter-case (↓)', color='#FF5722', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fms, rotation=35, ha='right')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(c) Intra vs Inter-Case Similarity')
    ax.legend(loc='lower right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_S1_alignment.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def _plot_s1_similarity_matrices(results, output_dir):
    """S1: Case×Case similarity heatmaps (top 3 FM)."""
    # Pick top 3 by alignment score
    scored = [(fm, results[fm]['alignment']) for fm in FM_ORDER if results.get(fm)]
    scored.sort(key=lambda x: -x[1])
    top3 = [fm for fm, _ in scored[:3]]

    if not top3:
        print("  ⚠️ No valid results for similarity matrix plot")
        return

    fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 4.5))
    if len(top3) == 1: axes = [axes]

    for idx, fm in enumerate(top3):
        ax = axes[idx]
        sim = results[fm]['sim_matrix']
        n = sim.shape[0]

        # Sort by diagonal value for cleaner visualization
        diag_order = np.argsort(-np.diag(sim))
        sim_sorted = sim[diag_order][:, diag_order]

        im = ax.imshow(sim_sorted, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
        ax.set_title(f'{fm}\n(align={results[fm]["alignment"]:.3f})', fontsize=11)
        ax.set_xlabel('FFPE cases')
        if idx == 0:
            ax.set_ylabel('FS cases')

        # Diagonal line
        ax.plot([-0.5, n-0.5], [-0.5, n-0.5], 'k--', linewidth=0.5, alpha=0.3)

    plt.colorbar(im, ax=axes[-1], label='Cosine Similarity', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_S1_similarity_matrix.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# S2: Nuisance Invariance vs Signal Sensitivity
# ================================================================
def exp_s2_invariance(fs_dir, output_dir, n_samples=200):
    """
    Good FM: nuisance(stain variation) → high similarity (invariant)
             signal(structural change) → low similarity (sensitive)
    Gap = nuisance_sim - signal_sim → 높을수록 좋음
    """
    print("\n" + "=" * 70)
    print("S2: Nuisance Invariance vs Signal Sensitivity")
    print("=" * 70)

    import torchvision.transforms as T
    from torchvision.transforms import functional as TF

    files = get_image_files(fs_dir, n_samples)
    if not files:
        print("  No images!"); return {}

    # Augmentation 정의
    nuisance_augs = {
        'Color Jitter':  T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        'Brightness':    T.ColorJitter(brightness=0.5),
        'Gaussian Blur': T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
    }
    signal_augs = {
        'Elastic Deform': T.ElasticTransform(alpha=50.0),
        'Heavy Crop':     T.RandomResizedCrop(224, scale=(0.3, 0.5)),
        'Rotation 90°':   lambda img: TF.rotate(img, 90),
    }

    # ---- Figure: Augmentation Gallery ----
    _plot_augmentation_gallery(files[0], nuisance_augs, signal_augs, output_dir)

    # ---- Run experiments ----
    all_results = {}  # {fm_name: {aug_name: [sim_values]}}
    summary = {}

    for fm_name in FM_ORDER:
        loader_fn = FM_LOADERS[fm_name]
        cat = FM_CATEGORIES[fm_name]
        print(f"\n--- [{cat}] {fm_name} ---")

        try:
            model, preprocess, feat_dim, encode_fn = loader_fn(DEVICE)
        except Exception as e:
            print(f"  ❌ {e}"); continue

        nuis_sims = defaultdict(list)
        sig_sims = defaultdict(list)

        for fpath in tqdm(files, desc=f"  {fm_name}", leave=False):
            try:
                img = Image.open(fpath).convert('RGB')
            except:
                continue

            with torch.no_grad():
                orig = encode_fn(model, preprocess(img).unsqueeze(0).to(DEVICE))

            for aug_name, aug_fn in nuisance_augs.items():
                try:
                    with torch.no_grad():
                        aug = encode_fn(model, preprocess(aug_fn(img)).unsqueeze(0).to(DEVICE))
                    nuis_sims[aug_name].append(F.cosine_similarity(orig, aug).item())
                except: pass

            for aug_name, aug_fn in signal_augs.items():
                try:
                    with torch.no_grad():
                        aug = encode_fn(model, preprocess(aug_fn(img)).unsqueeze(0).to(DEVICE))
                    sig_sims[aug_name].append(F.cosine_similarity(orig, aug).item())
                except: pass

        # Compute summary
        nuis_means = {k: np.mean(v) for k, v in nuis_sims.items()}
        sig_means = {k: np.mean(v) for k, v in sig_sims.items()}
        nuis_stds = {k: np.std(v) for k, v in nuis_sims.items()}
        sig_stds = {k: np.std(v) for k, v in sig_sims.items()}

        overall_nuis = np.mean(list(nuis_means.values())) if nuis_means else 0
        overall_sig = np.mean(list(sig_means.values())) if sig_means else 0
        gap = overall_nuis - overall_sig

        print(f"  Nuisance (↑): {overall_nuis:.4f}")
        for k in nuisance_augs:
            if k in nuis_means:
                print(f"    {k:<20}: {nuis_means[k]:.4f} ± {nuis_stds.get(k,0):.4f}")
        print(f"  Signal (↓):   {overall_sig:.4f}")
        for k in signal_augs:
            if k in sig_means:
                print(f"    {k:<20}: {sig_means[k]:.4f} ± {sig_stds.get(k,0):.4f}")
        print(f"  Gap (↑):      {gap:.4f}")

        all_results[fm_name] = {
            'nuisance': {k: {'mean': nuis_means[k], 'std': nuis_stds[k], 'values': nuis_sims[k]}
                         for k in nuis_sims},
            'signal': {k: {'mean': sig_means[k], 'std': sig_stds[k], 'values': sig_sims[k]}
                       for k in sig_sims},
        }
        summary[fm_name] = {
            'nuisance': overall_nuis,
            'signal': overall_sig,
            'gap': gap,
            'category': cat,
        }

        del model; torch.cuda.empty_cache()

    # ---- Summary Table ----
    print(f"\n  {'='*70}")
    print(f"  {'FM':<15} {'Cat':<14} {'Nuisance(↑)':<14} {'Signal(↓)':<14} {'Gap(↑)':<10}")
    print(f"  {'-'*70}")
    for fm in FM_ORDER:
        r = summary.get(fm)
        if r:
            marker = " ★" if fm == max(summary, key=lambda x: summary[x]['gap']) else ""
            print(f"  {fm:<15} {r['category']:<14} {r['nuisance']:.4f}        "
                  f"{r['signal']:.4f}        {r['gap']:.4f}{marker}")
    print(f"  {'='*70}")

    # ---- Figures ----
    _plot_s2_gap_bar(summary, output_dir)
    _plot_s2_per_augmentation(all_results, nuisance_augs, signal_augs, output_dir)

    # ---- CSV ----
    rows = []
    for fm in FM_ORDER:
        r = summary.get(fm)
        if r:
            row = {'FM': fm, 'Category': r['category'],
                   'Nuisance_mean': r['nuisance'], 'Signal_mean': r['signal'],
                   'Gap': r['gap']}
            # Per-augmentation details
            ar = all_results.get(fm, {})
            for aug_name in nuisance_augs:
                if aug_name in ar.get('nuisance', {}):
                    row[f'nuis_{aug_name}'] = ar['nuisance'][aug_name]['mean']
            for aug_name in signal_augs:
                if aug_name in ar.get('signal', {}):
                    row[f'sig_{aug_name}'] = ar['signal'][aug_name]['mean']
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'results_S2.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return summary, all_results


def _plot_augmentation_gallery(sample_path, nuisance_augs, signal_augs, output_dir):
    """논문 Figure: Augmentation 예시 갤러리."""
    import torchvision.transforms as T

    try:
        img = Image.open(sample_path).convert('RGB')
    except:
        print("  ⚠️ Cannot open sample for gallery"); return

    all_augs = {}
    all_augs['Original'] = img
    for name, fn in nuisance_augs.items():
        try:
            all_augs[f'[N] {name}'] = fn(img)
        except: pass
    for name, fn in signal_augs.items():
        try:
            result = fn(img)
            if isinstance(result, torch.Tensor):
                result = T.ToPILImage()(result)
            all_augs[f'[S] {name}'] = result
        except: pass

    n = len(all_augs)
    ncols = min(n, 7)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3 * nrows))
    if nrows == 1: axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for idx, (name, augimg) in enumerate(all_augs.items()):
        ax = axes_flat[idx]
        if isinstance(augimg, torch.Tensor):
            augimg = T.ToPILImage()(augimg)
        ax.imshow(augimg)
        ax.set_title(name, fontsize=9, fontweight='bold' if 'Original' in name else 'normal')
        ax.axis('off')

        # Color-code border: blue for nuisance, red for signal
        if '[N]' in name:
            for spine in ax.spines.values():
                spine.set_edgecolor('#2196F3'); spine.set_linewidth(3); spine.set_visible(True)
        elif '[S]' in name:
            for spine in ax.spines.values():
                spine.set_edgecolor('#F44336'); spine.set_linewidth(3); spine.set_visible(True)

    # Hide extra axes
    for idx in range(len(all_augs), len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('Augmentation Types:  Blue = Nuisance (stain)  |  Red = Signal (structure)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_augmentation_gallery.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_s2_gap_bar(summary, output_dir):
    """S2: Gap bar chart (nuisance - signal)."""
    fms = [fm for fm in FM_ORDER if fm in summary]
    if not fms:
        print("  ⚠️ No valid results for S2 gap bar plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Grouped bar: Nuisance vs Signal
    ax = axes[0]
    x = np.arange(len(fms))
    w = 0.35
    nuis = [summary[fm]['nuisance'] for fm in fms]
    sig = [summary[fm]['signal'] for fm in fms]
    ax.bar(x - w/2, nuis, w, label='Nuisance (stain) ↑', color='#4CAF50', alpha=0.85)
    ax.bar(x + w/2, sig, w, label='Signal (structure) ↓', color='#F44336', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(fms, rotation=35, ha='right')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Nuisance Invariance vs Signal Sensitivity')
    ax.legend()
    ax.set_ylim(0.6, 1.05)

    # (b) Gap
    ax = axes[1]
    gaps = [summary[fm]['gap'] for fm in fms]
    colors = [FM_COLORS.get(fm, '#999') for fm in fms]
    bars = ax.bar(range(len(fms)), gaps, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(fms)))
    ax.set_xticklabels(fms, rotation=35, ha='right')
    ax.set_ylabel('Gap (Nuisance − Signal)')
    ax.set_title('(b) Discrimination Gap ↑')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Highlight best
    best_idx = np.argmax(gaps)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_S2_invariance_gap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def _plot_s2_per_augmentation(all_results, nuisance_augs, signal_augs, output_dir):
    """S2: Per-augmentation detail chart."""
    fms = [fm for fm in FM_ORDER if fm in all_results]
    if not fms:
        print("  ⚠️ No valid results for per-augmentation plot")
        return

    all_aug_names = list(nuisance_augs.keys()) + list(signal_augs.keys())
    n_augs = len(all_aug_names)
    n_fms = len(fms)

    fig, ax = plt.subplots(figsize=(max(10, n_fms * 1.5), 5))

    x = np.arange(n_augs)
    bar_width = 0.8 / n_fms

    for i, fm in enumerate(fms):
        ar = all_results[fm]
        means = []
        stds = []
        for aug_name in all_aug_names:
            if aug_name in ar.get('nuisance', {}):
                means.append(ar['nuisance'][aug_name]['mean'])
                stds.append(ar['nuisance'][aug_name]['std'])
            elif aug_name in ar.get('signal', {}):
                means.append(ar['signal'][aug_name]['mean'])
                stds.append(ar['signal'][aug_name]['std'])
            else:
                means.append(0); stds.append(0)

        offset = (i - n_fms/2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=fm, color=FM_COLORS.get(fm, '#999'),
               alpha=0.85, capsize=2)

    # Separator line between nuisance and signal
    sep = len(nuisance_augs) - 0.5
    ax.axvline(x=sep, color='gray', linestyle='--', linewidth=1)
    ax.text(sep - 0.3, ax.get_ylim()[1] * 0.98, '← Nuisance', ha='right', fontsize=9, color='#4CAF50')
    ax.text(sep + 0.3, ax.get_ylim()[1] * 0.98, 'Signal →', ha='left', fontsize=9, color='#F44336')

    ax.set_xticks(x)
    ax.set_xticklabels(all_aug_names, rotation=35, ha='right')
    ax.set_ylabel('Cosine Similarity with Original')
    ax.set_title('Per-Augmentation Similarity by FM')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0.6, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_S2_per_augmentation.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# Feature Space t-SNE Visualization
# ================================================================
def exp_tsne_visualization(fs_dir, ffpe_dir, output_dir, max_patches=500):
    """t-SNE: FS vs FFPE feature space for top 3 FMs."""
    print("\n" + "=" * 70)
    print("t-SNE: Feature Space Visualization")
    print("=" * 70)

    from sklearn.manifold import TSNE

    # vis_fms = ['UNI', 'CONCH', 'Random ViT']
    vis_fms = ['UNI', 'CONCH', 'GigaPath', 'EfficientNet', 'ViT-B/16', 'Resnet']

    fig, axes = plt.subplots(1, len(vis_fms), figsize=(6 * len(vis_fms), 5))
    if len(vis_fms) == 1: axes = [axes]

    for idx, fm_name in enumerate(vis_fms):
        print(f"\n  {fm_name}...")
        
        safe_fm_name = fm_name.replace("/", "_")
        
        loader_fn = FM_LOADERS[fm_name]

        try:
            model, preprocess, feat_dim, encode_fn = loader_fn(DEVICE)
        except Exception as e:
            print(f"  ❌ {e}"); continue

        # Extract
        fs_feats = extract_features_by_case(
            fs_dir, model, preprocess, encode_fn, DEVICE, max_patches=max_patches)
        ffpe_feats = extract_features_by_case(
            ffpe_dir, model, preprocess, encode_fn, DEVICE, max_patches=max_patches)

        common = sorted(set(fs_feats.keys()) & set(ffpe_feats.keys()))
        if len(common) < 3:
            print(f"  Not enough cases"); continue

        # Sample patches for visualization (max 30 per case, max 10 cases)
        cases_to_show = common[:min(10, len(common))]
        all_feats = []
        all_domains = []  # 'FS' or 'FFPE'
        all_cases = []

        for cid in cases_to_show:
            fs_f = fs_feats[cid]['features'][:30]
            ffpe_f = ffpe_feats[cid]['features'][:30]
            all_feats.extend(fs_f.numpy())
            all_domains.extend(['FS'] * len(fs_f))
            all_cases.extend([cid] * len(fs_f))
            all_feats.extend(ffpe_f.numpy())
            all_domains.extend(['FFPE'] * len(ffpe_f))
            all_cases.extend([cid] * len(ffpe_f))

        all_feats = np.array(all_feats)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats)-1))
        emb = tsne.fit_transform(all_feats)
        
        # ========================================================
        # 💾 [데이터 저장 파트 추가] 그래프 그리기 전에 좌표와 특징을 저장!
        # ========================================================
        # 1. 원본 N차원 특징(Feature) 벡터 저장 (.npy 형식)
        npy_path = os.path.join(output_dir, f'raw_features_{safe_fm_name}.npy')
        # 나중에 파일 불러올 때 환자/도메인 정보를 알기 위해 딕셔너리로 묶어서 저장해!
        np.save(npy_path, {
            'features': all_feats, 
            'domains': all_domains, 
            'cases': all_cases
        })
        print(f"  💾 Saved Raw Features: {npy_path}")

        # 2. t-SNE 2차원 축소 좌표 저장 (.csv 형식)
        csv_path = os.path.join(output_dir, f'tsne_coords_{safe_fm_name}.csv')
        df_tsne = pd.DataFrame({
            'tsne_1': emb[:, 0],
            'tsne_2': emb[:, 1],
            'Domain': all_domains,
            'Case_ID': all_cases
        })
        df_tsne.to_csv(csv_path, index=False)
        print(f"  💾 Saved t-SNE Coords: {csv_path}")
        # ========================================================
        

        ax = axes[idx]
        case_colors = plt.cm.tab10(np.linspace(0, 1, len(cases_to_show)))

        for ci, cid in enumerate(cases_to_show):
            mask_fs = [(d == 'FS' and c == cid) for d, c in zip(all_domains, all_cases)]
            mask_ffpe = [(d == 'FFPE' and c == cid) for d, c in zip(all_domains, all_cases)]

            ax.scatter(emb[mask_fs, 0], emb[mask_fs, 1],
                      c=[case_colors[ci]], marker='o', s=15, alpha=0.6,
                      label=f'{cid[-4:]} FS' if ci == 0 else None)
            ax.scatter(emb[mask_ffpe, 0], emb[mask_ffpe, 1],
                      c=[case_colors[ci]], marker='^', s=15, alpha=0.6,
                      label=f'{cid[-4:]} FFPE' if ci == 0 else None)

        ax.set_title(f'{fm_name}', fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])

        # Legend: just show FS=circle, FFPE=triangle
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='FS'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='FFPE'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        del model; torch.cuda.empty_cache()

    fig.suptitle('t-SNE: FS vs FFPE Feature Space (colors = cases)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_feature_tsne.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# Summary Figure (Table as Figure)
# ================================================================
def create_summary_figure(s1_results, s2_summary, output_dir):
    """최종 요약 테이블을 figure로."""
    fms = [fm for fm in FM_ORDER if fm in (s2_summary or {})]
    if not fms:
        print("  ⚠️ No valid results for summary figure")
        return

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    # Build table data
    columns = ['FM', 'Category', 'S1\nAlignment↑', 'S1\nMRR↑', 'S2\nNuisance↑', 'S2\nSignal↓', 'S2\nGap↑']
    cell_data = []
    cell_colors = []

    # Find best values for highlighting
    s1_aligns = [s1_results[fm]['alignment'] for fm in fms if s1_results and s1_results.get(fm)]
    s2_gaps = [s2_summary[fm]['gap'] for fm in fms if s2_summary.get(fm)]
    best_align = max(s1_aligns) if s1_aligns else 0
    best_gap = max(s2_gaps) if s2_gaps else 0

    for fm in fms:
        s1 = s1_results.get(fm, {}) if s1_results else {}
        s2 = s2_summary.get(fm, {})

        row = [
            fm,
            FM_CATEGORIES.get(fm, ''),
            f"{s1['alignment']:.4f}" if s1 else '—',
            f"{s1['mrr']:.4f}" if s1 else '—',
            f"{s2['nuisance']:.4f}" if s2 else '—',
            f"{s2['signal']:.4f}" if s2 else '—',
            f"{s2['gap']:.4f}" if s2 else '—',
        ]
        cell_data.append(row)

        # Highlight row colors
        colors = ['#ffffff'] * len(columns)
        if s1 and s1.get('alignment') == best_align:
            colors[2] = '#E3F2FD'
        if s2 and s2.get('gap') == best_gap:
            colors[6] = '#E3F2FD'
        cell_colors.append(colors)

    table = ax.table(cellText=cell_data, colLabels=columns, cellColours=cell_colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header styling
    for j in range(len(columns)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Foundation Model Selection: Quantitative Comparison', fontsize=13, pad=20)

    path = os.path.join(output_dir, 'fig_summary_table.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='DeepThaw FM Validation v2 (Paper)')
    parser.add_argument('--fs-dir', type=str, required=True,
                        help='FS patch 디렉토리')
    parser.add_argument('--ffpe-dir', type=str, required=True,
                        help='FFPE patch 디렉토리')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Figure/CSV 출력 디렉토리')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['s1', 's2', 'tsne', 'all'],
                        help='실행할 실험')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='S2 sample 수')
    parser.add_argument('--max-patches', type=int, default=5000,
                        help='S1 max patches per domain')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"DeepThaw FM Validation v2")
    print(f"  FS:     {args.fs_dir}")
    print(f"  FFPE:   {args.ffpe_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {DEVICE}")

    s1_results = None
    s2_summary = None

    if args.exp in ('s1', 'all'):
        s1_results = exp_s1_alignment(
            args.fs_dir, args.ffpe_dir, args.output_dir,
            max_patches=args.max_patches)

    if args.exp in ('s2', 'all'):
        s2_summary, s2_details = exp_s2_invariance(
            args.fs_dir, args.output_dir, n_samples=args.n_samples)

    if args.exp in ('tsne', 'all'):
        exp_tsne_visualization(
            args.fs_dir, args.ffpe_dir, args.output_dir,
            max_patches=500)

    if s1_results or s2_summary:
        create_summary_figure(s1_results, s2_summary, args.output_dir)

    # ---- Final Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if s2_summary:
        best_fm = max(s2_summary, key=lambda x: s2_summary[x]['gap'])
        print(f"\n  S2 Best (invariance gap): {best_fm} (gap={s2_summary[best_fm]['gap']:.4f})")

    if s1_results:
        valid = {k: v for k, v in s1_results.items() if v}
        if valid:
            best_s1 = max(valid, key=lambda x: valid[x]['alignment'])
            print(f"  S1 Best (alignment):      {best_s1} (align={valid[best_s1]['alignment']:.4f})")

    print(f"\n  → Selected FM for DeepThaw perceptual loss: UNI")
    print(f"\n  Output: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"    {f}")

    print("\nDone! ✅")


if __name__ == '__main__':
    main()