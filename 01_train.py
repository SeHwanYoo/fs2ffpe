"""
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 200 * (1024**2)
DeepThaw v2: FS→FFPE Training (UNI Perceptual Loss)
======================================================

사용법:
    python train_fs2ffpe_v2.py --preset baseline --data-path /path/to/FS2FFPE
    python train_fs2ffpe_v2.py --preset uni-full --data-path /path/to/FS2FFPE
    python train_fs2ffpe_v2.py --preset uni-full --data-path /path --data-percent 30
    python train_fs2ffpe_v2.py --preset uni-full --data-path /path --rag-wrong
"""

import argparse
import os
import sys
import shutil
import random

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.presets import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser


# ============================================================
# Presets
# ============================================================
PRESETS = {
    'baseline': {
        'model': 'uvcgan2',
    },
    'uni': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': False,
        'use_rag': False,
    },
    'sc-only': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': False,
        'use_self_challenging': True,
        'use_rag': False,
    },
    'rag-only': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': False,
        'use_self_challenging': False,
        'use_rag': True,
    },
    'uni-sc': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': True,
        'use_rag': False,
    },
    'uni-rag': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': False,
        'use_rag': True,
    },
    'uni-rag-pixel': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': False,
        'use_rag': True,
        'rag_mode': 'pixel',
    },
    'uni-rag-hybrid': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': False,
        'use_rag': True,
        'rag_mode': 'hybrid',
    },
    'uni-crag': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': False,
        'use_rag': True,
        'rag_mode': 'contrastive',
    },
    'uni-crag-input': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': False,
        'use_rag': True,
        'rag_mode': 'input',
    },
    'uni-full': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': True,
        'use_rag': True,
    },
    'full': {
        'model': 'uvcgan2-deepthaw',
        'use_uni_loss': True,
        'use_self_challenging': True,
        'use_rag': True,
    },
}


# ============================================================
# Environment
# ============================================================
def detect_env():
    envs = [
        ('/home/users/astar/ares/yoosehwa', 'NSCC',
         '/home/users/astar/ares/yoosehwa/scratch/dataset/brain/images/linked/FS2FFPE',
         '/home/users/astar/ares/yoosehwa/scratch/dataset/brain/latent/linked/FS2FFPE/rag_cache'),
        ('/home/sehwan001', 'NEW_SERVER', None, None),
        ('/home/ntu/Desktop/Sehwan', 'NTU',
         '/home/ntu/Desktop/Sehwan/datasets/linked_v2/FS2FFPE', None),
        ('/home/ivpl-d29/Sehwan_Kim', 'LAB',
         '/home/ivpl-d29/Sehwan_Kim/datasets/linked/FS2FFPE', None),
    ]
    for check, name, data, rag in envs:
        if os.path.exists(check):
            return {'name': name, 'data': data, 'rag': rag}
    return {'name': 'UNKNOWN', 'data': None, 'rag': None}


# ============================================================
# Data Subset (for --data-percent)
# ============================================================
def create_data_subset(data_path, percent, label):
    """trainA/trainB에서 percent%만 symlink으로 사용."""
    if percent >= 100:
        return data_path

    subset_dir = os.path.join(
        os.path.dirname(data_path),
        f'_subset_{percent}pct_{label}'
    )

    for domain in ['trainA', 'trainB']:
        src = os.path.join(data_path, domain)
        dst = os.path.join(subset_dir, domain)

        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)

        if not os.path.exists(src):
            print(f"WARNING: {src} 없음, skip")
            continue

        files = sorted(os.listdir(src))
        n = max(1, int(len(files) * percent / 100))
        random.seed(42)
        selected = random.sample(files, n)

        for f in selected:
            sf = os.path.join(src, f)
            df = os.path.join(dst, f)
            if not os.path.exists(df):
                os.symlink(os.path.abspath(sf), df)

        print(f"  [{domain}] {n}/{len(files)} files ({percent}%)")

    for domain in ['testA', 'testB', 'valA', 'valB']:
        src = os.path.join(data_path, domain)
        dst = os.path.join(subset_dir, domain)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    print(f"  Subset dir: {subset_dir}")
    return subset_dir


# ============================================================
# Args
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='DeepThaw v2 Training')

    add_preset_name_parser(p, 'gen', GEN_PRESETS, 'uvcgan2')
    add_preset_name_parser(p, 'head', BH_PRESETS, 'bn', 'batch head')

    p.add_argument('--preset', type=str, required=True,
                   choices=list(PRESETS.keys()))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--data-path', type=str, default=None)
    p.add_argument('--label', type=str, default=None)
    p.add_argument('--run-tag', type=str, default=None,
                   help='Experiment tag (appended to label, used in output folder)')
    p.add_argument('--out-root', type=str, default=None,
                   help='Custom output root directory (e.g. ./outputs/kirc_gt)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--steps-per-epoch', type=int, default=1000)
    p.add_argument('--checkpoint-every', type=int, default=10)

    # UVCGAN2 base
    p.add_argument('--lambda-a', type=float, default=5.0)
    p.add_argument('--lambda-b', type=float, default=5.0)
    p.add_argument('--lambda-idt', type=float, default=0.5)

    # UNI
    p.add_argument('--lambda-uni-content', type=float, default=1.0)
    p.add_argument('--lambda-uni-distrib', type=float, default=1.0)

    # Self-Challenging
    p.add_argument('--challenge-weight', type=float, default=2.0)
    p.add_argument('--max-weight-ratio', type=float, default=5.0)

    # RAG
    p.add_argument('--rag-cache-dir', type=str, default=None)
    p.add_argument('--rag-k', type=int, default=5)
    p.add_argument('--lambda-rag', type=float, default=1.0)
    p.add_argument('--rag-random', action='store_true',
                   help='Random exemplars instead of nearest neighbors')
    p.add_argument('--rag-wrong', action='store_true',
                   help='Farthest (wrong) exemplars for stress test')
    p.add_argument('--rag-mode', type=str, default='feature',
                   choices=['feature', 'pixel', 'hybrid', 'contrastive', 'input'],
                   help='RAG mode: feature/pixel/hybrid/contrastive/input')
    p.add_argument('--ffpe-image-dir', type=str, default=None,
                   help='FFPE image dir (pixel/hybrid RAG에 필요)')
    p.add_argument('--rag-pixel-weight', type=float, default=1.0,
                   help='Pixel loss weight in hybrid mode')
    p.add_argument('--rag-feature-weight', type=float, default=1.0,
                   help='Feature loss weight in hybrid mode')
    p.add_argument('--stain-push-weight', type=float, default=0.5,
                   help='Stain push weight in contrastive (C-RAG) mode')

    # Data scaling
    p.add_argument('--data-percent', type=int, default=100,
                   help='Percentage of training data to use (10, 30, 50, 100)')

    # Pretrain
    p.add_argument('--no-pretrain', action='store_true')
    p.add_argument('--pretrain-path', type=str, default=None)

    return p.parse_args()


# ============================================================
# model_args
# ============================================================
def build_base_model_args(args):
    return {
        'lambda_a': args.lambda_a,
        'lambda_b': args.lambda_b,
        'lambda_idt': args.lambda_idt,
        'avg_momentum': 0.9999,
        'head_queue_size': 3,
        'head_config': {
            'name': 'batch-norm-2d',
            'input_features': 512,
            'output_features': 1,
            'activ': 'leakyrelu',
        },
    }


def build_deepthaw_model_args(args, preset_flags, rag_cache):
    ma = build_base_model_args(args)

    ma['use_uni_loss'] = preset_flags.get('use_uni_loss', False)
    ma['lambda_uni_content'] = args.lambda_uni_content
    ma['lambda_uni_distrib'] = args.lambda_uni_distrib

    ma['use_self_challenging'] = preset_flags.get('use_self_challenging', False)
    ma['challenge_weight'] = args.challenge_weight
    ma['max_weight_ratio'] = args.max_weight_ratio

    ma['use_rag'] = preset_flags.get('use_rag', False)
    ma['rag_cache_dir'] = rag_cache if ma['use_rag'] else None
    ma['rag_k_neighbors'] = args.rag_k
    ma['lambda_rag'] = args.lambda_rag
    ma['rag_mode'] = preset_flags.get('rag_mode', args.rag_mode)
    ma['ffpe_image_dir'] = args.ffpe_image_dir
    ma['rag_pixel_weight'] = args.rag_pixel_weight
    ma['rag_feature_weight'] = args.rag_feature_weight
    ma['stain_push_weight'] = args.stain_push_weight

    ma['xai_log_every'] = 500

    return ma


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    env = detect_env()
    preset = PRESETS[args.preset]
    model_name = preset['model']

    # Data path
    data_path = args.data_path or env['data']
    if data_path is None:
        print("ERROR: --data-path 필요")
        print("  예: --data-path /path/to/FS2FFPE")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} 없음")
        sys.exit(1)

    # Data subset
    if args.data_percent < 100:
        label_tag = args.label or f'v2-{args.preset}'
        print(f">>> Data subset: {args.data_percent}%")
        data_path = create_data_subset(data_path, args.data_percent, label_tag)

    # trainA/trainB symlink
    trainA = os.path.join(data_path, 'trainA')
    trainB = os.path.join(data_path, 'trainB')
    if not os.path.exists(trainA) or not os.path.exists(trainB):
        alt_a = os.path.join(data_path, 'train', 'A')
        alt_b = os.path.join(data_path, 'train', 'B')
        if os.path.exists(alt_a) and os.path.exists(alt_b):
            if not os.path.exists(trainA):
                os.symlink(os.path.abspath(alt_a), trainA)
            if not os.path.exists(trainB):
                os.symlink(os.path.abspath(alt_b), trainB)
        else:
            print(f"ERROR: trainA/trainB 없음. Contents: {os.listdir(data_path)}")
            sys.exit(1)

    # RAG cache
    rag_cache = args.rag_cache_dir or env.get('rag')
    if preset.get('use_rag', False) and rag_cache is None:
        print("WARNING: RAG preset인데 rag cache 없음 → RAG disabled")
        preset['use_rag'] = False

    # Label
    label = args.label or ('v2-' + args.preset)
    if args.run_tag:
        label = label + '-' + args.run_tag
    if args.data_percent < 100 and args.label is None:
        label += f'-{args.data_percent}pct'

    # model_args
    if model_name == 'uvcgan2':
        model_args = build_base_model_args(args)
    else:
        model_args = build_deepthaw_model_args(args, preset, rag_cache)

    # Print
    outdir = args.out_root if args.out_root else os.path.join(ROOT_OUTDIR, 'deepthaw')

    print(f"{'=' * 60}")
    print(f"DeepThaw v2 Training")
    print(f"{'=' * 60}")
    print(f"  Env:      {env['name']}")
    print(f"  Data:     {data_path}")
    print(f"  Model:    {model_name}")
    print(f"  Preset:   {args.preset}")
    print(f"  Label:    {label}")
    print(f"  Outdir:   {outdir}/{label}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Data%:    {args.data_percent}%")
    if model_name == 'uvcgan2-deepthaw':
        print(f"  UNI:      {preset.get('use_uni_loss', False)}")
        print(f"  SC:       {preset.get('use_self_challenging', False)}")
        print(f"  RAG:      {preset.get('use_rag', False)}")
        if preset.get('use_rag', False):
            rag_m = preset.get('rag_mode', args.rag_mode)
            print(f"  RAG mode: {rag_m}")
        if args.rag_random: print(f"  ⚠️ RAG RANDOM mode")
        if args.rag_wrong:  print(f"  ⚠️ RAG WRONG mode")
    print(f"{'=' * 60}")

    # Transfer
    transfer = None
    if not args.no_pretrain:
        base = args.pretrain_path or \
            'deepthaw/model_m(simple-autoencoder)_d(None)_g(vit-modnet)_deepthaw-pretrain-uvcgan2'
        transfer = {
            'base_model': base,
            'transfer_map': {'gen_ab': 'encoder', 'gen_ba': 'encoder'},
            'strict': True,
            'allow_partial': False,
            'fuzzy': None,
        }

    config = {
        'batch_size': args.batch_size,
        'data': {
            'datasets': [
                {
                    'dataset': {
                        'name': 'image-domain-hierarchy',
                        'path': data_path,
                        'domain': 'trainA',
                    },
                    'shape': [3, 256, 256],
                    'transform_train': [{'name': 'resize', 'size': 256}, 'random-flip-horizontal'],
                    'transform_test': None,
                },
                {
                    'dataset': {
                        'name': 'image-domain-hierarchy',
                        'path': data_path,
                        'domain': 'trainB',
                    },
                    'shape': [3, 256, 256],
                    'transform_train': [{'name': 'resize', 'size': 256}, 'random-flip-horizontal'],
                    'transform_test': None,
                },
            ],
            'workers': 4,
        },
        'epochs': args.epochs,
        'generator': {
            **GEN_PRESETS[args.gen],
            'optimizer': { 'name': 'Adam', 'lr': args.lr, 'betas': [0.5, 0.99] },
            'weight_init': { 'name': 'normal', 'init_gain': 0.02 },
        },
        'discriminator': {
            'model': 'basic',
            'model_args': { 'shrink_output': False },
            'optimizer': { 'name': 'Adam', 'lr': args.lr, 'betas': [0.5, 0.99] },
            'weight_init': { 'name': 'normal', 'init_gain': 0.02 },
            'spectr_norm': True,
        },
        'model': model_name,
        'model_args': model_args,
        'gradient_penalty': {
            'center': 0, 'lambda_gp': 1.0,
            'mix_type': 'real-fake', 'reduction': 'mean',
        },
        'scheduler': None,
        'loss': 'lsgan',
        'steps_per_epoch': args.steps_per_epoch,
        'transfer': transfer,
        'label': label,
        'outdir': outdir,
        'log_level': 'DEBUG',
        'checkpoint': args.checkpoint_every,
        'seed': 0,
    }

    print("\nStarting training...")
    train(config)


if __name__ == '__main__':
    main()