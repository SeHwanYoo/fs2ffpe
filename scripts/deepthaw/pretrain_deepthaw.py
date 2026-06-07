"""
DeepThaw Step 1: BERT Pretraining
=================================

위치: scripts/deepthaw/pretrain_deepthaw.py

사용법:
    python scripts/deepthaw/pretrain_deepthaw.py --batch-size 8

이게 끝나면 Step 2 (train_fs2ffpe.py) 실행
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser

import argparse

def parse_cmdargs():
    parser = argparse.ArgumentParser(description='DeepThaw BERT Pretraining')
    add_preset_name_parser(parser)
    add_batch_size_parser(parser, default=8)
    return parser.parse_args()

cmdargs = parse_cmdargs()

# ============================================================
# 데이터 경로 설정 (형 데이터에 맞게 수정)
# ============================================================
DATA_PATH = 'fs2ffpe'  # ${UVCGAN2_DATA}/fs2ffpe 에 데이터 있어야 함

ROOT_OUTDIR = './saved_results'

# 데이터 폴더 구조:
# fs2ffpe/
#   train/
#     FS/      ← Frozen Section 이미지들
#     FFPE/    ← FFPE 이미지들
#   val/
#     FS/
#     FFPE/

args_dict = {
    'batch_size': cmdargs.batch_size,
    
    # ============================================================
    # Data Configuration
    # ============================================================
    'data': {
        'datasets': {
            'train': {
                'dataset': {
                    'name': 'unpaired-image-domain-hierarchy',
                    'path': DATA_PATH,
                },
            },
            'val': {
                'dataset': {
                    'name': 'unpaired-image-domain-hierarchy',
                    'path': DATA_PATH,
                },
            },
        },
        'transform_train': [
            {'name': 'resize', 'size': (286, 286)},
            {'name': 'random-crop', 'size': (256, 256)},
            {'name': 'random-flip-horizontal'},
            {'name': 'to-tensor'},
            {'name': 'normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        ],
        'transform_val': [
            {'name': 'resize', 'size': (256, 256)},
            {'name': 'to-tensor'},
            {'name': 'normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        ],
    },
    
    # ============================================================
    # Model Configuration (BERT-style Autoencoder)
    # ============================================================
    'model': {
        'name': 'autoencoder',
        'model': {
            'name': 'vit-modnet-autoencoder',
            'input_shape': (3, 256, 256),
            'features': 384,
            'n_heads': 6,
            'n_blocks': 12,
            'ffn_features': 1536,
            'embed_features': 384,
            'activ': 'gelu',
            'norm': 'layer',
            'modnet_features_list': [64, 128, 256, 512],
            'modnet_activ': 'leaky-relu',
            'modnet_norm': None,
            'modnet_downsample': 'conv',
            'modnet_upsample': 'convtranspose',
            'modnet_rezero': True,
            'vit_positional_encoding': 'sinusoidal',
            'vit_rezero': True,
        },
        # Inpainting task: 40% random masking
        'masking': {
            'name': 'image-patch-random',
            'patch_size': (16, 16),
            'fraction': 0.4,
        },
    },
    
    # ============================================================
    # Training Configuration
    # ============================================================
    'scheduler': {
        'name': 'linear',
        'epochs': 500,
        'warmup_epochs': 5,
        'min_lr': 1e-6,
    },
    'optimizer': {
        'name': 'adamw',
        'lr': 1e-4,
        'weight_decay': 0.05,
        'betas': (0.9, 0.95),
    },
    
    'epochs': 500,
    'checkpoint': 50,
    
    # ============================================================
    # Output
    # ============================================================
    'label': 'deepthaw-bert-pretrain',
    'outdir': os.path.join(ROOT_OUTDIR, 'deepthaw'),
}

train(args_dict)
