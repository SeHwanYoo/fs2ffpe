"""
DeepThaw Step 2: FS→FFPE Translation Training
==============================================

위치: scripts/deepthaw/train_fs2ffpe.py

사용법:
    python scripts/deepthaw/train_fs2ffpe.py --batch-size 1

⚠️ 먼저 pretrain_deepthaw.py 실행해서 BERT pretrain 완료해야 함!
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser

import argparse

def parse_cmdargs():
    parser = argparse.ArgumentParser(description='DeepThaw FS2FFPE Training')
    add_preset_name_parser(parser)
    add_batch_size_parser(parser, default=1)
    return parser.parse_args()

cmdargs = parse_cmdargs()

# ============================================================
# 🔥 BERT Pretrain 경로 설정 (pretrain 끝나면 여기 수정!)
# ============================================================
# pretrain_deepthaw.py 실행 후 생성된 폴더명을 여기에 붙여넣기
# 예: 'deepthaw/model_d(autoencoder)_m(vit-modnet-autoencoder)_...'

PRETRAINED_PATH = 'deepthaw/deepthaw-bert-pretrain'  # ← 실제 경로로 수정!

# ============================================================
# 데이터 경로
# ============================================================
DATA_PATH = 'fs2ffpe'

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
                'domain_a': 'FS',
                'domain_b': 'FFPE',
            },
            'val': {
                'dataset': {
                    'name': 'unpaired-image-domain-hierarchy',
                    'path': DATA_PATH,
                },
                'domain_a': 'FS',
                'domain_b': 'FFPE',
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
    # Model Configuration (CycleGAN with ViT-ModNet)
    # ============================================================
    'model': {
        'name': 'cyclegan',
        
        # Generator: UVCGAN2의 ViT-ModNet
        'generator': {
            'name': 'vit-modnet',
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
        
        # Discriminator: StyleGAN2 style
        'discriminator': {
            'name': 'stylegan2',
            'features': 64,
            'max_features': 512,
        },
        
        # Loss weights
        'lambda_a': 10.0,      # Cycle loss A
        'lambda_b': 10.0,      # Cycle loss B
        'lambda_idt': 0.5,     # Identity loss
        
        # GAN loss type
        'gan_loss': 'lsgan',
    },
    
    # ============================================================
    # Transfer Learning (Pretrained BERT weights)
    # ============================================================
    'transfer': {
        'base_model': PRETRAINED_PATH,
        'transfer_map': {
            'gen_ab': 'encoder',
            'gen_ba': 'encoder',
        },
        'strict': False,
    },
    
    # ============================================================
    # Training Configuration
    # ============================================================
    'scheduler': {
        'name': 'linear',
        'epochs': 200,
        'warmup_epochs': 0,
        'min_lr': 1e-6,
    },
    'optimizer': {
        'name': 'adam',
        'lr': 1e-4,
        'betas': (0.5, 0.999),
    },
    
    'epochs': 200,
    'checkpoint': 10,
    
    # ============================================================
    # Output
    # ============================================================
    'label': 'deepthaw-fs2ffpe',
    'outdir': os.path.join(ROOT_OUTDIR, 'deepthaw'),
}

train(args_dict)
