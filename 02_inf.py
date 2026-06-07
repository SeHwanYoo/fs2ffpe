#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepThaw Inference — FS/TS/BS-like -> FFPE-like generation
==========================================================

Supports both:

1) Single-folder inference:
   python 02_inf.py \
     --checkpoint /path/to/model_best.pt \
     --input-dir /path/to/testA \
     --output-dir /path/to/generated_testB \
     --direction ab

2) Multi-domain inference from one UVCGAN2-style data directory:
   python 02_inf.py \
     --checkpoint /path/to/model_best.pt \
     --data-dir /path/to/patches_linked_gen_bs \
     --domains trainA testA \
     --output-dir /path/to/generated_ffpe \
     --direction ab

Multi-domain output folder naming:
   --output-domain-style translated  (default): trainA -> trainB, testA -> testB
   --output-domain-style same:                   trainA -> trainA, testA -> testA
   --output-domain-style split:                  trainA -> train,  testA -> test

RAG cache handling:
  If --rag-cache-dir has train/rag_lookup.pt and test/rag_lookup.pt,
  trainA automatically uses rag-cache-dir/train and testA uses rag-cache-dir/test.
  Otherwise it falls back to --rag-cache-dir directly.
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(description='DeepThaw Inference with optional train/test multi-domain support')
    p.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path (.pt)')
    p.add_argument('--config-dir', type=str, default=None,
                   help='Config directory. Auto-detected from checkpoint path if omitted.')

    # Backward-compatible single-folder mode.
    p.add_argument('--input-dir', type=str, default=None,
                   help='Single input image directory. Use this OR --data-dir.')

    # Multi-domain mode.
    p.add_argument('--data-dir', type=str, default=None,
                   help='UVCGAN2-style data root containing trainA/testA/etc. Use this OR --input-dir.')
    p.add_argument('--domains', nargs='+', default=['testA'],
                   help='Domains to run when --data-dir is used, e.g. trainA testA')
    p.add_argument('--output-domain-style', type=str, default='translated',
                   choices=['translated', 'same', 'split'],
                   help='translated: trainA->trainB for ab. same: trainA->trainA. split: trainA->train.')

    p.add_argument('--output-dir', type=str, required=True, help='Output root directory')
    p.add_argument('--direction', type=str, default='ab', choices=['ab', 'ba'],
                   help='Translation direction: ab=A→B, ba=B→A')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--save-format', type=str, default='png', choices=['png', 'jpg', 'tiff'])

    # RAG
    p.add_argument('--rag-cache-dir', type=str, default=None,
                   help='RAG cache root. Can be flat or split cache containing train/test subdirs.')
    p.add_argument('--ffpe-image-dir', type=str, default=None,
                   help='FFPE image dir for old pixel/hybrid RAG format')

    # GPU
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--fp16', action='store_true', help='Use CUDA autocast mixed precision')
    return p.parse_args()


def get_image_files(img_dir):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return []
    return [f for f in sorted(img_dir.rglob('*')) if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]


def load_model(checkpoint_path, config_dir=None, device='cuda'):
    checkpoint_path = Path(checkpoint_path)

    if config_dir is None:
        config_dir = checkpoint_path.parent.parent
    config_dir = Path(config_dir)

    config_path = config_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')

    with open(config_path) as f:
        config = json.load(f)

    print(f"  Model:  {config.get('model', 'unknown')}")
    print(f"  Config: {config_path}")

    sys.path.insert(0, str(Path(__file__).parent))
    from uvcgan2.cgan import construct_model

    model = construct_model(
        model_name=config['model'],
        model_args=config.get('model_args', {}),
        generator_config=config.get('generator', {}),
        discriminator_config=config.get('discriminator', {}),
        batch_size=1,
        data_config=config.get('data', {}),
        gradient_penalty=config.get('gradient_penalty'),
        is_train=False,
        device=device,
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    print(f'  Loaded checkpoint: {checkpoint_path}')
    return model, config


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, files, base_dir, image_size=256):
        self.files = files
        self.base_dir = Path(base_dir)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        rel = fpath.relative_to(self.base_dir)
        name = str(rel.with_suffix(''))
        try:
            img = Image.open(fpath).convert('RGB')
            tensor = self.transform(img)
            return tensor, name, True
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size), name, False


def tensor_to_pil(tensor):
    img = (tensor.clamp(-1, 1) + 1) / 2
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def split_name_from_domain(domain: str) -> Optional[str]:
    d = domain.lower()
    if d.startswith('train'):
        return 'train'
    if d.startswith('test'):
        return 'test'
    if d.startswith('val'):
        return 'val'
    return None


def translated_domain_name(domain: str, direction: str, output_domain_style: str) -> str:
    if output_domain_style == 'same':
        return domain
    if output_domain_style == 'split':
        split = split_name_from_domain(domain)
        return split if split is not None else domain

    # translated
    if direction == 'ab':
        return domain[:-1] + 'B' if domain.endswith('A') else domain + '_toB'
    return domain[:-1] + 'A' if domain.endswith('B') else domain + '_toA'


def resolve_rag_cache_for_domain(rag_cache_root: Optional[str], domain: Optional[str]) -> Optional[str]:
    if rag_cache_root is None:
        return None
    root = Path(rag_cache_root)
    if domain is None:
        return str(root)

    split = split_name_from_domain(domain)
    if split is not None:
        split_dir = root / split
        if (split_dir / 'rag_lookup.pt').exists():
            return str(split_dir)
    return str(root)


def load_rag_conditioner(model, rag_cache_dir: Optional[str], ffpe_image_dir: Optional[str], device):
    if rag_cache_dir is None:
        return None
    try:
        from uvcgan2.modules.rag_conditioner import RAGConditioner

        shared_uni = None
        if hasattr(model, 'uni_loss_fn') and model.uni_loss_fn is not None:
            shared_uni = model.uni_loss_fn.uni
        elif hasattr(model, 'rag_conditioner') and model.rag_conditioner is not None:
            shared_uni = model.rag_conditioner.uni_model

        if shared_uni is None:
            print('  WARNING: No shared UNI model found; RAGConditioner may load its own UNI if supported')

        rag_conditioner = RAGConditioner(
            cache_dir=rag_cache_dir,
            uni_model=shared_uni,
            ffpe_image_dir=ffpe_image_dir,
        ).to(device)
        rag_conditioner.eval()
        print(f'  RAG conditioner loaded from {rag_cache_dir}')
        return rag_conditioner
    except Exception as e:
        print(f'  WARNING: RAG conditioner failed: {e}')
        return None


@torch.no_grad()
def run_inference_one_dir(model, input_dir: str, output_dir: str, args, device, domain: Optional[str] = None):
    print('\n' + '-' * 60)
    print(f"Running inference: domain={domain or 'single'}")
    print(f'  Input : {input_dir}')
    print(f'  Output: {output_dir}')

    rag_cache_dir = resolve_rag_cache_for_domain(args.rag_cache_dir, domain)
    if args.rag_cache_dir is not None:
        print(f'  RAG   : {rag_cache_dir}')

    rag_conditioner = load_rag_conditioner(
        model=model,
        rag_cache_dir=rag_cache_dir,
        ffpe_image_dir=args.ffpe_image_dir,
        device=device,
    )

    files = get_image_files(input_dir)
    print(f'  Found {len(files)} images')
    if not files:
        print('  WARNING: no images found; skipping')
        return 0

    dataset = InferenceDataset(files, input_dir, args.image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = model.models.gen_ab if args.direction == 'ab' else model.models.gen_ba
    gen.eval()

    total = 0
    t0 = time.time()
    autocast_ctx = torch.amp.autocast('cuda') if args.fp16 and device.type == 'cuda' else nullcontext()

    for imgs, names, valid_flags in tqdm(loader, desc=f"Generating {domain or ''}".strip(), ascii=True):
        valid_idx = [i for i, v in enumerate(valid_flags) if bool(v)]
        if not valid_idx:
            continue

        batch = imgs[valid_idx].to(device, non_blocking=True)
        with autocast_ctx:
            if rag_conditioner is not None:
                fused, _ = rag_conditioner(batch)
                output = gen(fused)
            else:
                output = gen(batch)

        for i, vi in enumerate(valid_idx):
            name = names[vi]
            out_path = output_dir / f'{name}.{args.save_format}'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img = tensor_to_pil(output[i])
            pil_img.save(out_path)
            total += 1

    elapsed = time.time() - t0
    print(f'  Done: {total} images in {elapsed:.1f}s ({total / max(elapsed, 1e-6):.1f} images/s)')
    return total


def build_jobs(args) -> Tuple[Tuple[str, str, Optional[str]], ...]:
    if args.input_dir is not None and args.data_dir is not None:
        raise ValueError('Use either --input-dir or --data-dir, not both.')
    if args.input_dir is None and args.data_dir is None:
        raise ValueError('One of --input-dir or --data-dir is required.')

    if args.input_dir is not None:
        return ((args.input_dir, args.output_dir, None),)

    jobs = []
    data_dir = Path(args.data_dir)
    for domain in args.domains:
        in_dir = data_dir / domain
        out_domain = translated_domain_name(domain, args.direction, args.output_domain_style)
        out_dir = Path(args.output_dir) / out_domain
        jobs.append((str(in_dir), str(out_dir), domain))
    return tuple(jobs)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('DeepThaw Inference')
    print('=' * 60)
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Direction:  {args.direction}')
    print(f'  Device:     {device}')
    if args.input_dir:
        print(f'  Single input: {args.input_dir}')
    if args.data_dir:
        print(f'  Data dir:   {args.data_dir}')
        print(f'  Domains:    {args.domains}')
        print(f'  Out style:  {args.output_domain_style}')
    print(f'  Output:     {args.output_dir}')

    print('\nLoading model...')
    model, _config = load_model(args.checkpoint, args.config_dir, device)

    jobs = build_jobs(args)
    total_all = 0
    for input_dir, output_dir, domain in jobs:
        total_all += run_inference_one_dir(model, input_dir, output_dir, args, device, domain)

    print('\n' + '=' * 60)
    print(f'Done. Generated total {total_all} images')
    print(f'Output root: {args.output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
