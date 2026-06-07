"""
rename_samples.py — UVCGAN2 생성 결과 파일명 복원
===================================================

UVCGAN2 기본 test가 생성하는 파일:
  sample_000000_real_a.png
  sample_000000_fake_b.png
  sample_000000_reco_a.png
  sample_000001_real_a.png
  ...

이 스크립트가 하는 것:
  1. data config에서 원본 파일명 순서를 읽음
  2. sample_XXXXXX → 원본 파일명으로 매핑
  3. fake_b만 추출 + 이름 복원 + 정리된 폴더에 저장

사용법:
  # 기본 (fake_b만 추출, 원본 이름으로)
  python rename_samples.py \
    --samples-dir outdir/deepthaw/model_.../samples/epoch_200 \
    --data-dir /path/to/FS2FFPE \
    --domain trainA \
    --output-dir /path/to/renamed_FFPE

  # testA 도메인
  python rename_samples.py \
    --samples-dir outdir/deepthaw/model_.../samples/epoch_200 \
    --data-dir /path/to/FS2FFPE \
    --domain testA \
    --output-dir /path/to/renamed_FFPE_test

  # 모든 종류 (real_a, fake_b, reco_a) 다 저장
  python rename_samples.py \
    --samples-dir /path/to/samples \
    --data-dir /path/to/data \
    --domain trainA \
    --output-dir /path/to/output \
    --keep-all
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

# UVCGAN2 sample filename pattern: sample_000042_fake_b.png
SAMPLE_RE = re.compile(r'sample_(\d+)_(real_a|fake_b|reco_a|real_b|fake_a|reco_b)\.(png|jpg|tiff)')


def parse_args():
    p = argparse.ArgumentParser(description='Rename UVCGAN2 sample outputs')
    p.add_argument('--samples-dir', type=str, required=True,
                   help='Directory containing sample_XXXXXX_*.png files')
    p.add_argument('--data-dir', type=str, required=True,
                   help='Original data directory (FS2FFPE root)')
    p.add_argument('--domain', type=str, default='trainA',
                   help='Domain folder name (trainA, testA, etc.)')
    p.add_argument('--output-dir', type=str, required=True,
                   help='Output directory for renamed files')
    p.add_argument('--type', type=str, default='fake_b',
                   choices=['fake_b', 'real_a', 'reco_a', 'fake_a', 'real_b', 'reco_b'],
                   help='Which sample type to extract (default: fake_b)')
    p.add_argument('--keep-all', action='store_true',
                   help='Keep all types (fake_b, real_a, reco_a) in separate subfolders')
    p.add_argument('--ext', type=str, default='png',
                   help='Output file extension')
    p.add_argument('--dry-run', action='store_true',
                   help='Print mapping without copying')
    return p.parse_args()


def get_sorted_files(data_dir, domain):
    """원본 데이터의 파일명 순서 (sorted, dataset과 동일 순서)."""
    domain_dir = Path(data_dir) / domain
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

    files = []
    for f in sorted(domain_dir.rglob('*')):
        if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS:
            rel = f.relative_to(domain_dir)
            files.append(rel)

    print(f"  Original files ({domain}): {len(files)}")
    return files


def parse_sample_files(samples_dir):
    """sample_XXXXXX_type.ext 파일들 파싱."""
    samples_dir = Path(samples_dir)
    parsed = {}  # {(index, type): filepath}

    for f in sorted(samples_dir.iterdir()):
        if not f.is_file():
            continue
        m = SAMPLE_RE.match(f.name)
        if m:
            idx = int(m.group(1))
            stype = m.group(2)
            parsed[(idx, stype)] = f

    types_found = set(t for _, t in parsed.keys())
    indices = set(i for i, _ in parsed.keys())
    print(f"  Sample files: {len(parsed)} ({len(indices)} indices, types: {sorted(types_found)})")
    return parsed


def main():
    args = parse_args()

    print("=" * 60)
    print("UVCGAN2 Sample Renamer")
    print("=" * 60)
    print(f"  Samples: {args.samples_dir}")
    print(f"  Data:    {args.data_dir}")
    print(f"  Domain:  {args.domain}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Type:    {'all' if args.keep_all else args.type}")

    # 1. Get original filenames (sorted = dataset order)
    original_files = get_sorted_files(args.data_dir, args.domain)

    # 2. Parse sample files
    sample_files = parse_sample_files(args.samples_dir)

    # 3. Determine which types to process
    if args.keep_all:
        types_to_process = sorted(set(t for _, t in sample_files.keys()))
    else:
        types_to_process = [args.type]

    # 4. Rename and copy
    output_dir = Path(args.output_dir)
    total_copied = 0
    total_missing = 0

    for stype in types_to_process:
        if args.keep_all:
            type_dir = output_dir / stype
        else:
            type_dir = output_dir
        type_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Processing: {stype}")

        for idx in range(len(original_files)):
            key = (idx, stype)
            if key not in sample_files:
                total_missing += 1
                continue

            src = sample_files[key]
            original_name = original_files[idx]
            dst = type_dir / original_name.with_suffix(f'.{args.ext}')
            dst.parent.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                print(f"    {src.name} → {dst.relative_to(output_dir)}")
            else:
                shutil.copy2(src, dst)

            total_copied += 1

    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Done!")
    print(f"  Copied: {total_copied}")
    if total_missing:
        print(f"  Missing: {total_missing} (samples not found for some indices)")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
