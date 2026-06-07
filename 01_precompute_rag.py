"""
precompute_rag_matches_uni_split.py
===================================

FS/TS/BS/Generated-BS -> DX(FFPE) UNI retrieval cache with patient-level,
label-balanced train/test split.

Supported input_dir structures:
  1) Slide-folder layout, as in TCGA patch exports:
     input_dir/TCGA-..-TS1/**/<patch images>
     input_dir/TCGA-..-BS1/**/<patch images>
     input_dir/TCGA-..-DX1/**/<patch images>

  2) Source-folder layout:
     input_dir/TS/**/<patch images>
     input_dir/BS/**/<patch images>
     input_dir/DX/**/<patch images>

If --gen-dir is NOT given:
  query images = input_dir/TS + input_dir/BS

If --gen-dir is given:
  query images = input_dir/TS + TS-like images under gen_dir
  i.e. real BS in input_dir/BS is not used, and non-TS-like images under gen_dir
  are skipped. Generated TS-like images are semantically treated as BS-like and
  saved in lookup keys with prefix BS/ by default so downstream code that expects
  TS/... and BS/... keys can still work.

DX database:
  All folders/files classified as DX under input_dir. There is no --ffpe-dir argument anymore.

Outputs:
  output_dir/
    split.json
    train_patient_ids.txt
    test_patient_ids.txt
    train/
      rag_lookup.pt
      ffpe_features.npy
      ffpe_filenames.npy
      meta.json
    test/
      rag_lookup.pt
      meta.json

Default leakage-safe test behaviour:
  train query -> train DX database
  test query  -> train DX database

Use --test-db-scope same_split if you intentionally want:
  test query -> test DX database

Example, first run with real TS+BS:
  python precompute_rag_matches_uni_split.py \
    --input-dir /projects_vol/gp_wilsongoh/sehwan001/lung/patches \
    --label-file /projects_vol/gp_wilsongoh/sehwan001/lung/gdc_manifest_luad_kras_balanced.tsv \
    --output-dir /projects_vol/gp_wilsongoh/sehwan001/lung/rag_cache_uni_real_bs \
    --test-ratio 0.2 \
    --seed 42 \
    --k 5 \
    --batch-size 128

Example, reuse the same split with generated BS:
  python precompute_rag_matches_uni_split.py \
    --input-dir /projects_vol/gp_wilsongoh/sehwan001/lung/patches \
    --gen-dir /projects_vol/gp_wilsongoh/sehwan001/lung/Gen_bs \
    --split-json /projects_vol/gp_wilsongoh/sehwan001/lung/rag_cache_uni_real_bs/split.json \
    --output-dir /projects_vol/gp_wilsongoh/sehwan001/lung/rag_cache_uni_gen_bs \
    --k 5 \
    --batch-size 128
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception:  # allows --dry-run even before timm is installed/activated
    timm = None
    resolve_data_config = None
    create_transform = None

try:
    from huggingface_hub import login
except Exception:  # huggingface_hub may be unavailable in some envs until installed
    login = None


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
TCGA_CASE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
SLIDE_TYPE_RE = re.compile(r"(?:^|[-_/\\])(TS|BS|DX)\d*(?=$|[-_.\\/])", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    path: str
    name: str          # relative key saved in rag_lookup.pt, without extension
    case_id: str
    source: str        # TS, BS, GEN_BS, DX


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute UNI RAG matches with label-balanced patient train/test split"
    )

    # New data interface
    p.add_argument("--input-dir", type=str, required=True,
                   help="Root directory containing TCGA slide folders such as *-TS1/*-BS1/*-DX1, or TS/BS/DX subfolders")
    p.add_argument("--gen-dir", type=str, default=None,
                   help="Optional generated BS directory. If set, query images are input_dir/TS + gen_dir, not input_dir/BS")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output root directory")

    # Split interface
    p.add_argument("--label-file", type=str, default=None,
                   help="TSV/CSV containing patient_id and label columns. Required when --split-json is not provided")
    p.add_argument("--patient-col", type=str, default="patient_id",
                   help="Patient ID column in label file")
    p.add_argument("--label-col", type=str, default="label",
                   help="Label column in label file")
    p.add_argument("--split-json", type=str, default=None,
                   help="Existing split JSON. If provided, train/test patient IDs are reused exactly")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="Patient-level test ratio used only when creating a new split")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used only when creating a new split")
    p.add_argument("--test-db-scope", type=str, default="train", choices=["train", "same_split"],
                   help="For test queries, use train DX DB by default to avoid leakage, or same_split to use test DX DB")
    p.add_argument("--gen-key-prefix", type=str, default="BS",
                   help="Lookup key prefix for --gen-dir images. Default BS keeps compatibility with real BS keys")
    p.add_argument("--only-split", type=str, default="both", choices=["train", "test", "both"],
                   help="Which RAG lookup(s) to compute")
    p.add_argument("--dry-run", action="store_true",
                   help="Only scan records, create/reuse the patient split, and print counts. Do not load UNI or compute features")

    # Retrieval settings
    p.add_argument("--k", type=int, default=5,
                   help="Top-K nearest DX patches")
    p.add_argument("--batch-size", type=int, default=128,
                   help="UNI feature extraction batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--query-chunk-size", type=int, default=500,
                   help="Number of query features per similarity-search chunk")
    p.add_argument("--db-chunk-size", type=int, default=10000,
                   help="Number of DB features per similarity-search chunk")
    p.add_argument("--save-features", action="store_true",
                   help="Also save query feature arrays for debugging/reuse")
    p.add_argument("--reuse-ffpe-cache", action="store_true",
                   help="Reuse cached DX features when available")
    p.add_argument("--exclude-same-case", action="store_true",
                   help="Exclude same-case DX patches during top-k search")

    # UNI auth / model
    p.add_argument("--hf-token-env", type=str, default="HF_TOKEN",
                   help="Environment variable name for Hugging Face token. No token is hard-coded")
    p.add_argument("--uni-model", type=str, default="hf-hub:MahmoodLab/UNI",
                   help="timm model name for UNI")

    return p.parse_args()


def extract_case_id(text: str) -> str:
    """Extract TCGA patient/case ID from a path or file name."""
    text = str(text).replace("\\", "/")
    m = TCGA_CASE_RE.search(text)
    if m:
        return m.group(1).upper()

    # Fallback: use folder or filename stem. This keeps non-TCGA data usable.
    parts = [p for p in text.split("/") if p]
    for part in reversed(parts):
        stem = Path(part).stem
        if stem:
            return stem.split("_")[0]
    return Path(text).stem


def image_files_under(root: Path) -> List[Path]:
    if not root.exists():
        print(f"  ⚠️ Directory does not exist: {root}")
        return []
    return [f for f in sorted(root.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]


def infer_slide_type_from_relative(rel_path: Path) -> Optional[str]:
    """Infer TS / BS / DX from a relative path.

    Works for both:
      TS/<case>/<patch>.png
      TCGA-38-4628-11A-01-TS1/<patch>.png

    We inspect directory components first, then the filename stem as fallback.
    """
    parts = rel_path.parts

    # Prefer folder names; patch filenames can be arbitrary.
    for part in parts[:-1]:
        m = SLIDE_TYPE_RE.search(part)
        if m:
            return m.group(1).upper()

    # Fallback for flat files directly under input_dir.
    if parts:
        m = SLIDE_TYPE_RE.search(Path(parts[-1]).stem)
        if m:
            return m.group(1).upper()

    return None


def scan_input_records(input_dir: Path) -> Dict[str, List[ImageRecord]]:
    """Scan input_dir once and classify records into TS / BS / DX.

    The saved lookup name is input_dir-relative without extension.
    Examples:
      TCGA-38-4628-11A-01-TS1/patch_001
      TS/TCGA-38-4628/patch_001
    """
    records_by_source: Dict[str, List[ImageRecord]] = {"TS": [], "BS": [], "DX": []}
    unknown_count = 0

    files = image_files_under(input_dir)
    if not files:
        return records_by_source

    for f in files:
        try:
            rel = f.relative_to(input_dir)
        except ValueError:
            rel = Path(f.name)

        source = infer_slide_type_from_relative(rel)
        if source not in records_by_source:
            unknown_count += 1
            continue

        name = rel.with_suffix("").as_posix()
        case_id = extract_case_id(str(rel))
        records_by_source[source].append(
            ImageRecord(path=str(f), name=name, case_id=case_id, source=source)
        )

    for source in ["TS", "BS", "DX"]:
        recs = records_by_source[source]
        print(f"  {source:7s}: {len(recs):,} images from {len(set(r.case_id for r in recs)):,} patients/cases")
    if unknown_count:
        print(f"  ⚠️ Ignored {unknown_count:,} images because TS/BS/DX could not be inferred from path")

    return records_by_source


def collect_generated_records(gen_dir: Path, key_prefix: str = "BS") -> List[ImageRecord]:
    """Collect generated BS images from TS-like paths only.

    generating_bs.py preserves TS-like relative paths for BS-like images, e.g.:
      gen_dir/TCGA-..-TS1/patch.png

    Therefore, in --gen-dir mode, we ONLY accept images whose gen_dir-relative
    path is inferred as TS. Those files are semantically treated as GEN_BS and
    saved with a BS/ lookup-key prefix by default. Files inferred as BS, DX, or
    unknown are skipped to avoid accidentally mixing real BS or unrelated images.
    """
    records: List[ImageRecord] = []
    files = image_files_under(gen_dir)
    prefix = key_prefix.strip("/") if key_prefix is not None else ""
    skipped_non_ts = 0

    for f in files:
        try:
            rel_path = f.relative_to(gen_dir)
            rel_no_suffix = rel_path.with_suffix("").as_posix()
        except ValueError:
            rel_path = Path(f.name)
            rel_no_suffix = f.with_suffix("").name

        gen_slide_type = infer_slide_type_from_relative(rel_path)
        if gen_slide_type != "TS":
            skipped_non_ts += 1
            continue

        name = f"{prefix}/{rel_no_suffix}" if prefix else rel_no_suffix
        case_id = extract_case_id(rel_no_suffix)
        records.append(ImageRecord(path=str(f), name=name, case_id=case_id, source="GEN_BS"))

    print(f"  {'GEN_BS':7s}: {len(records):,} TS-like generated images from {len(set(r.case_id for r in records)):,} patients/cases")
    if skipped_non_ts:
        print(f"  ⚠️ Skipped {skipped_non_ts:,} gen-dir images because they were not TS-like paths")
    return records


def build_records(input_dir: Path, gen_dir: Optional[Path], gen_key_prefix: str = "BS") -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Return query records and DX records according to requested rules."""
    print("\n[0/5] Scanning image records...")

    input_records = scan_input_records(input_dir)
    ts_records = input_records["TS"]
    bs_records = input_records["BS"]
    dx_records = input_records["DX"]

    if gen_dir is None:
        query_records = ts_records + bs_records
        print("  Query mode: real TS + real BS")
    else:
        gen_records = collect_generated_records(gen_dir, key_prefix=gen_key_prefix)
        query_records = ts_records + gen_records
        print(f"  Query mode: real TS + generated BS. Real BS under input_dir is ignored. gen key prefix={gen_key_prefix!r}")

    return query_records, dx_records

def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(errors="ignore")[:4096]
    if "\t" in sample:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def load_patient_labels(label_file: str, patient_col: str, label_col: str) -> Dict[str, str]:
    """Load patient-level labels from a CSV/TSV manifest."""
    path = Path(label_file)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    delimiter = sniff_delimiter(path)
    patient_to_label: Dict[str, str] = {}

    with path.open("r", newline="", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in label file: {label_file}")

        # allow case-insensitive column matching
        col_map = {c.lower(): c for c in reader.fieldnames}
        pcol = col_map.get(patient_col.lower(), patient_col)
        lcol = col_map.get(label_col.lower(), label_col)

        if pcol not in reader.fieldnames or lcol not in reader.fieldnames:
            raise ValueError(
                f"Label file must contain columns '{patient_col}' and '{label_col}'. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            pid = str(row[pcol]).strip()
            label = str(row[lcol]).strip()
            if not pid or not label or pid.lower() == "nan" or label.lower() == "nan":
                continue
            pid = pid.upper() if pid.upper().startswith("TCGA-") else pid
            if pid in patient_to_label and patient_to_label[pid] != label:
                raise ValueError(
                    f"Conflicting labels for patient {pid}: "
                    f"{patient_to_label[pid]} vs {label}"
                )
            patient_to_label[pid] = label

    print(f"  Loaded labels for {len(patient_to_label):,} patients from {label_file}")
    print(f"  Label counts: {dict(Counter(patient_to_label.values()))}")
    return patient_to_label


def stratified_patient_split(
    patient_to_label: Dict[str, str],
    eligible_patients: Iterable[str],
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Patient-level stratified train/test split without sklearn dependency."""
    if not 0 < test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1")

    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = defaultdict(list)
    for pid in sorted(set(eligible_patients)):
        if pid in patient_to_label:
            by_label[patient_to_label[pid]].append(pid)

    train_ids: List[str] = []
    test_ids: List[str] = []

    print("\nCreating label-balanced patient split:")
    for label, ids in sorted(by_label.items(), key=lambda x: str(x[0])):
        ids = ids[:]
        rng.shuffle(ids)

        if len(ids) <= 1:
            n_test = 0
        else:
            n_test = int(round(len(ids) * test_ratio))
            n_test = max(1, n_test)
            n_test = min(n_test, len(ids) - 1)

        test_part = sorted(ids[:n_test])
        train_part = sorted(ids[n_test:])
        train_ids.extend(train_part)
        test_ids.extend(test_part)

        print(f"  label={label}: total={len(ids)}, train={len(train_part)}, test={len(test_part)}")

    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)
    if not train_ids or not test_ids:
        raise ValueError("Split failed: train or test patient list is empty")
    return train_ids, test_ids


def load_split_json(path: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    with open(path, "r") as f:
        data = json.load(f)

    if "train_patient_ids" in data and "test_patient_ids" in data:
        train_ids = data["train_patient_ids"]
        test_ids = data["test_patient_ids"]
    elif "train" in data and "test" in data:
        train_ids = data["train"]
        test_ids = data["test"]
    else:
        raise ValueError(
            "Split JSON must contain either train_patient_ids/test_patient_ids or train/test"
        )

    patient_labels = data.get("patient_labels", {})
    train_ids = [str(x).upper() if str(x).upper().startswith("TCGA-") else str(x) for x in train_ids]
    test_ids = [str(x).upper() if str(x).upper().startswith("TCGA-") else str(x) for x in test_ids]
    patient_labels = {
        (str(k).upper() if str(k).upper().startswith("TCGA-") else str(k)): str(v)
        for k, v in patient_labels.items()
    }
    return sorted(train_ids), sorted(test_ids), patient_labels


def records_by_patients(records: Sequence[ImageRecord], patients: Sequence[str]) -> List[ImageRecord]:
    patient_set = set(patients)
    return [r for r in records if r.case_id in patient_set]


def count_labels(patient_ids: Sequence[str], patient_to_label: Dict[str, str]) -> Dict[str, int]:
    return dict(Counter(patient_to_label.get(pid, "UNKNOWN") for pid in patient_ids))


def save_split_files(
    output_dir: Path,
    train_ids: List[str],
    test_ids: List[str],
    patient_to_label: Dict[str, str],
    args,
    eligible_patients: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Train/test patient overlap detected: {overlap[:10]}")

    split = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "source_label_file": args.label_file,
        "input_dir": str(args.input_dir),
        "gen_dir": str(args.gen_dir) if args.gen_dir else None,
        "train_patient_ids": train_ids,
        "test_patient_ids": test_ids,
        "patient_labels": {pid: patient_to_label.get(pid, "UNKNOWN") for pid in sorted(set(train_ids) | set(test_ids))},
        "label_counts": {
            "eligible": count_labels(eligible_patients, patient_to_label),
            "train": count_labels(train_ids, patient_to_label),
            "test": count_labels(test_ids, patient_to_label),
        },
    }

    with (output_dir / "split.json").open("w") as f:
        json.dump(split, f, indent=2)

    with (output_dir / "train_patient_ids.txt").open("w") as f:
        f.write("\n".join(train_ids) + "\n")

    with (output_dir / "test_patient_ids.txt").open("w") as f:
        f.write("\n".join(test_ids) + "\n")

    print(f"\nSaved split:")
    print(f"  {output_dir / 'split.json'}")
    print(f"  train labels: {split['label_counts']['train']}")
    print(f"  test labels:  {split['label_counts']['test']}")


# ================================================================
# UNI model and feature extraction
# ================================================================
def load_uni(device: torch.device, model_name: str, hf_token_env: str):
    print("🤖 Loading UNI Model...")

    token = os.environ.get(hf_token_env) or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token and login is not None:
        try:
            login(token=token)
            print(f"  Hugging Face token loaded from ${hf_token_env}")
        except Exception as e:
            print(f"  ⚠️ HF login failed, trying model load anyway: {e}")
    elif token and login is None:
        print("  ⚠️ huggingface_hub is not importable, trying model load without login")
    else:
        print("  ℹ️ No HF token found. Trying public/cache access")

    if timm is None or resolve_data_config is None or create_transform is None:
        print("\n❌ [ERROR] timm is not importable in this environment.")
        print("Activate/install the environment containing timm before computing UNI features.")
        sys.exit(1)

    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        model = model.to(device)
        model.eval()
        preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        return model, preprocess
    except Exception as e:
        print(f"\n❌ [ERROR] UNI model load failed: {e}")
        print("Check that timm, huggingface_hub, and HF_TOKEN are available in this environment.")
        sys.exit(1)


class ImageRecordDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec.path).convert("RGB")
            tensor = self.transform(img)
            return tensor, rec.name, rec.path, rec.case_id, rec.source, True
        except Exception:
            return torch.zeros(3, 224, 224), rec.name, rec.path, rec.case_id, rec.source, False


@torch.no_grad()
def extract_features_from_records(
    records: Sequence[ImageRecord],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
    desc: str = "Extracting UNI features",
):
    """
    ImageRecord list -> L2-normalized UNI features.

    Returns:
        names, features, paths, case_ids, sources
    """
    records = list(records)
    if not records:
        return [], None, [], [], []

    dataset = ImageRecordDataset(records, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_features: List[np.ndarray] = []
    all_names: List[str] = []
    all_paths: List[str] = []
    all_case_ids: List[str] = []
    all_sources: List[str] = []

    for imgs, names, paths, case_ids, sources, valid_flags in tqdm(loader, desc=desc, ascii=True):
        if isinstance(valid_flags, torch.Tensor):
            valid_flags = valid_flags.cpu().tolist()
        valid_idx = [i for i, v in enumerate(valid_flags) if bool(v)]
        if not valid_idx:
            continue

        batch = imgs[valid_idx].to(device, non_blocking=True)
        feats = model(batch)
        feats = F.normalize(feats, dim=1).detach().cpu().numpy()

        for out_i, src_i in enumerate(valid_idx):
            all_features.append(feats[out_i])
            all_names.append(names[src_i])
            all_paths.append(paths[src_i])
            all_case_ids.append(case_ids[src_i])
            all_sources.append(sources[src_i])

    if not all_features:
        return [], None, [], [], []

    features = np.stack(all_features, axis=0).astype(np.float32)
    print(f"  Extracted {features.shape[0]:,} valid images, dim={features.shape[1]}")
    return all_names, features, all_paths, all_case_ids, all_sources


# ================================================================
# Chunked Top-K Search
# ================================================================
def chunked_topk_search(
    query_matrix: np.ndarray,
    db_matrix: np.ndarray,
    k: int = 5,
    db_chunk_size: int = 10000,
    use_fp16: bool = True,
    query_case_ids: Optional[Sequence[str]] = None,
    db_case_ids: Optional[Sequence[str]] = None,
):
    """Query x DB cosine similarity -> top-k, chunked to avoid OOM."""
    if query_matrix is None or db_matrix is None:
        raise ValueError("query_matrix and db_matrix must not be None")

    Q = query_matrix.shape[0]
    N = db_matrix.shape[0]
    if Q == 0 or N == 0:
        raise ValueError(f"Empty query or DB features: Q={Q}, N={N}")

    actual_k = min(k, N)
    exclude_same_case = query_case_ids is not None and db_case_ids is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32

    all_scores = np.full((Q, actual_k), -np.inf, dtype=np.float32)
    all_indices = np.zeros((Q, actual_k), dtype=np.int64)

    q_tensor = torch.from_numpy(query_matrix).to(device=device, dtype=dtype)
    num_chunks = (N + db_chunk_size - 1) // db_chunk_size

    for ci in range(num_chunks):
        s = ci * db_chunk_size
        e = min(s + db_chunk_size, N)
        db_chunk = torch.from_numpy(db_matrix[s:e]).to(device=device, dtype=dtype)
        sim = torch.matmul(q_tensor, db_chunk.T)

        if exclude_same_case:
            # This is intentionally simple and robust. For very large chunks,
            # turn off --exclude-same-case or lower --query-chunk-size if slow.
            for q in range(Q):
                q_case = query_case_ids[q]
                for j in range(e - s):
                    if db_case_ids[s + j] == q_case:
                        sim[q, j] = -float("inf")

        ck = min(actual_k, sim.shape[1])
        chunk_scores, chunk_local = torch.topk(sim, k=ck, dim=1)
        chunk_scores = chunk_scores.float().cpu().numpy()
        chunk_global = chunk_local.cpu().numpy() + s

        for q in range(Q):
            combined_s = np.concatenate([all_scores[q], chunk_scores[q]])
            combined_i = np.concatenate([all_indices[q], chunk_global[q]])
            topk = np.argsort(-combined_s)[:actual_k]
            all_scores[q] = combined_s[topk]
            all_indices[q] = combined_i[topk]

        del db_chunk, sim, chunk_scores, chunk_local
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del q_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_indices, all_scores


def load_or_extract_db_features(
    split_dir: Path,
    db_records: Sequence[ImageRecord],
    model,
    preprocess,
    device: torch.device,
    args,
    tag: str,
):
    split_dir.mkdir(parents=True, exist_ok=True)
    feat_path = split_dir / "ffpe_features.npy"
    name_path = split_dir / "ffpe_filenames.npy"
    path_path = split_dir / "ffpe_paths.npy"
    case_path = split_dir / "ffpe_case_ids.npy"
    source_path = split_dir / "ffpe_sources.npy"

    if args.reuse_ffpe_cache and feat_path.exists() and name_path.exists() and case_path.exists():
        print(f"\nLoading cached {tag} DX features...")
        features = np.load(feat_path)
        names = np.load(name_path, allow_pickle=True).tolist()
        paths = np.load(path_path, allow_pickle=True).tolist() if path_path.exists() else [""] * len(names)
        case_ids = np.load(case_path, allow_pickle=True).tolist()
        sources = np.load(source_path, allow_pickle=True).tolist() if source_path.exists() else ["DX"] * len(names)
        print(f"  Loaded {features.shape[0]:,} DX features from {feat_path}")
        return names, features, paths, case_ids, sources

    print(f"\nExtracting {tag} DX features: {len(db_records):,} images")
    names, features, paths, case_ids, sources = extract_features_from_records(
        db_records, model, preprocess, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc=f"  {tag} DX features",
    )
    if features is None:
        raise RuntimeError(f"No valid DX features extracted for {tag}")

    np.save(feat_path, features)
    np.save(name_path, np.array(names, dtype=object))
    np.save(path_path, np.array(paths, dtype=object))
    np.save(case_path, np.array(case_ids, dtype=object))
    np.save(source_path, np.array(sources, dtype=object))
    print(f"  Saved {tag} DX cache to {split_dir}")
    return names, features, paths, case_ids, sources


def compute_lookup_for_split(
    split_name: str,
    query_records: Sequence[ImageRecord],
    db_bundle: Tuple[List[str], np.ndarray, List[str], List[str], List[str]],
    model,
    preprocess,
    device: torch.device,
    args,
    output_dir: Path,
    patient_to_label: Dict[str, str],
    db_scope: str,
):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    if not query_records:
        raise RuntimeError(f"No query images found for split: {split_name}")

    db_names, db_features, db_paths, db_case_ids, db_sources = db_bundle
    if db_features is None or len(db_names) == 0:
        raise RuntimeError(f"No DB features available for split: {split_name}")

    print(f"\n[{split_name}] Query features: {len(query_records):,} images")
    q_names, q_features, q_paths, q_case_ids, q_sources = extract_features_from_records(
        query_records, model, preprocess, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc=f"  {split_name} query features",
    )
    if q_features is None:
        raise RuntimeError(f"No valid query features extracted for split: {split_name}")

    print(f"\n[{split_name}] Top-K search")
    Q = len(q_names)
    qchunk = args.query_chunk_size
    actual_k = min(args.k, len(db_names))

    all_topk_indices = np.zeros((Q, actual_k), dtype=np.int32)
    all_topk_scores = np.zeros((Q, actual_k), dtype=np.float16)

    q_cases_for_exclusion = q_case_ids if args.exclude_same_case else None
    db_cases_for_exclusion = db_case_ids if args.exclude_same_case else None

    for qi in tqdm(range(0, Q, qchunk), desc=f"  {split_name} searching", ascii=True):
        qe = min(qi + qchunk, Q)
        topk_indices, topk_scores = chunked_topk_search(
            query_matrix=q_features[qi:qe],
            db_matrix=db_features,
            k=args.k,
            db_chunk_size=args.db_chunk_size,
            query_case_ids=q_cases_for_exclusion[qi:qe] if q_cases_for_exclusion else None,
            db_case_ids=db_cases_for_exclusion,
        )
        all_topk_indices[qi:qe] = topk_indices.astype(np.int32)
        all_topk_scores[qi:qe] = topk_scores.astype(np.float16)
        del topk_indices, topk_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lookup = {
        # Backward-compatible keys from the original script
        "fs_filenames": q_names,
        "ffpe_filenames": db_names,
        "fs_name2idx": {name: i for i, name in enumerate(q_names)},
        "topk_indices": all_topk_indices,
        "topk_scores": all_topk_scores,
        "k": actual_k,
        "fm": "UNI",
        "feat_dim": int(db_features.shape[1]),
        "exclude_same_case": bool(args.exclude_same_case),

        # Extra metadata for safer downstream use
        "split": split_name,
        "db_scope": db_scope,
        "fs_paths": q_paths,
        "ffpe_paths": db_paths,
        "fs_case_ids": q_case_ids,
        "ffpe_case_ids": db_case_ids,
        "fs_sources": q_sources,
        "ffpe_sources": db_sources,
        "patient_labels": {pid: patient_to_label.get(pid, "UNKNOWN") for pid in sorted(set(q_case_ids))},
    }

    lookup_path = split_dir / "rag_lookup.pt"
    torch.save(lookup, lookup_path)
    print(f"  Saved lookup: {lookup_path} ({lookup_path.stat().st_size / 1e6:.1f} MB)")

    if args.save_features:
        np.save(split_dir / "fs_features.npy", q_features)
        np.save(split_dir / "fs_filenames.npy", np.array(q_names, dtype=object))
        np.save(split_dir / "fs_paths.npy", np.array(q_paths, dtype=object))
        np.save(split_dir / "fs_case_ids.npy", np.array(q_case_ids, dtype=object))
        np.save(split_dir / "fs_sources.npy", np.array(q_sources, dtype=object))

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split_name,
        "db_scope": db_scope,
        "input_dir": str(args.input_dir),
        "gen_dir": str(args.gen_dir) if args.gen_dir else None,
        "query_mode": "TS+GEN_BS" if args.gen_dir else "TS+BS",
        "gen_key_prefix": args.gen_key_prefix if args.gen_dir else None,
        "dx_source": "DX-classified records under input_dir",
        "n_query": len(q_names),
        "n_db": len(db_names),
        "n_query_patients": len(set(q_case_ids)),
        "n_db_patients": len(set(db_case_ids)),
        "query_source_counts": dict(Counter(q_sources)),
        "db_source_counts": dict(Counter(db_sources)),
        "label_counts_query_patients": count_labels(sorted(set(q_case_ids)), patient_to_label),
        "k": actual_k,
        "requested_k": args.k,
        "exclude_same_case": bool(args.exclude_same_case),
        "test_db_scope": args.test_db_scope,
    }
    with (split_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return lookup_path


def main():
    args = parse_args()
    args.input_dir = Path(args.input_dir)
    args.gen_dir = Path(args.gen_dir) if args.gen_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DeepThaw / FS-to-FFPE RAG Precompute with Patient Split")
    print("=" * 70)
    print(f"  input_dir:     {args.input_dir}")
    print(f"  gen_dir:       {args.gen_dir}")
    print(f"  output_dir:    {output_dir}")
    print(f"  k:             {args.k}")
    print(f"  test_db_scope: {args.test_db_scope}")

    query_records, dx_records = build_records(args.input_dir, args.gen_dir, args.gen_key_prefix)
    if not query_records:
        raise RuntimeError("No query images found. Check that input_dir contains *-TS*/*-BS* slide folders, TS/BS folders, or pass --gen-dir")
    if not dx_records:
        raise RuntimeError("No DX images found. Check that input_dir contains *-DX* slide folders or a DX folder")

    query_cases = set(r.case_id for r in query_records)
    dx_cases = set(r.case_id for r in dx_records)
    data_eligible_cases = sorted(query_cases & dx_cases)
    if not data_eligible_cases:
        raise RuntimeError("No overlapping patient IDs between query images and DX images")

    # Split creation or reuse
    if args.split_json:
        print(f"\n[1/5] Loading existing split: {args.split_json}")
        train_ids, test_ids, split_labels = load_split_json(args.split_json)
        if args.label_file:
            patient_to_label = load_patient_labels(args.label_file, args.patient_col, args.label_col)
            patient_to_label.update({k: v for k, v in split_labels.items() if k not in patient_to_label})
        else:
            patient_to_label = split_labels
            if not patient_to_label:
                print("  ⚠️ Split JSON has no patient_labels and --label-file was not given. Label counts will be UNKNOWN.")
        eligible_patients = sorted(set(train_ids) | set(test_ids))
    else:
        print("\n[1/5] Creating new label-balanced split")
        if not args.label_file:
            raise ValueError("--label-file is required when --split-json is not provided")
        patient_to_label = load_patient_labels(args.label_file, args.patient_col, args.label_col)
        eligible_patients = sorted(set(data_eligible_cases) & set(patient_to_label.keys()))
        missing_label = sorted(set(data_eligible_cases) - set(patient_to_label.keys()))
        if missing_label:
            print(f"  ⚠️ {len(missing_label):,} image-eligible patients have no label and are excluded")
        if not eligible_patients:
            raise RuntimeError("No eligible patients after matching labels with available TS/BS/DX images")
        train_ids, test_ids = stratified_patient_split(patient_to_label, eligible_patients, args.test_ratio, args.seed)

    # Warn if split contains patients absent from current image mode.
    absent_train = sorted(set(train_ids) - set(data_eligible_cases))
    absent_test = sorted(set(test_ids) - set(data_eligible_cases))
    if absent_train:
        print(f"  ⚠️ {len(absent_train):,} train patients from split have no current query+DX images")
    if absent_test:
        print(f"  ⚠️ {len(absent_test):,} test patients from split have no current query+DX images")

    # Keep only patients available in this run.
    train_ids_available = sorted(set(train_ids) & set(data_eligible_cases))
    test_ids_available = sorted(set(test_ids) & set(data_eligible_cases))
    if not train_ids_available:
        raise RuntimeError("No available train patients after filtering current images")
    if not test_ids_available:
        raise RuntimeError("No available test patients after filtering current images")

    save_split_files(
        output_dir,
        train_ids_available,
        test_ids_available,
        patient_to_label,
        args,
        eligible_patients=sorted(set(train_ids_available) | set(test_ids_available)),
    )

    train_query_records = records_by_patients(query_records, train_ids_available)
    test_query_records = records_by_patients(query_records, test_ids_available)
    train_dx_records = records_by_patients(dx_records, train_ids_available)
    test_dx_records = records_by_patients(dx_records, test_ids_available)

    print("\n[2/5] Final split image counts")
    print(f"  train query: {len(train_query_records):,} images / {len(set(r.case_id for r in train_query_records)):,} patients")
    print(f"  test query:  {len(test_query_records):,} images / {len(set(r.case_id for r in test_query_records)):,} patients")
    print(f"  train DX:    {len(train_dx_records):,} images / {len(set(r.case_id for r in train_dx_records)):,} patients")
    print(f"  test DX:     {len(test_dx_records):,} images / {len(set(r.case_id for r in test_dx_records)):,} patients")

    if args.dry_run:
        print("\nDry run complete. Split files were written, but UNI features/RAG lookup were not computed.")
        print(f"  Split file: {output_dir / 'split.json'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3/5] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n[4/5] Loading UNI")
    model, preprocess = load_uni(device, args.uni_model, args.hf_token_env)

    # Extract DX DB features. Train DB is always needed for train and usually for test.
    train_db_bundle = load_or_extract_db_features(
        output_dir / "train", train_dx_records, model, preprocess, device, args, tag="train"
    )

    test_db_bundle = train_db_bundle
    test_db_scope_used = "train"
    if args.test_db_scope == "same_split":
        test_db_bundle = load_or_extract_db_features(
            output_dir / "test", test_dx_records, model, preprocess, device, args, tag="test"
        )
        test_db_scope_used = "same_split"

    print("\n[5/5] Computing lookup files")
    output_paths = []
    if args.only_split in ["train", "both"]:
        output_paths.append(compute_lookup_for_split(
            "train", train_query_records, train_db_bundle,
            model, preprocess, device, args, output_dir, patient_to_label,
            db_scope="train",
        ))

    if args.only_split in ["test", "both"]:
        output_paths.append(compute_lookup_for_split(
            "test", test_query_records, test_db_bundle,
            model, preprocess, device, args, output_dir, patient_to_label,
            db_scope=test_db_scope_used,
        ))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)
    for p in output_paths:
        print(f"  {p}")
    print(f"\nSplit file for reuse with generated BS:")
    print(f"  {output_dir / 'split.json'}")


if __name__ == "__main__":
    main()
