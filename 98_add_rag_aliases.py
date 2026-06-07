#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
from pathlib import Path
from collections import defaultdict
import torch

def no_ext(s):
    p = Path(str(s))
    return str(p.with_suffix("")) if p.suffix else str(s)

def strip_domain_prefix(s):
    parts = str(s).replace("\\", "/").split("/")
    return "/".join(parts[1:]) if parts and parts[0] in {"trainA", "testA", "valA"} else str(s)

def add_alias(mapping, alias, idx, collisions):
    if alias is None:
        return
    alias = str(alias).replace("\\", "/")
    if not alias:
        return
    if alias in mapping and mapping[alias] != idx:
        collisions[alias].add(mapping[alias]); collisions[alias].add(idx)
        return
    mapping[alias] = idx

def process_file(path):
    obj = torch.load(path, map_location="cpu")
    fs_names = [str(x) for x in obj["fs_filenames"]]
    fs_paths = [str(x) for x in obj.get("fs_paths", [""] * len(fs_names))]
    mapping = dict(obj.get("fs_name2idx", {name: i for i, name in enumerate(fs_names)}))
    collisions = defaultdict(set)

    stem_to_indices = defaultdict(set)
    for i, name in enumerate(fs_names):
        stem_to_indices[Path(name).stem].add(i)
    for i, p in enumerate(fs_paths):
        if p:
            stem_to_indices[Path(p).stem].add(i)

    for i, name in enumerate(fs_names):
        candidates = {name, no_ext(name), strip_domain_prefix(name), no_ext(strip_domain_prefix(name))}
        if fs_paths[i]:
            ap = os.path.abspath(fs_paths[i])
            rp = os.path.realpath(fs_paths[i])
            candidates |= {ap, no_ext(ap), rp, no_ext(rp)}
            norm = ap.replace("\\", "/")
            for token in ["/trainA/", "/testA/", "/valA/"]:
                if token in norm:
                    rel = norm.split(token, 1)[1]
                    domain = token.strip("/")
                    candidates |= {f"{domain}/{rel}", no_ext(f"{domain}/{rel}"), rel, no_ext(rel)}
        stem = Path(name).stem
        if len(stem_to_indices.get(stem, [])) == 1:
            candidates.add(stem)
        for c in candidates:
            add_alias(mapping, c, i, collisions)

    obj["fs_name2idx"] = mapping
    obj["fs_name2idx_alias_info"] = {
        "original_n": len(fs_names),
        "alias_n": len(mapping),
        "collision_n": len(collisions),
        "collision_examples": list(collisions.keys())[:20],
    }
    torch.save(obj, path)
    print(f"Updated {path}")
    print(f"  original fs_filenames: {len(fs_names):,}")
    print(f"  fs_name2idx aliases : {len(mapping):,}")
    print(f"  collisions skipped  : {len(collisions):,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-cache-dir", required=True)
    args = ap.parse_args()
    root = Path(args.rag_cache_dir)
    for p in [root / "train" / "rag_lookup.pt", root / "test" / "rag_lookup.pt"]:
        if p.exists():
            process_file(p)
        else:
            print(f"Skip missing: {p}")

if __name__ == "__main__":
    main()
