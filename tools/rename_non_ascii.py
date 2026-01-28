#!/usr/bin/env python3
"""
Rename any non-ASCII filenames under the repository to ASCII-safe names.
- Keeps extensions
- Writes a mapping to tools/rename_map.txt
Usage:
  python tools/rename_non_ascii.py
"""
from __future__ import annotations
import os, re, unicodedata, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAP_PATH = os.path.join(ROOT, "tools", "rename_map.txt")

def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False

def safe_slug(name: str, max_len: int = 48) -> str:
    # normalize -> remove accents -> keep alnum/_/-
    n = unicodedata.normalize("NFKD", name)
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    n = re.sub(r"[^A-Za-z0-9._-]+", "_", n).strip("_")
    if not n:
        n = "file"
    return n[:max_len]

def unique_name(dirpath: str, base: str, ext: str) -> str:
    cand = f"{base}{ext}"
    if not os.path.exists(os.path.join(dirpath, cand)):
        return cand
    for i in range(2, 9999):
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(os.path.join(dirpath, cand)):
            return cand
    # fallback hash
    h = hashlib.sha1((base+ext).encode("utf-8")).hexdigest()[:8]
    cand = f"{base}_{h}{ext}"
    return cand

def main():
    mapping = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # skip .git
        if os.path.basename(dirpath) == ".git":
            continue
        for fn in filenames:
            if is_ascii(fn):
                continue
            src = os.path.join(dirpath, fn)
            stem, ext = os.path.splitext(fn)
            base = safe_slug(stem)
            new_fn = unique_name(dirpath, base, ext)
            dst = os.path.join(dirpath, new_fn)
            os.rename(src, dst)
            rel_src = os.path.relpath(src, ROOT)
            rel_dst = os.path.relpath(dst, ROOT)
            mapping.append((rel_src, rel_dst))
            print(f"renamed: {rel_src} -> {rel_dst}")

    os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        for a,b in mapping:
            f.write(f"{a}\t{b}\n")
    print(f"\nDone. Mapping written to {os.path.relpath(MAP_PATH, ROOT)}")

if __name__ == "__main__":
    main()
