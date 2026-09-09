#!/usr/bin/env python
"""
scripts/prepare_data.py
-----------------------
Unpack the downloaded archives into the layout every dataset class in
data/datasets/ expects. Idempotent: skips any split already extracted.

Expected archives in the repo root (download links in data/README.md):

    LOLdataset.zip        LOL: our485/{low,high} + eval15/{low,high}
    BrighteningTrain.zip  the paper's 1000 synthetic pairs
    Test.zip              the unpaired benchmark bundle (LIME/MEF/DICM/...)

Produces:

    data/lol/our485/{low,high}      485 real training pairs
    data/lol/eval15/{low,high}       15 real eval pairs   (canonical split)
    data/lol/syn/{low,high}         1000 synthetic training pairs
    data/cross_eval/{LIME,MEF,DICM} unpaired, no ground truth

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --root . --force
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Archive member prefix -> destination directory, relative to the repo root.
# Everything under the prefix is flattened into the destination.
EXTRACTIONS = [
    ("LOLdataset.zip", "our485/low", "data/lol/our485/low"),
    ("LOLdataset.zip", "our485/high", "data/lol/our485/high"),
    ("LOLdataset.zip", "eval15/low", "data/lol/eval15/low"),
    ("LOLdataset.zip", "eval15/high", "data/lol/eval15/high"),
    ("BrighteningTrain.zip", "BrighteningTrain/low", "data/lol/syn/low"),
    ("BrighteningTrain.zip", "BrighteningTrain/high", "data/lol/syn/high"),
    ("Test.zip", "Test/LIME", "data/cross_eval/LIME"),
    ("Test.zip", "Test/MEF", "data/cross_eval/MEF"),
    ("Test.zip", "Test/DICM", "data/cross_eval/DICM"),
]

# macOS zips carry a parallel __MACOSX/ tree of AppleDouble resource forks
# ("._name.png"). They are not images and PIL chokes on them, so drop them.
def _is_junk(name: str) -> bool:
    tail = name.rsplit("/", 1)[-1]
    return (
        name.startswith("__MACOSX/")
        or tail.startswith(".")     # ._resource forks and .DS_Store
        or tail == ""               # directory entries
    )


def extract(archive: Path, prefix: str, dest: Path, force: bool) -> int:
    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"  skip   {dest}  (already populated, --force to redo)")
        return 0
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if _is_junk(member) or not member.startswith(prefix + "/"):
                continue
            with zf.open(member) as src, open(dest / Path(member).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    print(f"  ok     {dest}  ({count} files)")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repo root holding the .zip archives")
    parser.add_argument("--force", action="store_true", help="Re-extract even if the destination is populated")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    missing = sorted({a for a, _, _ in EXTRACTIONS if not (root / a).exists()})
    if missing:
        print(f"Missing archive(s) in {root}: {', '.join(missing)}", file=sys.stderr)
        print("See data/README.md for download links.", file=sys.stderr)
        return 1

    for archive, prefix, dest in EXTRACTIONS:
        extract(root / archive, prefix, root / dest, args.force)

    print("\nDone. Sanity check:")
    for d in ["data/lol/our485/low", "data/lol/eval15/low", "data/lol/syn/low",
              "data/cross_eval/LIME", "data/cross_eval/MEF", "data/cross_eval/DICM"]:
        p = root / d
        n = len(list(p.iterdir())) if p.exists() else 0
        print(f"  {d:<28} {n:>5} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
