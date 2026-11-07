"""Put downloaded Hugging Face repos back where the code expects them (the inverse of stage_hf.py).

Download each repo first (COMMANDS.md, "Restore"), e.g. into $DL/{results,data,ckpt,private}. Then, for every UPLOAD
area of freeze/areas.py, this moves <DL>/<repo>/<path in repo> to the area's source path (under PROJECT_ROOT, DATA_ROOT,
BIG_ROOT from 4/slurm/cluster.env). It refuses to overwrite an existing non-empty target unless --merge is given.
Afterwards: freeze/check_files.py (sizes and sha256 against the manifests) and, if the paths differ from the freeze-time
ones, freeze/remap_paths.py.
Usage: python freeze/restore.py --download-dir $DL [--repos results,data,ckpt] [--merge] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import UPLOAD  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--repos", default="results,data,ckpt")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dl = Path(args.download_dir)
    for area, src, repo, dst, _ in UPLOAD:
        if repo not in args.repos.split(","):
            continue
        frm = dl / repo / dst
        if not frm.exists():
            print(f"{area}: {frm} not downloaded, skipped")
            continue
        src = Path(src)
        if src.exists() and (src.is_file() or any(src.iterdir())) and not args.merge:
            sys.exit(f"{area}: {src} exists and is not empty (use --merge to add into it)")
        print(f"{area}: {frm} -> {src}")
        if args.dry_run:
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        if frm.is_file():
            shutil.move(str(frm), str(src))
            src.chmod(0o755)
        else:
            src.mkdir(parents=True, exist_ok=True)
            for p in sorted(frm.rglob("*")):
                if p.is_file():
                    t = src / p.relative_to(frm)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(t))


if __name__ == "__main__":
    main()
