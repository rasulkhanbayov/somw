"""Rewrite the freeze-time absolute paths stored inside result files, for a restore at different locations.

Result files (manifests of query files, run_meta.json, decisions, configs saved in runs) contain absolute paths of the
original machine. This replaces the three prefixes in every text file (.json .jsonl .md .yaml .yml .txt .csv .tsv) under
the given folders, in place. Run it after restore.py and before re-running anything; a copy of each changed file is
kept as <file>.prefreeze unless --no-backup.
Usage: python freeze/remap_paths.py --project NEW_PROJECT_ROOT --data NEW_DATA_ROOT --big NEW_BIG_ROOT [folders ...]
       (default folders: <NEW_PROJECT_ROOT>/results, <NEW_PROJECT_ROOT>/logs)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OLD = {"project": "/export/home/rakh90100/cvpr_physics_plausibility/4",
       "data": "/export/home/rakh90100/cvpr_physics_plausibility/data",
       "big": "/export/hbku-research/rakh90100/somw"}
TEXT = {".json", ".jsonl", ".md", ".yaml", ".yml", ".txt", ".csv", ".tsv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("folders", nargs="*")
    args = ap.parse_args()
    new = {"project": args.project.rstrip("/"), "data": args.data.rstrip("/"), "big": args.big.rstrip("/")}
    folders = args.folders or [f"{new['project']}/results", f"{new['project']}/logs"]
    changed = 0
    for d in folders:
        for p in Path(d).rglob("*"):
            if not p.is_file() or p.suffix not in TEXT:
                continue
            s = p.read_text(errors="surrogateescape")
            t = s
            for k in ("project", "data", "big"):          # project first: it is not a prefix of the others
                t = t.replace(OLD[k], new[k])
            if t != s:
                if not args.no_backup:
                    shutil.copyfile(p, str(p) + ".prefreeze")
                p.write_text(t, errors="surrogateescape")
                changed += 1
    print(f"rewrote {changed} files")


if __name__ == "__main__":
    main()
