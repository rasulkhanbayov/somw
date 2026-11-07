"""Build the upload trees for the Hugging Face repos without copying data: $BIG_ROOT/hf_stage/<repo>/ mirrors the repo
layout with one symlink per file (hf upload-large-folder reads through file symlinks), plus each repo's README card and,
for the results repo, the checksum manifests. Run after make_manifest.py; re-running rebuilds the tree from scratch.
Every staged file is checked against the manifest's file count per area (a mismatch stops the script).
Usage: python freeze/stage_hf.py [--repos results,data,ckpt,private]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import BIG, FREEZE, UPLOAD  # noqa: E402
from make_manifest import files  # noqa: E402

STAGE = BIG / "hf_stage"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", default="results,data,ckpt,private")
    args = ap.parse_args()
    summary = json.loads((FREEZE / "manifests" / "SUMMARY.json").read_text())
    for repo in args.repos.split(","):
        root = STAGE / repo
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        n = 0
        for area, src, r, dst, exclude in UPLOAD:
            if r != repo:
                continue
            if area not in summary["areas"]:
                sys.exit(f"{area}: not in the manifest; run make_manifest.py first")
            fs = files(src, exclude)
            if len(fs) != summary["areas"][area]["files"]:
                sys.exit(f"{area}: {len(fs)} files now, {summary['areas'][area]['files']} in the manifest; re-run make_manifest.py")
            for rel, p in fs:
                t = root / dst if Path(src).is_file() else root / dst / rel
                t.parent.mkdir(parents=True, exist_ok=True)
                t.symlink_to(p.resolve())
                n += 1
        if repo == "results":                       # manifests and environment records travel with the public repo
            shutil.copytree(FREEZE / "manifests", root / "manifests")
            shutil.copytree(FREEZE / "env", root / "env")
            shutil.copytree(FREEZE / "third_party", root / "third_party")
            for f in ("VERIFY.md", "VERIFY_NOTES.md"):
                if (FREEZE / f).exists():
                    shutil.copyfile(FREEZE / f, root / f)
        shutil.copyfile(FREEZE / "cards" / f"{repo}.md", root / "README.md")
        bad = [p for p in root.rglob("*") if p.is_symlink() and not p.exists()]
        if bad:
            sys.exit(f"{repo}: {len(bad)} broken links, e.g. {bad[0]}")
        print(f"[stage] {repo}: {n} data files linked under {root}")


if __name__ == "__main__":
    main()
