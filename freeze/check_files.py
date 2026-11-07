"""Check files on disk against the freeze-time manifests (after a restore, or before deleting anything).

For each area: every manifest line must exist at <area source>/<path> (sources from freeze/areas.py, i.e. from
PROJECT_ROOT / DATA_ROOT / BIG_ROOT) with the same size and sha256 (--fast: size only). Symlinks must point to the same
target. Prints a per-area count and every missing or different file; exit 1 if any.
Usage: python freeze/check_files.py --manifests freeze/manifests [--areas a,b] [--fast] [--workers 16]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import THIRD_PARTY, UPLOAD  # noqa: E402
from make_manifest import sha256  # noqa: E402


def check(args):
    root, rel, size, digest, fast = args
    p = Path(root) / rel if Path(root).is_dir() or rel != Path(root).name else Path(root)
    if digest.startswith("symlink->"):
        # uploads replace links by the file they point to, so a restored copy holds a real file here
        return rel, "ok" if p.exists() or (p.is_symlink() and os.readlink(p) == digest[9:]) else "missing"
    if not p.exists():
        return rel, "missing"
    if p.stat().st_size != int(size):
        return rel, f"size {p.stat().st_size} != {size}"
    if not fast and sha256(p) != digest:
        return rel, "sha256 differs"
    return rel, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", default=str(Path(__file__).resolve().parent / "manifests"))
    ap.add_argument("--areas", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    src = {a[0]: a[1] for a in UPLOAD} | {t[0]: t[1] for t in THIRD_PARTY}
    names = args.areas.split(",") if args.areas else sorted(src)
    bad = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for a in names:
            f = Path(args.manifests) / f"{a}.tsv"
            if not f.exists():
                print(f"{a}: no manifest {f}")
                bad += 1
                continue
            rows = [line.rstrip("\n").split("\t") for line in f.read_text().splitlines()[1:]]
            res = list(ex.map(check, [(src[a], r[0], r[1], r[2], args.fast) for r in rows], chunksize=8))
            wrong = [(r, s) for r, s in res if s != "ok"]
            bad += len(wrong)
            print(f"{a}: {len(res) - len(wrong)}/{len(res)} ok" + ("" if not wrong else ""))
            for r, s in wrong[:50]:
                print(f"   {s}: {r}")
            if len(wrong) > 50:
                print(f"   ... and {len(wrong) - 50} more")
    print("ALL FILES MATCH" if bad == 0 else f"{bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
