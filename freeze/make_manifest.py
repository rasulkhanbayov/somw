"""Checksum manifest of everything the archive covers (run on a dev server before uploading; read-only).

For every UPLOAD and THIRD_PARTY area of freeze/areas.py: one line per file (path relative to the area, size in bytes,
sha256) in freeze/manifests/<area>.tsv, plus freeze/manifests/SUMMARY.json (files and bytes per area and per repo).
Completeness guard: every entry directly under data/ and $BIG_ROOT must be covered by an UPLOAD, THIRD_PARTY or EXCLUDED
root, otherwise the script stops and names it (nothing is silently left out).
Symlinks inside an area are recorded as links (target, no hash) and not followed.
Usage: python freeze/make_manifest.py [--workers 16] [--only area1,area2]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import BIG, DATA, EXCLUDED, FREEZE, THIRD_PARTY, UPLOAD, covered_roots  # noqa: E402

OUT = FREEZE / "manifests"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(root, exclude):
    root = Path(root)
    if root.is_file():
        return [(root.name, root)]
    out = []
    for d, dirs, fs in os.walk(root):
        dirs.sort()
        for f in sorted(fs):
            p = Path(d) / f
            rel = p.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, g) for g in exclude):
                continue
            out.append((rel, p))
    return out


def one(args):
    rel, p = args
    if p.is_symlink():
        return rel, -1, "symlink->" + os.readlink(p)
    return rel, p.stat().st_size, sha256(p)


def completeness():
    """Every path under data/ and $BIG_ROOT must lie inside a covered root; folders that only contain covered roots
    (e.g. data/checkpoints) are opened and their other children checked too."""
    roots = [Path(r) for r in covered_roots()]
    missing = []

    def visit(e):
        if e.name.startswith(".nfs") or any(e == r or e.is_relative_to(r) for r in roots):
            return
        if e.is_dir() and any(r.is_relative_to(e) for r in roots):
            for c in sorted(e.iterdir()):
                visit(c)
            return
        missing.append(str(e))
    for top in (DATA, BIG):
        for e in sorted(top.iterdir()):
            visit(e)
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    miss = completeness()
    if miss:
        sys.exit("Not covered by freeze/areas.py (add each to UPLOAD, THIRD_PARTY or EXCLUDED):\n  " + "\n  ".join(miss))
    OUT.mkdir(parents=True, exist_ok=True)
    areas = [(a, src, repo, dst, ex) for a, src, repo, dst, ex in UPLOAD] + [(a, src, "third_party", None, ex) for a, src, ex in THIRD_PARTY]
    if args.only:
        keep = set(args.only.split(","))
        areas = [a for a in areas if a[0] in keep]
    summary = json.loads((OUT / "SUMMARY.json").read_text()) if (OUT / "SUMMARY.json").exists() else {"areas": {}}
    with ProcessPoolExecutor(args.workers) as ex:
        for area, src, repo, dst, exclude in areas:
            if not Path(src).exists():
                if area == "private_session":
                    print(f"[manifest] {area}: {src} not made yet (run redact_transcript.py first); skipped", flush=True)
                    continue
                sys.exit(f"[manifest] {area}: source {src} does not exist")
            rows = list(ex.map(one, files(src, exclude), chunksize=8))
            with open(OUT / f"{area}.tsv", "w") as f:
                f.write("path\tbytes\tsha256\n")
                for r in rows:
                    f.write("\t".join(map(str, r)) + "\n")
            n_bytes = sum(r[1] for r in rows if r[1] > 0)
            summary["areas"][area] = {"source": str(src), "repo": repo, "path_in_repo": dst, "files": len(rows),
                                      "bytes": n_bytes, "symlinks": sum(r[1] < 0 for r in rows)}
            print(f"[manifest] {area}: {len(rows)} files, {n_bytes / 1e9:.2f} GB", flush=True)
    per_repo = {}
    for v in summary["areas"].values():
        p = per_repo.setdefault(v["repo"], {"files": 0, "bytes": 0})
        p["files"] += v["files"]
        p["bytes"] += v["bytes"]
    summary["repos"] = per_repo
    summary["excluded"] = {str(k): v for k, v in EXCLUDED.items()}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    for k, v in per_repo.items():
        print(f"[manifest] repo {k}: {v['files']} files, {v['bytes'] / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
