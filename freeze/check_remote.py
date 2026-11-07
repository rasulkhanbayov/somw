"""After uploading: check that each Hugging Face repo holds every manifest file with the same size and content.

For each UPLOAD area of freeze/areas.py whose repo is given, every manifest line must appear in the remote repo at
<path in repo>/<path> with the same size and, for LFS/Xet files, the same sha256 (small files stored in git: the git blob
id is compared with the one computed from the local file). Files that are links locally were uploaded as the file they
point to, so they are compared with that file. Extra remote files that no manifest lists are reported too (apart from
README.md, VERIFY*.md, manifests/, env/, third_party/ and .gitattributes). Run it while the local files still exist.
If everything matches for results, data and ckpt, it writes freeze/REMOTE_OK.json (revisions, manifest hash), which
delete_local.py requires.
Usage: python freeze/check_remote.py --repo results=USER/somw-results:dataset --repo data=USER/somw-data:dataset \
                                     --repo ckpt=USER/somw-checkpoints:model --repo private=USER/somw-private:dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import FREEZE, UPLOAD  # noqa: E402
from make_manifest import sha256  # noqa: E402

EXTRA_OK = ("README.md", "VERIFY", ".gitattributes", "manifests/", "env/", "third_party/")


def git_blob_id(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", required=True, help="key=repo_id:repo_type")
    args = ap.parse_args()
    api, bad, stamp = HfApi(), 0, {}
    for spec in args.repo:
        key, rest = spec.split("=", 1)
        repo_id, repo_type = rest.rsplit(":", 1)
        rev = api.repo_info(repo_id, repo_type=repo_type).sha
        remote = {f.path: f for f in api.list_repo_tree(repo_id, repo_type=repo_type, revision=rev, recursive=True)
                  if getattr(f, "size", None) is not None}
        n_ok = 0
        seen = set()
        for area, src, repo, dst, _ in UPLOAD:
            if repo != key:
                continue
            rows = [line.split("\t") for line in (FREEZE / "manifests" / f"{area}.tsv").read_text().splitlines()[1:]]
            wrong = []
            for rel, size, digest in rows:
                path = dst if Path(src).is_file() else f"{dst}/{rel}"
                seen.add(path)
                f = remote.get(path)
                local = Path(src) if Path(src).is_file() else Path(src) / rel
                if digest.startswith("symlink->"):      # uploaded as the file it points to
                    size, digest = local.stat().st_size, sha256(local)
                if f is None:
                    wrong.append(f"missing: {path}")
                elif f.size != int(size):
                    wrong.append(f"size {f.size} != {size}: {path}")
                elif f.lfs is not None and f.lfs.sha256 != digest:
                    wrong.append(f"sha256 differs: {path}")
                elif f.lfs is None and f.blob_id != git_blob_id(local):
                    wrong.append(f"content differs: {path}")
            bad += len(wrong)
            n_ok += len(rows) - len(wrong)
            print(f"{repo_id} {area}: {len(rows) - len(wrong)}/{len(rows)} ok")
            for w in wrong[:20]:
                print("   " + w)
        extra = [p for p in remote if p not in seen and not p.startswith(EXTRA_OK)]
        if extra:
            print(f"{repo_id}: {len(extra)} remote files in no manifest, e.g. {extra[:5]}")
        stamp[key] = {"repo_id": repo_id, "repo_type": repo_type, "revision": rev, "files_ok": n_ok}
    print("REMOTE MATCHES THE MANIFESTS" if bad == 0 else f"{bad} problems")
    if bad == 0 and all(k in stamp for k in ("results", "data", "ckpt")):
        (FREEZE / "REMOTE_OK.json").write_text(json.dumps({
            "checked_utc": time.strftime("%FT%TZ", time.gmtime()), "repos": stamp,
            "manifest_sha256": hashlib.sha256((FREEZE / "manifests" / "SUMMARY.json").read_bytes()).hexdigest()}, indent=1) + "\n")
        print("wrote freeze/REMOTE_OK.json")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
