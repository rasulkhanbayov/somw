"""Free local storage, only after the archive is proven complete. Without --yes it deletes nothing: it prints the plan.

Guards (all must hold, each checked live, before anything is deleted):
  1. freeze/REMOTE_OK.json exists (check_remote.py passed for results, data, ckpt) and matches the current manifests;
  2. each Hugging Face repo is publicly reachable (asked without a token) at the revision check_remote.py verified;
  3. the GitHub repo is pushed: clean working tree, local HEAD = origin/main, tag freeze-2026-11-03 on the remote;
  4. the private repo (~/somw-private) is pushed: local HEAD = its origin/main.
Tiers (choose with --tier; only paths from freeze/areas.py under DATA_ROOT and BIG_ROOT, nothing else is ever touched):
  ours      our trajectories, generations, renders and checkpoints (verified on Hugging Face)
  scratch   upload trees, the verification copy, the redacted-record stage, caches, job scratch
  renderer  renderer weights, FLUX / Qwen2-VL / SV3D cache, PolyFEM source and builds (re-downloadable at pinned
            revisions: fetch_third_party.sh, polyfem/build_polyfem.sh)
  physctrl  PhysCtrl's checkout, weights, test and sample sets. Deleting these relies on PhysCtrl's own repos staying
            online, and no copy of them is in our archive, so keeping them (about 6 GB) is recommended.
Each deletion is appended to freeze/deleted_local.log.
Usage: python freeze/delete_local.py                        (plan for all tiers and guard status; deletes nothing)
       python freeze/delete_local.py --tier ours,scratch --yes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import BIG, DATA, FREEZE, PRIVATE_STAGE, THIRD_PARTY, UPLOAD, WS  # noqa: E402

TAG = "freeze-2026-11-03"
PRIVATE_REPO = Path.home() / "somw-private"


def tiers():
    return {
        "ours": [Path(a[1]) for a in UPLOAD if a[2] in ("data", "ckpt")],
        "scratch": [BIG / "hf_stage", BIG / "freeze_verify", BIG / "freeze_verify.log", BIG / "hf_upload_data.log",
                    PRIVATE_STAGE, BIG / "tmp", BIG / "checkpoints/.cache", DATA / "checkpoints/.cache", DATA / "warp_cache"],
        "renderer": [BIG / "hf_cache", BIG / "checkpoints/checkpoints", BIG / "tools"],
        "physctrl": [Path(t[1]) for t in THIRD_PARTY if t[0].startswith("tp_physctrl")],
    }


def git(*args, cwd=WS):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120)


def remote_head(cwd):
    r = git("ls-remote", "origin", "refs/heads/main", cwd=cwd)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None


def guards():
    """Returns the list of unmet conditions (empty = safe to delete)."""
    bad = []
    stamp = FREEZE / "REMOTE_OK.json"
    if not stamp.exists():
        return ["freeze/REMOTE_OK.json is missing: uploads are not verified yet (run check_remote.py)"]
    st = json.loads(stamp.read_text())
    if st["manifest_sha256"] != hashlib.sha256((FREEZE / "manifests" / "SUMMARY.json").read_bytes()).hexdigest():
        bad.append("the manifests changed after check_remote.py ran: run it again")
    from huggingface_hub import HfApi
    api = HfApi(token=False)                       # no token: this is what a stranger can reach
    for key in ("results", "data", "ckpt"):
        if key not in st["repos"]:
            bad.append(f"repo {key} was not verified")
            continue
        r = st["repos"][key]
        try:
            info = api.repo_info(r["repo_id"], repo_type=r["repo_type"])
            if info.private or info.sha != r["revision"]:
                bad.append(f"{r['repo_id']}: private or changed since it was verified (run check_remote.py again)")
        except Exception as ex:                     # noqa: BLE001
            bad.append(f"{r['repo_id']}: not publicly reachable ({type(ex).__name__})")
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        bad.append("no git repository here: GitHub push not done")
    else:
        if git("status", "--porcelain").stdout.strip():
            bad.append("git working tree not clean: commit and push first")
        if remote_head(WS) != git("rev-parse", "HEAD").stdout.strip():
            bad.append("GitHub main is not at local HEAD: push first")
        if TAG not in git("ls-remote", "--tags", "origin", TAG).stdout:
            bad.append(f"tag {TAG} is not on GitHub")
    if not (PRIVATE_REPO / ".git").exists():
        bad.append(f"{PRIVATE_REPO} not found: private material not pushed (COMMANDS.md step 6b)")
    elif remote_head(PRIVATE_REPO) != git("rev-parse", "HEAD", cwd=PRIVATE_REPO).stdout.strip():
        bad.append("somw-private is not pushed")
    else:                                          # it must not be readable by a stranger (added Nov 8: it was public)
        import re
        import urllib.error
        import urllib.request
        url = git("remote", "get-url", "origin", cwd=PRIVATE_REPO).stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
        if not m:
            bad.append(f"cannot tell whether {url} is private")
        else:
            try:
                urllib.request.urlopen(f"https://api.github.com/repos/{m[1]}/{m[2]}", timeout=30)
                bad.append(f"github.com/{m[1]}/{m[2]} is PUBLIC: make it private (Settings > Danger Zone)")
            except urllib.error.HTTPError as ex:
                if ex.code != 404:                 # 404 without a token = private (or gone; the push check covers that)
                    bad.append(f"cannot tell whether {m[1]}/{m[2]} is private (HTTP {ex.code})")
            except OSError as ex:
                bad.append(f"cannot tell whether {m[1]}/{m[2]} is private ({type(ex).__name__})")
    return bad


def size(path):
    r = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default=None, help="comma-separated: ours, scratch, renderer, physctrl")
    ap.add_argument("--yes", action="store_true", help="really delete (needs --tier and all guards)")
    args = ap.parse_args()
    t = tiers()
    names = args.tier.split(",") if args.tier else list(t)
    if any(n not in t for n in names):
        sys.exit(f"unknown tier; choose from {list(t)}")
    if args.yes and not args.tier:
        sys.exit("--yes needs an explicit --tier")
    for n in names:                                # the table in areas.py is the only source, but check anyway
        for p in t[n]:
            if not (p.is_relative_to(DATA) or p.is_relative_to(BIG)) or p in (DATA, BIG):
                sys.exit(f"refusing to touch {p}")
    bad = guards()
    print("Guards: " + ("all met" if not bad else f"{len(bad)} NOT met, nothing can be deleted"))
    for b in bad:
        print("  - " + b)
    total = 0
    for n in names:
        sz = 0
        print(f"\n{n}:")
        for p in t[n]:
            if p.exists() or p.is_symlink():
                s = size(p)
                sz += s
                print(f"  {s / 1e9:8.2f} GB  {p}")
        print(f"  {sz / 1e9:8.2f} GB  total")
        total += sz
    print(f"\nPlan total: {total / 1e9:.1f} GB")
    if not args.yes:
        print("Nothing deleted (dry run). To delete: --tier <names> --yes")
        return
    if bad:
        sys.exit("Nothing deleted: guards not met.")
    with open(FREEZE / "deleted_local.log", "a") as log:
        for n in names:
            for p in t[n]:
                if not (p.exists() or p.is_symlink()):
                    continue
                s = size(p)
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)            # removes links inside, never follows them
                else:
                    p.unlink()
                log.write(f"{time.strftime('%F %T')}\t{n}\t{s}\t{p}\n")
                print(f"deleted {p}", flush=True)
    print("Done. Restore at any time: freeze/COMMANDS.md, 'Restore and reproduce'.")


if __name__ == "__main__":
    main()
