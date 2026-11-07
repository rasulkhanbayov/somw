"""Phase 11 (c) preference study: item files for the study page (Oct 21, 2026; design pre-registered Oct 20, PLAN.md
"Phase 11"). CPU.

Items: every rejected shifted case (force_coeff 0.40 Phase 11 images and PhysCtrl's 3x-force examples), as in
p11_summary.py. Per item, the two candidate videos (PhysCtrl = generation, ours = certified_v2) are copied under opaque
names <item>_a.mp4 / <item>_b.mp4, with a/b assigned by a fixed seed, plus the reference as <item>_r.mp4; the page never
learns which is which. Item codes are random tokens (seed 11), so file names reveal nothing.
Outputs:
  <out>/v/*.mp4 and <out>/items.json   (published with the page: prompt and file names only)
  results/p11_study_design/mapping.json (local only, never published: item -> case, which letter is ours)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, load_config  # noqa: E402

SEED = 11


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/p11_judge.yaml (render roots)")
    ap.add_argument("--out", required=True, help="folder for the published files (outside results/)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cases = []
    for root, ex_root in cfg["render_roots"]:
        root, ex_root = Path(os.path.expandvars(root)), Path(os.path.expandvars(ex_root))
        for c in sorted(root.glob("*/case_v2.json")):
            rec = json.loads(c.read_text())
            name = rec["case"]
            shifted = name.endswith("__coeff_0.4") or name.endswith("__force_x3")
            if not shifted or rec["decision"]["accepted"]:
                continue
            ex = rec.get("example") or name.split("__")[0]
            prompt = json.loads((ex_root / ex / "config.json").read_text())["prompt"]
            vids = {v: c.parent / "variants" / v / "result.mp4" for v in ("generation", "certified_v2", "ground_truth")}
            if not all(p.exists() for p in vids.values()):
                raise RuntimeError(f"{name}: missing render")
            cases.append((name, prompt, vids))
    rng = np.random.default_rng(SEED)
    out = Path(args.out)
    (out / "v").mkdir(parents=True, exist_ok=True)
    items, mapping, used = [], {}, set()
    for name, prompt, vids in cases:
        while True:
            code = "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), 5))
            if code not in used:
                used.add(code)
                break
        ours_letter = "a" if rng.random() < 0.5 else "b"
        src = {ours_letter: vids["certified_v2"], ("b" if ours_letter == "a" else "a"): vids["generation"], "r": vids["ground_truth"]}
        for k, p in src.items():
            shutil.copyfile(p, out / "v" / f"{code}_{k}.mp4")
        items.append({"id": code, "prompt": prompt})
        mapping[code] = {"case": name, "ours": ours_letter}
    (out / "items.json").write_text(json.dumps(items, indent=1))
    md = Path(RESULTS_ROOT) / "p11_study_design"
    md.mkdir(parents=True, exist_ok=True)
    (md / "mapping.json").write_text(json.dumps({"seed": SEED, "items": mapping}, indent=1))
    print(f"[p11 study] {len(items)} items -> {out}; mapping -> {md / 'mapping.json'}")


if __name__ == "__main__":
    main()
