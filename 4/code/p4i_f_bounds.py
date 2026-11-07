"""Phase 4i: singular-value bounds for the stable F-lift, from training objects only. CPU.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4i"), before this ran: s_lo, s_hi = the 0.1% and
99.9% quantiles of the singular values of the stored ground-truth F, over all particles and the
odd frames 1..47, of the first file (sorted) of each 4g training object in data/fc_train/.
Those objects are disjoint from every evaluation object. The bounds are fixed afterwards.

Output: results/p4i_f_bounds/bounds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, start_run  # noqa: E402
from physctrl_bridge import load_h5  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run("p4i_f_bounds", cfg, synthetic=False)
    first = {}
    for f in sorted(Path(cfg["paths"]["fc_train_dir"]).glob("*/*.h5")):
        first.setdefault(f.name.split("__")[0], f)          # object id -> its first file
    S = []
    for f in first.values():
        F = np.asarray(load_h5(f)["F"])[1::2].reshape(-1, 3, 3)
        S.append(np.linalg.svd(F, compute_uv=False).ravel())
    S = np.concatenate(S)
    lo, hi = (float(np.quantile(S, q)) for q in cfg["F_bound_quantiles"])
    out = {"s_lo": lo, "s_hi": hi, "n_objects": len(first), "n_values": int(S.size),
           "quantiles": cfg["F_bound_quantiles"], "min": float(S.min()), "max": float(S.max())}
    run.path("bounds.json").write_text(json.dumps(out, indent=1))
    print(f"[p4i_f_bounds] {out}")
    run.finish("ok")


if __name__ == "__main__":
    main()
