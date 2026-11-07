"""Phase 12 S0 diagnosis (CPU, mesh only, no simulation; Oct 24, 2026; PLAN.md "Phase 12", S0 run 1 FAIL).

Freezes as a script the inline analysis of Oct 24: on the 20 S0b/S0c queries of run 1 (job 432388), the ratio between
the mesh volume inside the drag box (Monte Carlo, 2,000 points per occupied cell near the box, seed 0) and the masked
particles' volume (what run 1 normalised the force by), its correlation with the S0c errors, and the ratio for the
snapped box used by v2.
Output: results/p12_s0_diagnose/{SUMMARY.md, diagnose.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, load_json, save_json, start_run  # noqa: E402
import p12_s0 as S  # noqa: E402
from p12_fem import BOX, DX, load_query  # noqa: E402


def main():
    r = load_json(Path(RESULTS_ROOT) / "p12_s0" / "s0.json")
    paths = S.valid_queries("id_new_1", 20, cond=lambda q: np.log10(float(q["E"])) >= 6)
    U = np.random.default_rng(0).random((2000, 3))
    rows = []
    for p, rel in zip(paths, r["s0c_rel"]):
        q = load_query(p)
        x0, vol = q["x0"], q["vol"].reshape(-1)
        m = q["drag_mask"].reshape(-1).astype(bool)
        dp = q["drag_point"].reshape(-1)[:3]
        cells = np.unique(np.floor(x0 / DX).astype(int), axis=0)
        near = cells[(np.abs((cells + 0.5) * DX - dp) <= BOX + DX).all(1)]
        inside = (np.abs((near[:, None, :] + U[None]) * DX - dp) < BOX).all(-1).mean(1)
        vbox = inside.sum() * DX ** 3
        vsnap = (np.abs((cells + 0.5) * DX - dp) < BOX).all(1).sum() * DX ** 3
        rows.append({"qid": q["qid"], "s0c_rel": rel, "mesh_over_particles": vbox / vol[m].sum(),
                     "snapped_over_particles": vsnap / vol[m].sum()})
    a = np.array([[x["s0c_rel"], abs(x["mesh_over_particles"] - 1)] for x in rows])
    corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    md = ["# Phase 12 S0 run 1: push-normalisation diagnosis (Oct 24, 2026; mesh only)\n",
          "| query | S0c relative error | mesh-in-box / masked-particle volume | snapped / masked-particle |", "|---|---|---|---|"]
    md += [f"| {x['qid']} | {x['s0c_rel']:.4f} | {x['mesh_over_particles']:.4f} | {x['snapped_over_particles']:.4f} |" for x in rows]
    md += ["", f"Correlation between the S0c error and |ratio − 1|: {corr:.2f}"]
    run = start_run("p12_s0_diagnose", {"source": "results/p12_s0/s0.json"}, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"rows": rows, "corr": corr}, run.path("diagnose.json"))
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
