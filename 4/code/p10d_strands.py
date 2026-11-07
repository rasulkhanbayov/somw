"""Phase 10d (pre-registered Oct 22, 2026; PLAN.md "Phase 10d"): are the reference's "strands" numerical?

GPU step (default): the 23 shifted Phase 11 items ($BIG_ROOT/renders_p11/*__coeff_0.4/query_v2.h5), re-simulated at
  R1 dt 5e-5 / grid 125, R2 dt 2.5e-5 / grid 125 (stored particles), and on a body-preserving filled body
  (3x3x3 particles of volume dx^3/27 in every occupied 125-grid cell: the same shell, the same mass) at
  R3 dt 1e-4 / grid 125 and R4 dt 5e-5 / grid 250. R0 is the stored reference (dt 1e-4, grid 125).
  Positions at simulator frame 23 (model frame 11) -> $BIG_ROOT/p10d/<case>__<R>.npz; rows -> results/p10d_strands/.
--summarize (CPU): stretch, left-behind share and Chamfer to R0 per item and setting; the pre-registered decision on
  the 6 strand items (picked by viewing, post hoc), the other 17 as controls -> results/p10d_summary/SUMMARY.md.
--check (CPU): builds the filled bodies and checks volume and total force against the stored reference; no simulation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, append_jsonl, cuda_context_lost, exit_cuda_lost, save_json  # noqa: E402

FRAME, DX, BOX = 23, 10.0 / 125, 0.4
STRANDS = ("p11_03_", "p11_04_", "p11_06_", "p11_12_", "p11_21_", "p11_24_")   # audit §1.3, picked by viewing
SETTINGS = {"R1": ("stored", {"substep_dt": 5e-5, "n_grid": 125}),
            "R2": ("stored", {"substep_dt": 2.5e-5, "n_grid": 125}),
            "R3": ("filled", {"substep_dt": 1e-4, "n_grid": 125}),
            "R4": ("filled", {"substep_dt": 5e-5, "n_grid": 250})}
OUT = Path(os.path.expandvars("${BIG_ROOT}/p10d"))


def cases():
    """The 23 valid shifted Phase 11 items: cases recorded invalid (unstable reference, Phase 11 amendment) are skipped."""
    out = []
    for p in sorted(glob.glob(os.path.expandvars("${BIG_ROOT}/renders_p11/*__coeff_0.4/query_v2.h5"))):
        cj = Path(p).parent / "case.json"
        if cj.exists() and json.loads(cj.read_text()).get("invalid"):
            continue
        out.append(p)
    return out


def filled_body(q):
    """The stored body (occupied 125-grid cells) filled with 27 particles per cell; same push in total."""
    from regimes import _base_drag_coeff
    x0, vol = np.asarray(q["x0"], np.float64), np.asarray(q["vol"], np.float64).reshape(-1)
    cells = np.unique(np.floor(x0 / DX).astype(np.int64), axis=0)
    off = (np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / 3.0
    xf = ((cells[:, None, :] + off[None]) * DX).reshape(-1, 3)
    vf = np.full(len(xf), DX ** 3 / 27.0)
    dp = np.asarray(q["drag_point"], np.float64).reshape(-1)[:3]
    m0 = np.asarray(q["drag_mask"]).reshape(-1).astype(bool)
    mf = (np.abs(xf - dp) < BOX).all(1)
    coeff_dir = np.asarray(q["drag_force"], np.float64).reshape(-1)[:3] / _base_drag_coeff(vol, m0)
    ff = coeff_dir * _base_drag_coeff(vf, mf)
    fq = dict(q, qid=q["qid"] + "_filled", x0=xf.astype(np.float32), vol=vf.astype(np.float32),
              drag_mask=mf[None].astype(np.int64), drag_force=ff[None].astype(np.float32),
              x=None, v=None, F=None, C=None)
    check = {"volume_stored": float(vol.sum()), "volume_filled": float(vf.sum()),
             "force_stored": (m0.sum() * np.asarray(q["drag_force"]).reshape(-1)[:3]).tolist(),
             "force_filled": (mf.sum() * ff).tolist(), "n_filled": int(len(xf)), "n_masked_filled": int(mf.sum())}
    return fq, check


def measures(x, x0, vol, L):
    w = vol / vol.sum()
    ext = lambda y: np.max(np.percentile(y, 99, axis=0) - np.percentile(y, 1, axis=0))   # noqa: E731
    disp = np.linalg.norm(x - x0, axis=1) / L
    return {"stretch": float(ext(x) / ext(x0)), "left_behind": float(w[disp < 0.2].sum())}


def chamfer(a, b, L):
    from scipy.spatial import cKDTree
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2 / L)


def run(args):
    from backends import make_backend, query_from_h5
    from common import load_config
    from physctrl_bridge import is_unstable
    cfg = load_config(args.config)
    backend = make_backend(cfg, False)
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(RESULTS_ROOT) / "p10d_strands"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in cases():
        case = Path(path).parent.name
        q = query_from_h5(path, "p11", case)
        for name, (body, ov) in SETTINGS.items():
            dst = OUT / f"{case}__{name}.npz"
            if dst.exists():
                continue
            qq, chk = (filled_body(q) if body == "filled" else (q, {}))
            try:
                traj = backend.make_sim(qq, overrides=ov).step_frames(FRAME)
            except Exception as ex:
                if cuda_context_lost(ex):
                    exit_cuda_lost(f"{case} {name}", ex)
                raise
            unstable = bool(is_unstable(traj))
            np.savez_compressed(dst, x=traj[-1].astype(np.float32), x0=np.asarray(qq["x0"], np.float32),
                                vol=np.asarray(qq["vol"], np.float32))
            append_jsonl({"case": case, "setting": name, "body": body, **ov, "unstable": unstable, **chk},
                         out_dir / "rows.jsonl")
            print(f"[p10d] {case} {name} done (unstable {unstable})", flush=True)


def summarize():
    from backends import query_from_h5
    rows, md = [], ["# Phase 10d: are the strands numerical? (pre-registered Oct 22, 2026)\n",
                    "Measures at model frame 11; stretch = longest extent / initial; left-behind = mass share moved "
                    "< 0.2 object sizes; Chamfer to R0 in object sizes. Strand items picked by viewing (post hoc).\n",
                    "| item | group | " + " | ".join(f"stretch {r}" for r in ("R0", "R1", "R2", "R3", "R4"))
                    + " | " + " | ".join(f"left {r}" for r in ("R0", "R2", "R4")) + " | Chamfer R4–R0 |",
                    "|---|---|" + "---|" * 9]
    for path in cases():
        case = Path(path).parent.name
        q = query_from_h5(path, "p11", case)
        x0, vol = np.asarray(q["x0"], np.float64), np.asarray(q["vol"], np.float64).reshape(-1)
        L = float(np.ptp(x0, axis=0).max())
        m = {"R0": measures(np.asarray(q["x"][FRAME], np.float64), x0, vol, L)}
        pos = {"R0": np.asarray(q["x"][FRAME], np.float64)}
        for name in SETTINGS:
            f = OUT / f"{case}__{name}.npz"
            if not f.exists():
                raise RuntimeError(f"missing {f}")
            z = np.load(f)
            m[name] = measures(z["x"].astype(np.float64), z["x0"].astype(np.float64), z["vol"].astype(np.float64), L)
            pos[name] = z["x"].astype(np.float64)
        grp = "strand" if case.startswith(STRANDS) else "control"
        ch = chamfer(pos["R4"], pos["R0"], L)
        rows.append({"case": case, "group": grp, "m": m, "chamfer_R4_R0": ch})
        md.append(f"| {case.split('__')[0]} | {grp} | " + " | ".join(f"{m[r]['stretch']:.2f}" for r in ("R0", "R1", "R2", "R3", "R4"))
                  + " | " + " | ".join(f"{m[r]['left_behind']:.2f}" for r in ("R0", "R2", "R4")) + f" | {ch:.3f} |")
    S = [r for r in rows if r["group"] == "strand"]
    med = lambda f: float(np.median([f(r) for r in S]))                                        # noqa: E731
    rel = lambda a, b: med(lambda r: abs(r["m"][a]["stretch"] / r["m"][b]["stretch"] - 1))     # noqa: E731
    dlb = lambda a, b: med(lambda r: abs(r["m"][a]["left_behind"] - r["m"][b]["left_behind"]))  # noqa: E731
    st4, lb4 = med(lambda r: r["m"]["R4"]["stretch"]), med(lambda r: r["m"]["R4"]["left_behind"])
    conv = rel("R4", "R0") <= 0.15 and dlb("R4", "R0") <= 0.10 and rel("R2", "R0") <= 0.15 and dlb("R2", "R0") <= 0.10
    num = st4 <= 1.5 or lb4 <= 0.05
    verdict = "CONVERGED" if conv else ("NUMERICAL" if num else "PARTLY NUMERICAL")
    md += ["", f"Strand items (median of 6): stretch R0 {med(lambda r: r['m']['R0']['stretch']):.2f}, R4 {st4:.2f}; "
               f"left-behind R0 {med(lambda r: r['m']['R0']['left_behind']):.2f}, R4 {lb4:.2f}; "
               f"|stretch ratio − 1| R4/R0 {rel('R4', 'R0'):.3f}, R2/R0 {rel('R2', 'R0'):.3f}; "
               f"|Δ left-behind| R4 {dlb('R4', 'R0'):.3f}, R2 {dlb('R2', 'R0'):.3f}.",
           f"Controls (median of {len(rows) - len(S)}): |stretch ratio − 1| R4/R0 "
           f"{float(np.median([abs(r['m']['R4']['stretch'] / r['m']['R0']['stretch'] - 1) for r in rows if r['group'] == 'control'])):.3f}.",
           "", f"**Verdict (pre-registered rule): {verdict}**"]
    out = Path(RESULTS_ROOT) / "p10d_summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"verdict": verdict, "rows": rows}, out / "summary.json")
    print("\n".join(md[-4:]))


def check():
    from backends import query_from_h5
    for path in cases():
        case = Path(path).parent.name
        q = query_from_h5(path, "p11", case)
        _, c = filled_body(q)
        dv = abs(c["volume_filled"] / c["volume_stored"] - 1)
        df = np.linalg.norm(np.subtract(c["force_filled"], c["force_stored"])) / np.linalg.norm(c["force_stored"])
        print(f"{case[:40]:40s} particles {c['n_filled']:6d} masked {c['n_masked_filled']:5d} | volume diff {dv:.1e} | "
              f"total force diff {df:.1e}")
        if dv > 5e-3 or df > 5e-3:     # float32 points on a cell boundary can switch cells (rubber boot: 0.2%)
            raise RuntimeError(f"{case}: filled body does not preserve volume or total force")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p10d_strands.yaml")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--resume", action="store_true", help="accepted for run_resumable; finished npz files are skipped")
    args = ap.parse_args()
    if args.check:
        check()
    elif args.summarize:
        summarize()
    else:
        run(args)


if __name__ == "__main__":
    main()
