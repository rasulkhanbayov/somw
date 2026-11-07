"""Phase 12 gate S0 (pre-registered Oct 22, 2026; PLAN.md "Phase 12" and its amendment). CPU only (`somw-fem` env),
development queries only (Phase 4 calibration pool data/calib/id_new_*; never a test set).

Steps, in order (all positions saved under $BIG_ROOT/p12_traj/s0/):
  dt rule  first 10 valid queries of id_new_0 (sorted file names) at dt = frame/4, /8, /16 (and /32 only if needed);
           the chosen n_div is the smallest of {4, 8, 16} whose result changes by < 0.005 (median over queries of the
           mean point distance over all 24 frames) when dt is halved; none -> S0 FAIL.
  S0a      first 5 valid queries of id_new_2, gravity only (no push, no floor), frames 1..11: mass-weighted centre of
           mass within 1e-3 of 1/2 g t^2 at every frame.
  S0b      first 20 valid queries of id_new_1 with log10 E >= 6, at the chosen dt: median over queries of the mean
           MPM-S2 point distance over model frames 1-8 <= 0.02.
  S0c      on the S0b runs: mass-weighted centre-of-mass displacement at simulator frame 1 equals 1/2 (n_masked f / M) t^2
           within 2% (relative norm) on every query.
Amendment Oct 24 (PLAN "Phase 12"): run 1 (job 432388) failed S0c (max 0.064) because the box force was normalised by
the masked particles' volume, not the mesh volume inside the box (p12_fem docstring). v2 = the snapped-box force; the dt
rule and S0b/S0c are re-run under new tags (S0a, gravity only, is unchanged and reused). S0c is measured on the mesh's
own centre of mass (where a FEM body's mass is); the particle-weighted one is reported as secondary. One rerun only:
if v2 fails, the pre-registered S0-FAIL outcome stands. Output: results/p12_s0_v2/ (run 1 stays in results/p12_s0/).
Also recorded: runtime per query, exclusions by the mesh rule.
Output: results/p12_s0_v2/{SUMMARY.md, s0.json}
  --smoke DIR: code test only (1 query per step, 3 frames, dt rule on /4 and /8); everything goes to DIR, never results/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, save_json  # noqa: E402
from p12_fem import FRAME, load_query, mesh_from_points, run_query  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data")) / "calib"   # cluster.env sets DATA_ROOT
TRAJ = Path(os.path.expandvars("${BIG_ROOT}/p12_traj/s0"))
WORK = Path(os.path.expandvars("${BIG_ROOT}/tmp/p12_s0"))


SMOKE = None
V = "v2_"            # tag prefix of the amended drag runs (run 1's cached positions are never reused for them)
OUT = "p12_s0_v2"


def valid_queries(pool, n, cond=lambda q: True):
    n = 1 if SMOKE else n
    out = []
    for f in sorted(glob.glob(str(DATA / pool / "*.h5"))):
        q = load_query(f)
        if not cond(q):
            continue
        if mesh_from_points(q["x0"], q["vol"])[3]["valid"]:
            out.append(f)
        if len(out) == n:
            break
    return out


def job(args):
    path, n_div, tag, threads, gravity_only, tend = args
    q = load_query(path)
    dst = TRAJ / tag / f"{q['qid']}.npz"
    if dst.exists():
        z = np.load(dst)
        return q["qid"], z["s2"], json.loads(str(z["info"]))
    pos, info = run_query(q, n_div, WORK / f"{tag}_{q['qid']}", threads, gravity_only=gravity_only, tend_frames=tend)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, s2=pos.astype(np.float32), info=json.dumps(info))
    return q["qid"], pos, info


def run_set(paths, n_div, tag, workers, threads, gravity_only=False, tend=47):
    if SMOKE:
        tend = 3
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(job, [(p, n_div, tag, threads, gravity_only, tend) for p in paths]))


def com(x, vol):
    w = vol / vol.sum()
    return (x * w[:, None]).sum(-2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", default=None)
    args = ap.parse_args()
    global SMOKE, TRAJ, WORK
    if args.smoke:
        SMOKE, TRAJ, WORK = Path(args.smoke), Path(args.smoke) / "traj", Path(args.smoke) / "work"
    md, res = ["# Phase 12 gate S0, v2 (development queries only; pre-registered Oct 22, 2026; snapped-box amendment Oct 24)\n"], {}

    # dt rule
    dt_paths = valid_queries("id_new_0", 10)
    runs = {n: {qid: p for qid, p, _ in run_set(dt_paths, n, f"{V}dt{n}", args.workers, args.threads)}
            for n in ((4, 8) if SMOKE else (4, 8, 16))}
    def change(a, b):
        return float(np.median([np.linalg.norm(runs[a][k] - runs[b][k], axis=2).mean() for k in runs[a]]))
    n_div = None
    for a, b in (((4, 8),) if SMOKE else ((4, 8), (8, 16))):
        res[f"change_{a}_{b}"] = change(a, b)
        if res[f"change_{a}_{b}"] < 0.005:
            n_div = a
            break
    if n_div is None:
        runs[32] = {qid: p for qid, p, _ in run_set(dt_paths, 32, f"{V}dt32", args.workers, args.threads)}
        res["change_16_32"] = change(16, 32)
        n_div = 16 if res["change_16_32"] < 0.005 else None
    res["n_div"] = n_div
    md += [f"- dt rule (10 queries): " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if k.startswith("change"))
           + f" -> n_div = {n_div}"]
    if n_div is None:
        md += ["- **S0 FAIL: no time step in {frame/4, /8, /16} converges to < 0.005.**"]
        out = SMOKE or Path(RESULTS_ROOT) / OUT
        out.mkdir(parents=True, exist_ok=True)
        (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
        save_json(res, out / "s0.json")
        return

    # S0a free fall
    a_paths = valid_queries("id_new_2", 5)
    fall = run_set(a_paths, n_div, "s0a", args.workers, args.threads, gravity_only=True, tend=11)
    dev = []
    for (qid, pos, _), p in zip(fall, a_paths):
        q = load_query(p)
        t = np.array([fr * FRAME for fr in range(1, 12, 2)])[:len(pos)]
        dy = com(pos, q["vol"].reshape(-1))[:, 1] - com(q["x0"], q["vol"].reshape(-1))[1]
        dev.append(float(np.abs(dy + 0.5 * 9.81 * t ** 2).max()))
    res["s0a_max_dev"] = dev
    res["S0a"] = bool(max(dev) <= 1e-3)

    # S0b small strain and S0c push
    b_paths = valid_queries("id_new_1", 20, cond=lambda q: np.log10(float(q["E"])) >= 6)
    small = run_set(b_paths, n_div, f"{V}s0b", args.workers, args.threads)
    dists, rel, rel_p, ratio, secs = [], [], [], [], []
    for (qid, pos, info), p in zip(small, b_paths):
        q = load_query(p)
        dists.append(float(np.linalg.norm(pos[:8] - q["gt"][:len(pos[:8])], axis=2).mean()))
        vol = q["vol"].reshape(-1)
        m = q["drag_mask"].reshape(-1).astype(bool)
        acc = m.sum() * q["drag_force"].reshape(-1)[:3] / (1000.0 * vol.sum())
        want = 0.5 * acc * FRAME ** 2
        got = np.asarray(info["mesh_com"][0]) - np.asarray(info["mesh_com0"])
        rel.append(float(np.linalg.norm(got - want) / np.linalg.norm(want)))
        got_p = com(pos[0], vol) - com(q["x0"], vol)
        rel_p.append(float(np.linalg.norm(got_p - want) / np.linalg.norm(want)))
        ratio.append(info["box_volume_ratio"])
        secs.append(info.get("seconds", np.nan))
    res.update(s0b_dists=dists, s0b_median=float(np.median(dists)), s0c_rel=rel, s0c_rel_particles=rel_p,
               box_volume_ratio=ratio, seconds=secs,
               S0b=bool(np.median(dists) <= 0.02), S0c=bool(max(rel) <= 0.02))
    res["S0"] = bool(res["S0a"] and res["S0b"] and res["S0c"])
    md += [f"- S0a free fall (5 queries): max deviation {max(dev):.2e} (limit 1e-3): **{'PASS' if res['S0a'] else 'FAIL'}**",
           f"- S0b small strain (20 queries, log E >= 6): median MPM-S2 distance over frames 1-8 {res['s0b_median']:.4f} "
           f"(limit 0.02; per query max {max(dists):.4f}): **{'PASS' if res['S0b'] else 'FAIL'}**",
           f"- S0c push (same 20): max relative error of the centre-of-mass displacement at frame 1 {max(rel):.4f} "
           f"(limit 0.02): **{'PASS' if res['S0c'] else 'FAIL'}** (mesh centre of mass; secondary, particle-weighted: "
           f"max {max(rel_p):.4f}; snapped-box / masked-particle volume {min(ratio):.3f}-{max(ratio):.3f})",
           f"- Runtime per full query at n_div {n_div}: median {np.nanmedian(secs) / 60:.1f} min ({args.threads} threads)",
           f"- **S0: {'PASS' if res['S0'] else 'FAIL'}**"]
    out = SMOKE or Path(RESULTS_ROOT) / OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json(res, out / "s0.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
