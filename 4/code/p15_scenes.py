"""Phase 15a (CPU; pre-registered Oct 22, 2026, PLAN.md "Phase 15"): two-object scene design for the contact-aware
generator pilot. A pushed object A moves into an object B at rest.

Every rule here was fixed in PLAN before any 15a data existed (clarifications of Oct 24, also recorded there before data):
  components   A = an existing single-object in-distribution rollout at the reference setting: training scenes use the
               4g re-simulations of PhysCtrl's sample/ objects (data/fc_train/id_new_<j>, j = 0..14), development
               scenes the calibration rollouts of the test objects (data/calib/id_new_<j>, j = 0, 1). The scene's
               (E, nu) is A's draw (in distribution, as id_new_), so A's stored rollout is exactly A's motion before
               contact. B is another object of the same set, at rest, with no force (its x0 from its own draw-j file).
  pairing      set s, draw j: objects sorted, permuted by default_rng(crc32("p15<s>_<j>")); object perm[k] is A with
               B = perm[k + 1] (cyclic), so every object is A once per draw. 20% controls: default_rng(crc32(scene +
               "/kind")).random() < 0.2.
  placement    contact: k* ~ U{5..18}; u = A's centre-of-mass displacement from frame 0 to model frame k*; B is
               centred on A's frame-k* centre, then moved along u by the smallest s in [0, 6] with no approach
               < D_C (= 2 dx) before k*; translation rounded to whole cells. Up to 20 tries (new k*).
               control: random direction u, s grows by 0.2 (to 8) until A and B stay >= 0.5 apart in every frame.
  valid        the first approach k_c is a model frame in [3, 21] (contact) or never (control); the initial distance
               >= D_C; every point of A (all frames) and of B stays in [0.5, 9.5] (the Phase 10e wall rule).
  points       the model sees 1,024 farthest-point-sampled points per object (start index 0, from x0), fixed per object.
Positions in simulator units; model frame k = simulator frame 2k + 1.
Output: results/p15a_scenes/{scenes.jsonl (every scene, valid or not, with its reason), SUMMARY.md}
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append_jsonl, load_config, load_json, results_dir, start_run  # noqa: E402

DX = 0.08
D_C = 2 * DX
LO, HI = 0.5, 9.5
N_PER_OBJ = 1024
N_DESIGN_PTS = 256
CONTROL_SHARE = 0.2


def _seed(s: str) -> int:
    return zlib.crc32(s.encode())


def fps(x0: np.ndarray, k: int = N_PER_OBJ) -> np.ndarray:
    """Farthest-point sampling from index 0 (deterministic); returns k indices into x0."""
    x0 = np.asarray(x0, np.float64)
    idx = np.zeros(k, np.int64)
    d = np.linalg.norm(x0 - x0[0], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(x0 - x0[idx[i]], axis=1))
    return idx


def components(runs: list[str], splits: list[str]) -> dict:
    """{(object id, split): manifest entry} from the build runs."""
    out = {}
    for r in runs:
        for f in sorted(results_dir(r).glob("shard_*/ood_manifest.json")):
            for e in load_json(f):
                if e["split"] in splits:
                    out[(e["qid"].split("__")[0], e["split"])] = e
    return out


def pairing(objs: list[str], set_name: str, j: int) -> list[dict]:
    ids = sorted(objs)
    perm = np.random.default_rng(_seed(f"p15{set_name}_{j}")).permutation(len(ids))
    out = []
    for k in range(len(ids)):
        a, b = ids[perm[k]], ids[perm[(k + 1) % len(ids)]]
        sc = f"{a}__{b}__p15{set_name}_{j}"
        kind = "control" if np.random.default_rng(_seed(sc + "/kind")).random() < CONTROL_SHARE else "contact"
        out.append({"scene": sc, "a": a, "b": b, "set": set_name, "j": j, "split": f"p15{set_name}_{j}", "kind": kind})
    return out


def _min_dist(trees, b, shift):
    return np.array([t.query(b + shift, k=1)[0].min() for t in trees])


def _in_box(xa, xb0, t_b):
    pts = np.concatenate([xa.reshape(-1, 3), xb0 + t_b])
    return bool(pts.min() >= LO and pts.max() <= HI)


def design(rec: dict, xa: np.ndarray, xb0: np.ndarray) -> dict:
    """B's translation for one scene from A's stored rollout xa (49, N, 3) and B's rest points xb0 (N, 3)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(_seed(rec["scene"] + "/design"))
    ia = rng.choice(xa.shape[1], N_DESIGN_PTS, replace=False)
    ib = rng.choice(xb0.shape[0], N_DESIGN_PTS, replace=False)
    a_m, b0 = xa[1::2][:24][:, ia], xb0[ib]
    trees = [cKDTree(f) for f in a_m]
    tree0 = cKDTree(xa[0][ia])
    for tries in range(1, 21):
        k_star = None
        if rec["kind"] == "contact":
            k_star = int(rng.integers(5, 19))
            u = a_m[k_star].mean(0) - xa[0][ia].mean(0)
            if np.linalg.norm(u) < 1e-3:
                return {**rec, "t_b": None, "k_c": None, "k_star": k_star, "tries": tries, "valid": False,
                        "reason": "A does not move"}
            u = u / np.linalg.norm(u)
            base = a_m[k_star].mean(0) - b0.mean(0)

            def early(s):
                return bool((_min_dist(trees[:k_star], b0, base + s * u) < D_C).any())
            lo, hi = 0.0, 6.0
            if early(hi):
                continue
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if early(mid) else (lo, mid)
            t_b = np.round((base + hi * u) / DX) * DX
        else:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            t_b = None
            for s in np.arange(0.0, 8.0 + 1e-9, 0.2):
                cand = np.round(s * u / DX) * DX
                if _min_dist(trees, b0, cand).min() >= 0.5 and _in_box(xa, xb0, cand):
                    t_b = cand
                    break
            if t_b is None:
                continue
        d = _min_dist(trees, b0, t_b)
        d0 = tree0.query(b0 + t_b, k=1)[0].min()
        k_c = int(np.argmax(d < D_C)) if (d < D_C).any() else None
        if not _in_box(xa, xb0, t_b) or d0 < D_C:
            continue
        if rec["kind"] == "contact" and (k_c is None or not 3 <= k_c <= 21):
            continue
        if rec["kind"] == "control" and d.min() < 0.5:
            continue
        return {**rec, "t_b": t_b.tolist(), "k_c": k_c, "k_star": k_star, "tries": tries, "valid": True, "reason": ""}
    return {**rec, "t_b": None, "k_c": None, "k_star": None, "tries": 20, "valid": False, "reason": "no valid placement"}


def main():
    import h5py
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p15a_scenes.yaml")
    ap.add_argument("--limit", type=int, default=None, help="code test only: first N scenes per set, to a scratch run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["run_name"] if args.limit is None else cfg["run_name"] + "_codetest", cfg, synthetic=False)
    log = run.path("scenes.jsonl")
    if log.exists():
        log.unlink()
    md = ["# Phase 15a scene design (pre-registered Oct 22, 2026; clarified Oct 24 before data)\n",
          "| set | scenes | valid contact | valid control | invalid (reasons) |", "|---|---|---|---|---|"]
    for set_name, sc in cfg["sets"].items():
        comp = components(sc["comp_runs"], sc["splits"])
        recs = []
        for j in sc["draws"]:
            split = f"id_new_{j}"
            objs = sorted({o for o, s in comp if s == split})
            recs += [dict(r, a_path=comp[(r["a"], split)]["path"], b_path=comp[(r["b"], split)]["path"])
                     for r in pairing(objs, set_name, j)]
        recs = recs[:args.limit] if args.limit else recs
        rows = []
        for r in recs:
            with h5py.File(r["a_path"], "r") as h:
                xa = np.asarray(h["x"], np.float64)
            with h5py.File(r["b_path"], "r") as h:
                xb0 = np.asarray(h["x"][0], np.float64)
            d = design(r, xa, xb0)
            append_jsonl(d, log)
            rows.append(d)
        bad = {}
        for d in rows:
            if not d["valid"]:
                bad[d["reason"]] = bad.get(d["reason"], 0) + 1
        md.append(f"| {set_name} | {len(rows)} | {sum(d['valid'] and d['kind'] == 'contact' for d in rows)} | "
                  f"{sum(d['valid'] and d['kind'] == 'control' for d in rows)} | "
                  f"{sum(bad.values())} ({', '.join(f'{k}: {v}' for k, v in sorted(bad.items())) or '—'}) |")
        ks = [d["k_c"] for d in rows if d["valid"] and d["kind"] == "contact"]
        if ks:
            md.append(f"|  | first approach k_c: median {np.median(ks):.0f}, range {min(ks)}-{max(ks)} | | | |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
