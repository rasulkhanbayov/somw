"""Phase 4l: two-object contact scenes built from single-object test queries.

Pre-registered Oct 15, 2026 (PLAN.md "Phase 4l"); every rule here was fixed before any data.
  pairing        draw j permutes the 100 test objects with default_rng(crc32("mo_<j>")) and pairs
                 consecutive ones; pairs 0-39 are contact scenes, 40-49 controls.
  physics        one shared (E, nu) per scene (in-distribution) from crc32(scene id); each object
                 keeps its own in-distribution drag, drawn as for id_new_ from crc32(component qid).
  placement      A stays; B is translated by a multiple of dx = 0.08 per axis (isolated dynamics
                 unchanged). Contact: target frame k* ~ U{5..18}, direction u; smallest s in [0, 6]
                 with no approach < D_C before k*; valid if the first approach k_c is in [3, 21],
                 the initial distance >= D_C and all points stay in [0.5, 9.5]. Control: s grows by
                 0.2 until the minimum distance over all frames >= 0.5 (s <= 8).
Positions in simulator units; model frame k = simulator frame 2k+1.
"""
from __future__ import annotations

import zlib

import numpy as np

DX = 0.08                 # reference grid spacing (10 / 125)
D_C = 2 * DX              # "approach" distance
N_DRAWS = 8
N_CONTACT, N_PAIRS = 40, 50
LO, HI = 0.5, 9.5         # all points of both objects stay inside this box
N_DESIGN_PTS = 256


def _seed(s: str) -> int:
    return zlib.crc32(s.encode())


def scene_id(a: str, b: str, j: int) -> str:
    return f"{a}__{b}__mo_{j}"


def pairing(obj_ids: list[str], j: int) -> list[dict]:
    """50 pairs for draw j: [{'scene', 'a', 'b', 'pair', 'kind', 'j'}], deterministic."""
    ids = sorted(obj_ids)
    perm = np.random.default_rng(_seed(f"mo_{j}")).permutation(len(ids))
    out = []
    for p in range(len(ids) // 2):
        a, b = ids[perm[2 * p]], ids[perm[2 * p + 1]]
        out.append({"scene": scene_id(a, b, j), "a": a, "b": b, "pair": p, "j": j,
                    "kind": "contact" if p < N_CONTACT else "control"})
    return out


def component_context(obj_ids: list[str], splits: list[str]) -> dict:
    """{component qid: scene record} for the mo_comp_<j> splits being built."""
    ctx = {}
    for sp in splits:
        j = int(sp.rsplit("_", 1)[1])
        for r in pairing(obj_ids, j):
            for role in ("a", "b"):
                ctx[f"{r[role]}__{sp}"] = {**r, "role": role}
    return ctx


def scene_physics(scene: str) -> tuple[float, float]:
    rng = np.random.default_rng(_seed(scene))
    return float(10 ** rng.uniform(4, 7)), float(rng.uniform(0.05, 0.45))


def _min_dist(trees, b_pts, shift):
    """Per-frame minimum distance between A (KD-trees) and B + shift."""
    return np.array([t.query(b + shift, k=1)[0].min() for t, b in zip(trees, b_pts)])


def _in_box(a_all, b_all, shift):
    pts = np.concatenate([a_all.reshape(-1, 3), (b_all + shift).reshape(-1, 3)])
    return bool(pts.min() >= LO and pts.max() <= HI)


def design(rec: dict, xa: np.ndarray, xb: np.ndarray) -> dict:
    """Translation of B for one scene, from the isolated ground truths xa, xb (49, N, 3).
    Returns rec + {'t_b', 'k_c', 'k_star', 'tries', 'valid', 'reason'}."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(_seed(rec["scene"] + "/design"))
    ia = rng.choice(xa.shape[1], N_DESIGN_PTS, replace=False)
    ib = rng.choice(xb.shape[1], N_DESIGN_PTS, replace=False)
    a_m, b_m = xa[1::2][:24][:, ia], xb[1::2][:24][:, ib]      # model frames, subsampled
    a0, b0 = xa[0][ia], xb[0][ib]
    trees = [cKDTree(f) for f in a_m]
    tree0 = cKDTree(a0)
    k_star = int(rng.integers(5, 19)) if rec["kind"] == "contact" else None
    for tries in range(1, 21):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        if rec["kind"] == "contact":
            base = a_m[k_star].mean(0) - b_m[k_star].mean(0)

            def early_contact(s):
                d = _min_dist(trees[:k_star], b_m[:k_star], base + s * u)
                return bool((d < D_C).any())
            lo, hi = 0.0, 6.0
            if early_contact(hi):
                continue
            for _ in range(30):                                  # smallest s with no early contact
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if early_contact(mid) else (lo, mid)
            t_b = np.round((base + hi * u) / DX) * DX
        else:
            t_b = None
            for s in np.arange(0.0, 8.0 + 1e-9, 0.2):
                cand = np.round((s * u) / DX) * DX
                if _min_dist(trees, b_m, cand).min() >= 0.5 and _in_box(xa, xb, cand):
                    t_b = cand
                    break
            if t_b is None:
                continue
        d = _min_dist(trees, b_m, t_b)
        d0 = tree0.query(b0 + t_b, k=1)[0].min()
        k_c = int(np.argmax(d < D_C)) if (d < D_C).any() else None
        if not _in_box(xa, xb, t_b) or d0 < D_C:
            continue
        if rec["kind"] == "contact" and (k_c is None or not 3 <= k_c <= 21):
            continue
        if rec["kind"] == "control" and d.min() < 0.5:
            continue
        return {**rec, "t_b": t_b.tolist(), "k_c": k_c, "k_star": k_star, "tries": tries, "valid": True, "reason": ""}
    return {**rec, "t_b": None, "k_c": None, "k_star": k_star, "tries": 20, "valid": False, "reason": "no valid placement"}


def compose(a: np.ndarray, b: np.ndarray, t_b) -> np.ndarray:
    """Concatenate object A and object B translated by t_b along the point axis (..., N, 3)."""
    return np.concatenate([a, b + np.asarray(t_b, dtype=a.dtype)], axis=-2)


def gen_contact_frame(g: np.ndarray, object_id: np.ndarray, d_c: float = D_C) -> int | None:
    """First model frame at which the two objects of ONE trajectory g (24, N, 3) come within d_c
    (available at test time from the generation itself)."""
    from scipy.spatial import cKDTree

    m = np.asarray(object_id) == 0
    for k in range(g.shape[0]):
        if cKDTree(g[k][m]).query(g[k][~m], k=1)[0].min() < d_c:
            return k
    return None
