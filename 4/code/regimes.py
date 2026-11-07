"""Phase 1 scene regimes, built from released test-set objects (no meshes needed).

physctrl     the test query unchanged: 2,048 particles, n_grid 125, substep 1e-4
dense10x     each particle split into 10 jittered children (20,480 particles)
dense10x_fine dense10x on a 250^3 grid
multiobj{2,3} 2 or 3 test objects, shrunk to 0.6x, stacked under gravity onto
             the floor so they collide with each other and the floor

Physics is kept sane when particles are split: particle volume is recomputed
the way PhysCtrl does, and the per-particle drag force is rescaled so the
force per unit mass is unchanged (PhysCtrl sets force via a base coefficient
proportional to mean masked particle volume).
"""
from __future__ import annotations

import numpy as np

from physctrl_bridge import particle_volume

GRID_LIM = 10.0
DEFAULT_N_GRID = 125
DENSITY = 1000.0


def _base_drag_coeff(vol: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    total, masked = vol.sum(), vol[m].sum()
    return 9.8 * DENSITY * vol[m].mean() / (masked / total)


def upsample(query: dict, factor: int, n_grid: int = DEFAULT_N_GRID, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x0 = np.asarray(query["x0"], dtype=np.float64)
    from lift import knn

    nn = np.linalg.norm(x0[knn(x0, 1)[:, 0]] - x0, axis=1)
    r = 0.3 * np.median(nn)
    children = [x0] + [x0 + rng.normal(scale=r, size=x0.shape) for _ in range(factor - 1)]
    x_new = np.concatenate(children, 0).astype(np.float32)
    vol_new = particle_volume(x_new, n_grid, GRID_LIM)
    q = dict(query)
    q.update(x0=x_new, vol=vol_new, x=None, v=None, F=None, C=None, qid=f"{query['qid']}_up{factor}")
    masks = np.asarray(query.get("drag_mask", []))
    forces = np.asarray(query.get("drag_force", []))
    if forces.size:
        masks = masks.reshape(len(forces.reshape(-1, 3)), -1)
        new_masks, new_forces = [], []
        for m, f in zip(masks, forces.reshape(-1, 3)):
            m_new = np.tile(m, factor)
            scale = _base_drag_coeff(vol_new, m_new) / _base_drag_coeff(np.asarray(query["vol"]), m)
            new_masks.append(m_new)
            new_forces.append(f * scale)
        q["drag_mask"] = np.stack(new_masks, 0)
        q["drag_force"] = np.stack(new_forces, 0)
    return q


def multi_object(queries: list[dict], n_grid: int = DEFAULT_N_GRID) -> dict:
    n = len(queries)
    if n == 2:
        centres = [(5.0, 3.0, 5.0), (5.1, 4.0, 5.0)]
    elif n == 3:
        centres = [(4.6, 3.0, 5.0), (5.4, 3.0, 5.0), (5.0, 4.0, 5.0)]
    else:
        raise ValueError("multi_object supports 2 or 3 objects")
    parts = []
    for q, c in zip(queries, centres):
        x = np.asarray(q["x0"], dtype=np.float64)
        x = (x - x.mean(0)) * 0.6 + np.asarray(c)
        parts.append(x)
    x_all = np.concatenate(parts, 0).astype(np.float32)
    floor = float(x_all[:, 1].min() - 0.05)
    return {
        "qid": "multi_" + "_".join(q["qid"] for q in queries),
        "split": f"multiobj{n}", "path": None,
        "x0": x_all, "x": None, "v": None, "F": None, "C": None,
        "vol": particle_volume(x_all, n_grid, GRID_LIM),
        "floor_height": floor,
        "drag_point": np.zeros((0, 3)), "drag_mask": np.zeros((0, len(x_all))), "drag_force": np.zeros((0, 3)),
        "E": float(queries[0]["E"]), "nu": float(queries[0]["nu"]),
        "gravity": 1, "mat_type": 0, "n_objects": n,
    }


def build_regime(name: str, base_queries: list[dict]) -> tuple[dict, dict]:
    """Returns (query, simulator overrides)."""
    q0 = base_queries[0]
    if name == "physctrl":
        return q0, {}
    if name == "dense10x":
        return upsample(q0, 10), {}
    if name == "dense10x_fine":
        return upsample(q0, 10, n_grid=250), {"n_grid": 250}
    if name == "multiobj2":
        return multi_object(base_queries[:2]), {}
    if name == "multiobj3":
        return multi_object(base_queries[:3]), {}
    raise ValueError(name)
