"""State lifting: recover a simulator state (x, v, F, C) from point positions.

A generated trajectory gives positions only, at simulator frames
{0 (the true input), 1, 3, ..., 47}. Restarting the MPM solver needs, per particle:
  x  position                          -> read directly
  v  velocity                          -> quadratic (3-point) finite difference in time
  F  deformation gradient (rest -> now) -> least-squares fit over k rest-space neighbours
  C  APIC affine velocity (~ grad v)    -> least-squares fit of neighbour velocity differences

All fits are exact for affine motion, and the velocity rule is exact for motion
quadratic in time; selftest.py checks both. Errors on real trajectories are what
the Phase 2a "lift floor" experiment measures.
"""
from __future__ import annotations

import numpy as np


def knn(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of each point, excluding itself. (N, k)"""
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(X).query(X, k=k + 1)
        return idx[:, 1:]
    except ImportError:
        out = np.empty((len(X), k), dtype=np.int64)
        for s in range(0, len(X), 1024):
            d = np.linalg.norm(X[s:s + 1024, None] - X[None], axis=-1)
            out[s:s + 1024] = np.argsort(d, axis=1)[:, 1:k + 1]
        return out


def ls_gradient(A: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    """Per-point G minimising sum_j ||b_j - G a_j||^2 + lam ||G||^2.

    A, B: (N, k, 3) neighbour differences. Returns G: (N, 3, 3) with b ~ G a.
    """
    AtA = np.einsum("nki,nkj->nij", A, A) + lam * np.eye(3)[None]
    AtB = np.einsum("nki,nkj->nij", A, B)
    M = np.linalg.solve(AtA, AtB)          # b_row ~ a_row @ M
    return np.transpose(M, (0, 2, 1))


def project_F(F: np.ndarray, s_lo: float, s_hi: float) -> np.ndarray:
    """Closest proper deformation with singular values in [s_lo, s_hi]. (N, 3, 3)

    Least-squares F from neighbour offsets can come out inverted or near-singular on a few
    particles, which blew up every F-lifted restart in Phase 2 v1. Phase 4i (Oct 13, 2026):
    take the proper rotation of the SVD (flip the smallest singular direction if det < 0) and
    clamp the stretches."""
    U, S, Vt = np.linalg.svd(F)
    flip = np.linalg.det(U) * np.linalg.det(Vt) < 0
    U[flip, :, -1] *= -1
    S[flip, -1] *= -1
    S = np.clip(S, s_lo, s_hi)
    return U @ (S[:, :, None] * Vt)


def quad_velocity(t: np.ndarray, x: np.ndarray, at: int) -> np.ndarray:
    """Velocity at sample `at` from 3 samples (t: (3,), x: (3, N, 3)) via the
    derivative of the interpolating quadratic. Exact for motion quadratic in t."""
    t0, t1, t2 = t
    tq = t[at]
    w0 = ((tq - t1) + (tq - t2)) / ((t0 - t1) * (t0 - t2))
    w1 = ((tq - t0) + (tq - t2)) / ((t1 - t0) * (t1 - t2))
    w2 = ((tq - t0) + (tq - t1)) / ((t2 - t0) * (t2 - t1))
    return w0 * x[0] + w1 * x[1] + w2 * x[2]


def lift_state(frames: dict[int, np.ndarray], sim_frame: int, frame_dt: float,
               rest: np.ndarray, k: int = 12, lam: float = 1e-6,
               use_F: bool = True, use_C: bool = True, nbrs: np.ndarray | None = None,
               F_clamp: tuple[float, float] | None = None) -> dict:
    """Lift (x, v, F, C) at `sim_frame` from available position frames.

    frames: {simulator frame index: (N, 3) positions}. Must contain sim_frame.
    rest:   (N, 3) rest configuration (the true frame-0 positions).
    """
    avail = sorted(frames)
    if sim_frame not in frames:
        raise KeyError(f"sim frame {sim_frame} not in available frames {avail[:6]}...")
    i = avail.index(sim_frame)
    if 0 < i < len(avail) - 1:
        sel, at = [avail[i - 1], sim_frame, avail[i + 1]], 1
    elif i == 0:
        sel, at = avail[:3], 0
    else:
        sel, at = avail[-3:], 2
    t = np.array(sel, dtype=np.float64) * frame_dt
    X = np.stack([frames[s] for s in sel], 0).astype(np.float64)
    x = frames[sim_frame].astype(np.float64)
    v = quad_velocity(t, X, at)

    n = len(x)
    if nbrs is None:
        nbrs = knn(rest, k)
    if use_F:
        A = rest[nbrs] - rest[:, None]              # rest-space neighbour offsets
        B = x[nbrs] - x[:, None]                    # current offsets
        F = ls_gradient(A, B, lam)
        if F_clamp is not None:
            F = project_F(F, *F_clamp)
    else:
        F = np.tile(np.eye(3), (n, 1, 1))
    if use_C:
        A = x[nbrs] - x[:, None]
        B = v[nbrs] - v[:, None]
        C = ls_gradient(A, B, lam)
    else:
        C = np.zeros((n, 3, 3))
    return {"x": x.astype(np.float32), "v": v.astype(np.float32),
            "F": F.reshape(n, 9).astype(np.float32), "C": C.reshape(n, 9).astype(np.float32)}
