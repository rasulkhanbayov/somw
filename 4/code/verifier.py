"""Window verifier: restart the exact simulator from a state lifted out of a
trajectory, run a short window, and measure how far the trajectory drifts from it.

Windows always start and end on simulator frames the generator outputs
(odd frames 1..47), so a window of m model frames spans 2m simulator frames.
The anchor window instead starts from the TRUE initial state (frame 0) and runs
2m-1 frames to land on model frame m-1.

Defect (per window):
  raw  = mean_i || sim_end_i - traj_end_i ||                   (simulator units)
  sigma = mean_i std_p( sim_end_i under P perturbed restarts )  (chaos scale)
  norm = raw / max(sigma, sigma_floor)
Perturbations: position jitter (fraction of grid dx), velocity jitter (relative),
and Young's modulus jitter (log-normal). They estimate how sensitive the window
is, so sensitive contacts are not flagged as model error.

Cost is counted in simulator substeps and returned with every result.
"""
from __future__ import annotations

import numpy as np

from lift import knn, lift_state


def generated_frames(x0: np.ndarray, gen: np.ndarray) -> dict[int, np.ndarray]:
    """{sim frame: positions} for a generated trajectory gen (24, N, 3)."""
    frames = {0: np.asarray(x0, dtype=np.float32)}
    for k in range(gen.shape[0]):
        frames[2 * k + 1] = gen[k]
    return frames


def gt_frames(x: np.ndarray, odd_only: bool = True) -> dict[int, np.ndarray]:
    """Ground-truth frames restricted to what a generator would expose."""
    if odd_only:
        return {0: x[0], **{s: x[s] for s in range(1, x.shape[0], 2)}}
    return {s: x[s] for s in range(x.shape[0])}


def _perturb_state(state: dict, rng, pos_eps: float, vel_rel: float) -> dict:
    out = dict(state)
    out["x"] = state["x"] + rng.normal(scale=pos_eps, size=state["x"].shape).astype(np.float32)
    vscale = np.linalg.norm(state["v"], axis=1, keepdims=True).mean() + 1e-8
    out["v"] = state["v"] + rng.normal(scale=vel_rel * vscale, size=state["v"].shape).astype(np.float32)
    return out


def blew_up(x: np.ndarray, sim) -> bool:
    """Non-finite positions, or particles outside the simulation grid."""
    lim = getattr(sim, "grid_lim", None)
    return (not np.all(np.isfinite(x))) or (lim is not None and (x.min() < 0 or x.max() > lim))


def run_window(backend, query: dict, frames: dict[int, np.ndarray], start: int, length: int,
               lift_cfg: dict, n_perturb: int = 0, perturb_cfg: dict | None = None,
               sim_overrides: dict | None = None, state: dict | None = None,
               anchor: bool = False, seed: int = 0, nbrs=None, return_frames: bool = False) -> dict:
    """One window. start/length in simulator frames. If anchor, start must be 0 and the
    true initial state is used. If `state` is given it is used instead of lifting
    (oracle restarts)."""
    perturb_cfg = perturb_cfg or {}
    rng = np.random.default_rng(seed)
    end = start + length
    if end not in frames:
        raise KeyError(f"window end frame {end} not available")

    sim = backend.make_sim(query, overrides=sim_overrides)
    frame_dt = sim.frame_dt
    if anchor:
        if start != 0:
            raise ValueError("anchor windows start at frame 0")
        state = {"x": np.asarray(query["x0"], dtype=np.float32), "v": None, "F": None, "C": None}
    elif state is None:
        state = lift_state(frames, start, frame_dt, rest=np.asarray(query["x0"], dtype=np.float64),
                           k=lift_cfg.get("k", 12), lam=lift_cfg.get("lam", 1e-6),
                           use_F=lift_cfg.get("use_F", True), use_C=lift_cfg.get("use_C", True),
                           nbrs=nbrs, F_clamp=lift_cfg.get("F_clamp"))

    sim.set_state(state["x"], state.get("v"), state.get("F"), state.get("C"), time=start * frame_dt)
    traj = sim.step_frames(length)
    sim_end = traj[-1]
    target = frames[end]
    per_point = np.linalg.norm(sim_end - target, axis=1)
    substeps = sim.substeps_done
    blown = blew_up(sim_end, sim)
    blown_perturb = False

    sigma = float("nan")
    if n_perturb > 0:
        ends = []
        for p in range(n_perturb):
            q_p = dict(query)
            q_p["E"] = float(query["E"] * np.exp(rng.normal(scale=perturb_cfg.get("logE_sd", 0.05))))
            sp = backend.make_sim(q_p, overrides=sim_overrides)
            dx = getattr(sp, "grid_lim", 10.0) / getattr(sp, "n_grid", 125)
            st = state if anchor and state.get("v") is None else _perturb_state(
                {k: (v if v is not None else _default(k, len(state["x"]))) for k, v in state.items()},
                rng, perturb_cfg.get("pos_eps_dx", 0.05) * dx, perturb_cfg.get("vel_rel", 0.05))
            sp.set_state(st["x"], st.get("v"), st.get("F"), st.get("C"), time=start * frame_dt)
            ends.append(sp.step_frames(length)[-1])
            blown_perturb = blown_perturb or blew_up(ends[-1], sp)
            substeps += sp.substeps_done
        ends = np.stack(ends, 0)
        sigma = float(np.linalg.norm(ends - ends.mean(0, keepdims=True), axis=2).mean())

    raw = float(per_point.mean())
    floor = perturb_cfg.get("sigma_floor", 1e-4)
    return {
        "start": start, "length": length, "anchor": anchor,
        "raw": raw, "raw_p90": float(np.quantile(per_point, 0.9)),
        "sigma": sigma,
        "norm": raw / max(sigma, floor) if np.isfinite(sigma) else float("nan"),
        "substeps": int(substeps),
        "finite": bool(np.all(np.isfinite(sim_end))),
        # Pre-registered (Oct 10, 2026): p2_summarize scores a blown main rollout as
        # raw = norm = +inf, and a blown perturbed rollout as norm = +inf.
        "blown": bool(blown), "blown_perturb": bool(blown_perturb),
        **({"frames": {start + 1 + i: traj[i] for i in range(len(traj))}} if return_frames else {}),
    }


def _default(key: str, n: int):
    if key == "v":
        return np.zeros((n, 3), dtype=np.float32)
    if key == "F":
        return np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (n, 1))
    if key == "C":
        return np.zeros((n, 9), dtype=np.float32)
    raise KeyError(key)


# --------------------------------------------------------------------------- #
# Window placement
# --------------------------------------------------------------------------- #
def verified_model_frames(start: int, length: int) -> list[int]:
    """Model frames whose positions a window (start, length in SIM frames) produces by
    simulation: the odd simulator frames after the restart, start+1 .. start+length.
    The restart frame itself is an input, not a verified output."""
    return [(f - 1) // 2 for f in range(start + 1, start + length + 1) if f % 2 == 1 and (f - 1) // 2 < 24]


def candidate_starts(m: int, n_model_frames: int = 24) -> list[int]:
    """Model-frame indices k such that a window k -> k+m fits."""
    return list(range(0, n_model_frames - m))


GRAVITY = 9.8   # |g| in simulator units (physctrl_bridge: material_params["g"])
DENSITY = 1000.0
GRID_LIM = 10.0


def cfl_number(E: float, nu: float, substep_dt: float, n_grid: int) -> float:
    """Explicit-MPM stability number: dilatational wave speed x dt / grid spacing.
    Above about 1, waves cross more than a cell per substep and the rollout blows up.
    Phase 2c (Oct 11, 2026): every window that lost the CUDA context at dt 8e-4 / grid 64
    had CFL 1.18-1.76; every query with no blow-up had CFL <= 1.03."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return substep_dt * np.sqrt((lam + 2 * mu) / DENSITY) / (GRID_LIM / n_grid)


def resolve_fidelity(spec: dict, query: dict) -> dict:
    """Simulator overrides for one query. A spec may carry cfl_target: then the substep is
    min(substep_dt, cfl_target x dx / wave speed), the standard explicit-MPM choice."""
    spec = dict(spec or {})
    target = spec.pop("cfl_target", None)
    if target is not None:
        n_grid = int(spec.get("n_grid", 125))
        dt = float(spec.get("substep_dt", 1e-4))
        c1 = cfl_number(float(query["E"]), float(query["nu"]), 1.0, n_grid)   # CFL per unit dt
        spec["substep_dt"] = float(min(dt, target / c1))
    return spec


def prior_contact_frame(query: dict, frame_dt: float) -> int | None:
    """Model frame at which the object first reaches the floor under free fall from rest,
    predicted from the query's inputs alone (gravity flag, floor height, initial points),
    never from the generated trajectory. A generator that ignores gravity never falls, so
    events read off its output miss exactly the contacts it gets wrong (Phase 2 diagnostics:
    base event windows covered 0% of gravity-drop contacts)."""
    if int(np.asarray(query.get("gravity", 0))) != 1:
        return None
    h = float(np.asarray(query["x0"])[:, 1].min() - float(np.asarray(query["floor_height"])))
    if h <= 0:
        return 0
    sim_frame = np.sqrt(2.0 * h / GRAVITY) / frame_dt
    return int(np.clip(round((sim_frame - 1) / 2), 0, 23))


def select_windows(strategy: str, gen_samples: np.ndarray, query: dict, n_windows: int, m: int,
                   rng, gt: np.ndarray | None = None,
                   frame_dt: float = 0.0417) -> list[tuple[int, int, bool]]:
    """Return windows as (start_sim_frame, length_sim_frames, is_anchor).

    gen_samples: (S, 24, N, 3); sample 0 is the trajectory being verified.
    strategy: uniform | random | event | event_union | disagree | proposal | oracle
      event_union = event plus the physics-prior contact frame (prior_contact_frame).
      proposal = anchor + (n_windows-2) event windows + 1 random window (the
      proposal's placement rule). oracle needs gt (24,N,3) and is analysis-only.
    """
    g = gen_samples[0]
    cands = candidate_starts(m, g.shape[0])
    as_window = lambda k: (2 * k + 1, 2 * m, False)
    anchor = (0, 2 * m - 1, True)

    def top(score, n, exclude=()):
        order = [c for c in np.argsort(-np.asarray(score)) if cands[c] not in exclude]
        return [cands[c] for c in order[:n]]

    if strategy == "uniform":
        ks = np.linspace(0, len(cands) - 1, n_windows).round().astype(int)
        return [as_window(cands[i]) for i in ks]
    if strategy == "random":
        return [as_window(k) for k in rng.choice(cands, size=n_windows, replace=False)]
    if strategy in ("event", "proposal", "event_union"):
        acc = np.zeros(len(g))
        acc[1:-1] = np.linalg.norm(g[2:] - 2 * g[1:-1] + g[:-2], axis=2).max(axis=1)
        floor_gap = (g[..., 1] - float(query["floor_height"])).min(axis=1)
        contact = (floor_gap < 0.1).astype(float)
        frame_score = acc / (acc.max() + 1e-12) + contact
        if strategy == "event_union":
            # Pre-registered Oct 11, 2026 (Phase 2c): add the physics-prior contact frame,
            # weighted above any generation-derived score so a predicted contact is always probed.
            kc = prior_contact_frame(query, frame_dt)
            if kc is not None:
                frame_score[max(kc - 1, 0):kc + 2] += 3.0
        score = [frame_score[k:k + m + 1].max() for k in cands]
        if strategy in ("event", "event_union"):
            return [as_window(k) for k in top(score, n_windows)]
        n_event = max(n_windows - 2, 0)
        ev = top(score, n_event)
        rest = [k for k in cands if k not in ev]
        rnd = [int(rng.choice(rest))] if rest and n_windows >= 2 else []
        return [anchor] + [as_window(k) for k in ev + rnd]
    if strategy == "disagree":
        if gen_samples.shape[0] < 2:
            raise ValueError("disagree placement needs >= 2 samples")
        spread = np.linalg.norm(gen_samples - gen_samples.mean(0, keepdims=True), axis=3).mean(axis=(0, 2))
        score = [spread[k:k + m + 1].max() for k in cands]
        return [as_window(k) for k in top(score, n_windows)]
    if strategy == "oracle":
        if gt is None:
            raise ValueError("oracle placement needs ground truth")
        err = np.linalg.norm(g - gt, axis=2).mean(axis=1)
        growth = [err[k + m] - err[k] for k in cands]
        return [as_window(k) for k in top(growth, n_windows)]
    raise ValueError(strategy)


def aggregate(windows: list[dict], key: str, how: str = "max") -> float:
    vals = np.array([w[key] for w in windows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(vals.max() if how == "max" else vals.mean())


def nbrs_for(query: dict, k: int):
    """kNN on the rest shape. For two-object scenes (Phase 4l) neighbours are found within each
    object only, so lifted F and C never mix particles across the gap between objects."""
    x0 = np.asarray(query["x0"], dtype=np.float64)
    oid = query.get("object_id")
    if oid is None:
        return knn(x0, k)
    oid = np.asarray(oid)
    out = np.empty((len(x0), k), dtype=np.int64)
    for o in np.unique(oid):
        idx = np.nonzero(oid == o)[0]
        out[idx] = idx[knn(x0[idx], k)]
    return out
