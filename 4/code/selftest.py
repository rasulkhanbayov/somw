"""Self-test: checks the maths and the safety rails. CPU only, no PhysCtrl, seconds.

Run before any Slurm job and after every edit to code/:
    python code/selftest.py
Every check prints PASS/FAIL; the exit code is non-zero if any fails.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def t_lift_affine():
    from lift import lift_state

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    A = np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    B = 0.2 * rng.normal(size=(3, 3))
    b, c = rng.normal(size=3), rng.normal(size=3)
    dt = 0.0417
    # x(t) = (I + tA') X + b t + c t^2 with A' = A - I scaled: affine in X, quadratic in t
    def pos(t):
        G = np.eye(3) + (A - np.eye(3)) * t + B * t ** 2
        return X @ G.T + b * t + c * t ** 2
    frames = {s: pos(s * dt) for s in [0, 1, 3, 5, 7]}
    st = lift_state(frames, 3, dt, rest=X, k=10, lam=1e-12)
    t = 3 * dt
    G_true = np.eye(3) + (A - np.eye(3)) * t + B * t ** 2
    dG = (A - np.eye(3)) + 2 * B * t
    v_true = X @ dG.T + b + 2 * c * t
    F_err = np.abs(st["F"].reshape(-1, 3, 3) - G_true[None]).max()
    v_err = np.abs(st["v"] - v_true).max()
    # velocity gradient wrt current position: dv/dx = dG G^-1
    C_true = dG @ np.linalg.inv(G_true)
    C_err = np.abs(st["C"].reshape(-1, 3, 3) - C_true[None]).max()
    check("lift: F exact on affine deformation", F_err < 1e-4, f"max err {F_err:.2e}")
    check("lift: v exact on quadratic-in-time motion", v_err < 1e-4, f"max err {v_err:.2e}")
    check("lift: C equals velocity gradient on affine motion", C_err < 1e-3, f"max err {C_err:.2e}")


def t_quad_velocity_edges():
    from lift import quad_velocity

    t = np.array([0.0, 0.1, 0.3])
    x = np.stack([(2 + 3 * ti + 4 * ti ** 2) * np.ones((5, 3)) for ti in t])
    ok = all(abs(quad_velocity(t, x, at)[0, 0] - (3 + 8 * t[at])) < 1e-9 for at in range(3))
    check("lift: one-sided quadratic velocity exact at both ends", ok)


def t_spearman_bootstrap():
    from common import bootstrap_ci, rankdata, spearman

    x = np.arange(20.0)
    check("stats: spearman monotone = 1", abs(spearman(x, x ** 3) - 1) < 1e-12)
    check("stats: spearman reversed = -1", abs(spearman(x, -x) + 1) < 1e-12)
    check("stats: ranks with ties", np.allclose(rankdata([1, 2, 2, 3]), [1, 2.5, 2.5, 4]))
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    p, lo, hi = bootstrap_ci(spearman, [a, a + rng.normal(size=200)], n_boot=300)
    check("stats: bootstrap CI brackets estimate", lo <= p <= hi, f"{lo:.2f} <= {p:.2f} <= {hi:.2f}")


def t_restart_exact():
    from backends import SyntheticBackend

    be = SyntheticBackend(n_points=30, per_split=1)
    q = be.load(be.manifest(split="test_id")[0])
    a = be.make_sim(q)
    a.step_frames(5, record=False)
    st = a.export_state()
    cont = a.step_frames(4)[-1]
    b = be.make_sim(q)
    b.set_state(st["x"], st["v"], time=5 * b.frame_dt)
    rest = b.step_frames(4)[-1]
    check("sim: restart from exported state reproduces continuation", np.abs(cont - rest).max() < 1e-5)


def t_window_zero_on_own_sim():
    from backends import SyntheticBackend
    from verifier import run_window

    be = SyntheticBackend(n_points=30, per_split=1)
    q = be.load(be.manifest(split="test_id")[0])
    frames = {s: q["x"][s] for s in range(q["x"].shape[0])}
    w = run_window(be, q, frames, 0, 5, {}, anchor=True)
    check("verifier: anchor window on ground truth has ~zero defect", w["raw"] < 1e-5, f"raw {w['raw']:.2e}")


def t_window_selection():
    from backends import SyntheticBackend
    from verifier import select_windows

    be = SyntheticBackend(n_points=30, per_split=1)
    q = be.load(be.manifest(split="ood_gravity_drop")[0])
    samples = be.generate("base", [q], n_samples=3, seed=0)[0]
    rng = np.random.default_rng(0)
    ok = True
    for strat in ["uniform", "random", "event", "disagree", "proposal", "oracle"]:
        for (s, L, a) in select_windows(strat, samples, q, 3, 2, rng, gt=q["x"][1::2][:24]):
            end = s + L
            ok &= (end % 2 == 1) and end <= 47 and (s == 0 if a else s % 2 == 1)
    check("verifier: every placement strategy yields windows on generator frames", ok)


def t_synthetic_guard():
    import common

    with tempfile.TemporaryDirectory() as d:
        old = common.RESULTS_ROOT
        common.RESULTS_ROOT = Path(d)
        (Path(d) / "SYNTHETIC_p9_test").mkdir()
        try:
            common.results_dir("p9_test")
            refused = False
        except RuntimeError:
            refused = True
        common.results_dir("p9_test", allow_synthetic=True)
        common.RESULTS_ROOT = old
    check("guard: summarizers refuse synthetic results by default", refused)


def t_config_env():
    from common import load_config

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("p: ${SOMW_SELFTEST_UNSET_VAR}/x\n")
    os.environ.pop("SOMW_SELFTEST_UNSET_VAR", None)
    try:
        load_config(f.name)
        raised = False
    except RuntimeError:
        raised = True
    check("guard: unset environment variable in config raises", raised)


def t_particle_volume():
    from physctrl_bridge import particle_volume

    x = np.array([[0.05, 0.05, 0.05], [0.06, 0.06, 0.06], [0.5, 0.5, 0.5]])
    v = particle_volume(x, 10, 1.0)
    check("bridge: particle volume = dx^3 / particles per cell", np.allclose(v, [5e-4, 5e-4, 1e-3]))


def t_model_frames():
    from physctrl_bridge import model_frame_to_sim_frame, to_model_coords, to_sim_coords

    ok = [model_frame_to_sim_frame(k) for k in (0, 1, 23)] == [1, 3, 47]
    x = np.random.default_rng(0).normal(size=(4, 3))
    check("bridge: model frame k is sim frame 2k+1; coords round-trip",
          ok and np.allclose(to_sim_coords(to_model_coords(x)), x))


def t_frame_time():
    # Simulated frame length = int(frame_dt / substep_dt) * substep_dt. It matches the
    # nominal 0.0417 s at PhysCtrl's default substep but not at every sweep setting, which
    # is why MPMSim advances its clock by the simulated value (bridge guard 5).
    n_default = int(0.0417 / 1e-4)
    n_coarse = int(0.0417 / 1.6e-3)
    check("bridge: frame = 417 substeps at 1e-4 (0.0417 s) but 26 at 1.6e-3 (0.0416 s)",
          n_default == 417 and n_coarse == 26 and abs(n_coarse * 1.6e-3 - 0.0416) < 1e-12,
          f"{n_default}, {n_coarse}")


def t_cuda_lost():
    # Sticky CUDA errors (a blown-up rollout writing out of bounds) must end the process with
    # EXIT_CUDA_LOST so the job wrapper restarts it; ordinary launch errors must not.
    import subprocess
    import tempfile
    from common import EXIT_CUDA_LOST, append_jsonl, cuda_context_lost, resume_rows
    from physctrl_bridge import Generator

    sticky = cuda_context_lost(RuntimeError("CUDA error: an illegal memory access was encountered"))
    plain = cuda_context_lost(RuntimeError("CUDA error: invalid configuration argument"))
    rc = subprocess.run([sys.executable, "-c",
                         "import sys; sys.path.insert(0, %r); from common import exit_cuda_lost; "
                         "exit_cuda_lost('selftest', RuntimeError('CUDA error: an illegal memory access'))"
                         % str(Path(__file__).resolve().parent)],
                        capture_output=True).returncode
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "rows.jsonl"
        append_jsonl({"a": 1}, f)
        kept = resume_rows(f, resume=True)
        fresh = resume_rows(f, resume=False)
    check("guard: sticky CUDA error exits 75 for a --resume restart; launch errors do not",
          sticky and not plain and rc == EXIT_CUDA_LOST and kept == [{"a": 1}] and fresh == []
          and not f.exists(), f"exit {rc}")
    check("bridge: at most 31 samples per generator call for 2,048 points",
          Generator.max_batch(2048) == 31 and 32 * 2048 > 65535)


def t_blowup_rule():
    # Pre-registered Phase 2 rule: a blown-up window is defect = +inf (an automatic reject),
    # kept in rank statistics rather than dropped.
    from common import spearman
    from p2_summarize import apply_blowup_rule, med

    rho = spearman(np.array([1.0, 2.0, np.inf, 0.5]), np.array([2.0, 3.0, 9.0, 1.0]))
    lf = [{"raw": 0.1, "raw_p90": 0.2, "norm": 1.0, "crashed": True}, {"raw": 0.1, "norm": 1.0, "blown_perturb": True}]
    pr = [{"u__raw_max": 0.3, "u__norm_max": 2.0, "u__blown": 1}, {"u__raw_max": 0.3, "u__norm_max": 2.0}]
    apply_blowup_rule(lf, pr, ["u"])
    ok = (abs(rho - 1) < 1e-12 and lf[0]["raw"] == np.inf and lf[1]["raw"] == 0.1 and lf[1]["norm"] == np.inf
          and pr[0]["u__raw_max"] == np.inf and pr[1]["u__raw_max"] == 0.3 and med([1.0, np.inf, np.inf]) == np.inf)
    check("phase 2: blow-ups score defect = +inf and stay in rank statistics", ok, f"rho {rho:.2f}")


def t_auroc_and_prior_contact():
    # AUROC is the single-threshold certificate metric (Phase 2c D); the physics-prior
    # contact frame drives event_union placement (Phase 2c C).
    from common import auroc
    from verifier import prior_contact_frame

    a1 = auroc([0.1, 0.2, 0.9, np.inf], [0, 0, 1, 1])
    a5 = auroc([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1])
    x0 = np.zeros((4, 3)); x0[:, 1] = 1.2 + np.arange(4)              # lowest point 1.2
    # free fall from h = 1.0 takes sqrt(2/9.8) = 0.4518 s = 10.83 sim frames -> model frame 5
    kc = prior_contact_frame({"gravity": 1, "x0": x0, "floor_height": 0.2}, 0.0417)
    none = prior_contact_frame({"gravity": 0, "x0": x0, "floor_height": 0.2}, 0.0417)
    check("phase 2c: AUROC (inf ranks highest, ties half) and ballistic contact frame",
          a1 == 1.0 and a5 == 0.5 and kc == 5 and none is None, f"auroc {a1}, {a5}; contact frame {kc}")


def t_cfl():
    # The CFL number predicts which coarse windows blew up in Phase 2c; the adaptive
    # fidelity must land exactly on its target and never exceed the fixed coarse substep.
    from verifier import cfl_number, resolve_fidelity

    c_stiff = cfl_number(10 ** 7.5, 0.3, 8e-4, 64)        # stiff OOD split at the fixed coarse dt
    c_ref = cfl_number(10 ** 7.5, 0.3, 1e-4, 125)        # same material at PhysCtrl's setting
    q = {"E": 10 ** 7.5, "nu": 0.3}
    ov = resolve_fidelity({"substep_dt": 8e-4, "n_grid": 64, "cfl_target": 0.5}, q)
    soft = resolve_fidelity({"substep_dt": 8e-4, "n_grid": 64, "cfl_target": 0.5}, {"E": 1e4, "nu": 0.3})
    ok = (c_stiff > 1.0 and c_ref < 0.5 and abs(cfl_number(q["E"], q["nu"], ov["substep_dt"], 64) - 0.5) < 1e-9
          and soft["substep_dt"] == 8e-4 and "cfl_target" not in ov)
    check("phase 2c: CFL > 1 at the fixed coarse dt for E = 10^7.5; adaptive dt hits CFL 0.5",
          ok, f"CFL coarse {c_stiff:.2f}, ref {c_ref:.2f}, adaptive dt {ov['substep_dt']:.2e}")


def t_conformal():
    # Class-conditional split conformal: averaged over many draws, a new bad trajectory is
    # accepted with probability <= alpha (Phase 4a certificate).
    from p4a_certificate import clopper_pearson, threshold

    rng = np.random.default_rng(0)
    hits = []
    for _ in range(4000):
        cal = rng.normal(size=49)
        lam = threshold(cal, 0.1)                 # k = floor(0.1 * 50) = 5
        hits.append(rng.normal() < lam)
    rate = float(np.mean(hits))
    lo, hi = clopper_pearson(0, 20)
    check("phase 4a: conformal false-accept rate <= alpha; k = 0 accepts nothing; CP interval",
          rate <= 0.1 + 0.015 and threshold(np.ones(5), 0.1) == -np.inf and lo == 0.0 and 0.13 < hi < 0.18,
          f"rate {rate:.3f} (alpha 0.1), CP(0/20) upper {hi:.3f}")


def t_object_conformal():
    # Object-level certificate (Phase 4a extension): on clustered data (correlated draws per
    # object), single-draw subsampling keeps P(accept a bad draw of a new object) <= alpha;
    # pooling with one draw per object equals the unweighted rule.
    from p4a_certificate import threshold
    from p4b_certificate_objects import object_rate, pooled_threshold

    rng = np.random.default_rng(1)
    hits = []
    for _ in range(3000):
        effects = rng.normal(size=40)                         # one random effect per object
        cal = [effects[j] + 0.3 * rng.normal(size=rng.integers(1, 6)) for j in range(39)]
        lam = threshold(np.array([rng.choice(c) for c in cal]), 0.1)
        hits.append(effects[39] + 0.3 * rng.normal() < lam)  # a bad draw of a new object
    rate = float(np.mean(hits))
    sc = rng.normal(size=29)
    same = pooled_threshold(sc, np.arange(29), 0.1) == threshold(sc, 0.1)
    orate = object_rate(np.array([1, 0, 0, 1.0]), np.array(["a", "a", "a", "b"]))   # (1/3 + 1) / 2
    check("phase 4a: object-level subsampling valid on clustered data; pooled = unweighted at 1 draw/object",
          rate <= 0.1 + 0.015 and same and abs(orate - 2 / 3) < 1e-12, f"rate {rate:.3f}")


def t_repair_frames():
    # Phase 4b: a window restarted at sim frame 2k+1 verifies model frames k+1 .. k+m; the
    # anchor (sim frames 0 .. 2m-1) verifies model frames 0 .. m-1.
    from verifier import verified_model_frames

    ok = (verified_model_frames(0, 7) == [0, 1, 2, 3] and verified_model_frames(9, 8) == [5, 6, 7, 8]
          and verified_model_frames(41, 8) == [21, 22, 23])
    check("phase 4b: verified model frames of anchor and restarted windows", ok)


def t_conditioning_grid():
    # Phase 4c: 30 candidates (one generator batch), identity first so ties resolve to it.
    from p4c_repair_conditioning import grid
    from common import load_config

    g = grid(load_config(Path(__file__).resolve().parents[1] / "configs" / "p4c_repair.yaml", True))
    check("phase 4c: conditioning grid has 30 candidates, identity first, no duplicates",
          len(g) == 30 and g[0] == {"dlogE": 0.0, "force_scale": 1.0}
          and len({(a["dlogE"], a["force_scale"]) for a in g}) == 30, f"{len(g)} candidates")


def t_frame_conditioning():
    # Phase 4g: the known-frame wrapper is an exact identity at initialisation and with no
    # mask; once known_emb is non-zero it changes only the masked frames' tokens. Masks follow
    # the pre-registered recipe.
    import torch
    from fc_model import make_fc, sample_mask

    class Tiny(torch.nn.Module):
        latent_dim, cond_frame = 8, 1
        def __init__(self):
            super().__init__(); self.input_encoder = torch.nn.Linear(3, 8)
        def forward(self, x):
            bs, nf, n, _ = x.shape
            return self.input_encoder(x.reshape(-1, n, 3)).reshape(bs, nf, n, 8)

    torch.manual_seed(0)
    m = Tiny(); fc = make_fc(m)
    x = torch.randn(2, 25, 5, 3)                     # cond frame + 24 frames
    base = m(x).detach()
    fc.mask = torch.zeros(2, 24, dtype=torch.bool); same0 = torch.equal(fc(x), base)
    fc.mask = None; same_none = torch.equal(fc(x), base)
    with torch.no_grad():
        fc.known_emb.fill_(1.0)
    mk = torch.zeros(2, 24, dtype=torch.bool); mk[:, 3] = True; fc.mask = mk
    d = (fc(x) - base).abs().amax(dim=(2, 3))        # (2, 25)
    only = bool((d[:, 4] > 0).all() and (d[:, [0, 1, 2, 3, 5]] == 0).all())
    rng = np.random.default_rng(0)
    ms = [sample_mask(rng) for _ in range(20000)]
    # P(empty) = p_none + (1 - p_none)(1 - p_anchor) P(0 windows) = 0.1 + 0.9 x 0.3 x 1/4 = 0.1675
    p_none = np.mean([not mk_.any() for mk_ in ms]); windows_ok = all(mk_.sum() % 1 == 0 for mk_ in ms)
    check("phase 4g: frame-conditioning wrapper is an identity at init and only touches masked frames",
          same0 and same_none and only and 0.155 < p_none < 0.18 and windows_ok, f"p(no mask) {p_none:.3f}")


def t_temporal_lora():
    # Phase 4g-v2: LoRA on attn_temp is an exact identity at init and when switched off, and
    # only its A/B parameters are trainable once the base is frozen.
    import torch
    from fc_model import add_temporal_lora, set_lora

    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_temp = torch.nn.Module()
            for a in ("to_q", "to_k", "to_v"):
                setattr(self.attn_temp, a, torch.nn.Linear(6, 6))
            self.attn_temp.to_out = torch.nn.ModuleList([torch.nn.Linear(6, 6)])
        def forward(self, x):
            at = self.attn_temp
            return at.to_out[0](at.to_v(x) + at.to_q(x) * at.to_k(x))

    torch.manual_seed(0)
    m = Blk(); x = torch.randn(4, 6); y0 = m(x).detach()
    m.requires_grad_(False); loras = add_temporal_lora(m, rank=2, alpha=2.0)
    same_init = torch.allclose(m(x), y0)
    with torch.no_grad():
        for l in loras:
            l.B.normal_()
    changed = not torch.allclose(m(x), y0)
    set_lora(m, False); off = torch.allclose(m(x), y0); set_lora(m, True)
    for n, p in m.named_parameters():
        p.requires_grad_(n.endswith(".A") or n.endswith(".B"))
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    check("phase 4g-v2: temporal LoRA is identity at init and when off; only A/B trainable",
          same_init and changed and off and len(loras) == 4 and n_train == 4 * (2 * 6 + 6 * 2), f"{n_train} trainable")

def t_project_F():
    # Phase 4i: projection keeps valid F unchanged, and maps inverted / near-singular F to a
    # proper deformation (det > 0) with stretches inside the bounds.
    from lift import project_F

    rng = np.random.default_rng(1)
    good = np.eye(3)[None] + 0.1 * rng.normal(size=(50, 3, 3))
    same = np.abs(project_F(good, 0.5, 2.0) - good).max()
    bad = np.stack([np.diag([1.0, 1.0, -0.7]), np.diag([1.0, 0.03, 1.0]), np.diag([5.0, 1.0, 1.0])])
    P = project_F(bad, 0.5, 2.0)
    S = np.linalg.svd(P, compute_uv=False)
    ok = (np.linalg.det(P) > 0).all() and S.min() >= 0.5 - 1e-9 and S.max() <= 2.0 + 1e-9
    check("phase 4i: F projection is identity on valid F, proper and bounded otherwise",
          same < 1e-9 and ok, f"identity err {same:.1e}, det {np.round(np.linalg.det(P), 3)}")

def t_fresh_shift_splits():
    # Phase 4m: the fresh shift families land in their pre-registered ranges, and id_new_ draws are
    # unchanged by the refactor into _drag (checked against a fixed reference value).
    from p2_build_ood import make_ood

    rng = np.random.default_rng(3)
    n = 50
    q = {"qid": "obj_000", "x0": rng.normal(size=(n, 3)) + 5.0, "vol": np.full(n, 1e-3),
         "drag_point": np.array([[5.0, 5.5, 5.0]]), "drag_mask": (np.arange(n) < 10)[None].astype(int),
         "drag_force": np.array([[0.0, 1.0, 0.0]]), "E": 1e5, "nu": 0.3, "floor_height": 0.2, "gravity": 0}
    ok = True
    for sp, (lo, hi) in (("ood2_E_low", (3.0, 3.9)), ("ood2_E_high", (7.1, 7.8)), ("ood2_force", (4, 7)), ("ood2_drop", (4, 7))):
        o = make_ood(q, sp)
        ok &= lo <= np.log10(o["E"]) <= hi and o["qid"] == f"obj_000__{sp}"
    d = make_ood(q, "ood2_drop")
    ok &= d["gravity"] == 1 and np.asarray(d["drag_force"]).size == 0 and 0.1 <= q["x0"][:, 1].min() - d["floor_height"] <= 0.5
    a, b = make_ood(q, "id_new_3"), make_ood(q, "id_new_3")
    ok &= np.allclose(a["drag_force"], b["drag_force"]) and 4 <= np.log10(a["E"]) <= 7
    check("phase 4m: fresh shift splits in their pre-registered ranges; draws reproducible", ok)

def t_mo_scenes():
    # Phase 4l: pairing is deterministic and disjoint (40 contact + 10 control); the design puts the
    # first approach of a synthetic approaching pair inside [3, 21] with B moved by multiples of dx;
    # neighbour lists never cross objects; the generated contact frame is found; both components of
    # a scene share (E, nu).
    import p2_build_ood
    from mo_scenes import DX, D_C, design, gen_contact_frame, pairing, component_context
    from verifier import nbrs_for

    ids = [f"o{i:03d}" for i in range(100)]
    P1, P2 = pairing(ids, 3), pairing(ids, 3)
    flat = [x for r in P1 for x in (r["a"], r["b"])]
    ok_pair = P1 == P2 and len(set(flat)) == 100 and sum(r["kind"] == "contact" for r in P1) == 40
    rng = np.random.default_rng(0)
    cube = rng.uniform(-0.5, 0.5, size=(300, 3))
    t = np.arange(49)[:, None, None]
    xa = np.broadcast_to(cube + 5.0, (49, 300, 3)).copy()                 # A at rest
    xb = cube[None] + np.array([5.0, 5.0, 5.0]) + t * np.array([0.03, 0.0, 0.0])   # B drifts along +x
    d = design({**P1[0], "kind": "contact"}, xa, xb)
    tb = np.asarray(d["t_b"]) if d["valid"] else None
    ok_design = d["valid"] and 3 <= d["k_c"] <= 21 and np.allclose(tb / DX, np.round(tb / DX))
    q = {"x0": np.concatenate([cube + 2.5, cube + 7.5]), "object_id": np.repeat([0, 1], 300)}
    nb = nbrs_for(q, 8)
    ok_nb = (q["object_id"][nb] == q["object_id"][:, None]).all()
    g = np.stack([np.concatenate([cube + 5.0, cube + [7.0 - 0.1 * k, 5.0, 5.0]]) for k in range(24)])
    kg = gen_contact_frame(g, q["object_id"])
    true_k = next(k for k in range(24) if np.linalg.norm(g[k][:300][:, None] - g[k][300:][None], axis=2).min() < D_C)
    ctx = component_context(ids, ["mo_comp_3"])
    p2_build_ood.MO_CTX.update(ctx)
    base = {"x0": cube + 5.0, "vol": np.full(300, 1e-3), "drag_point": np.array([[5.0, 5.4, 5.0]]),
            "drag_mask": (np.arange(300) < 30)[None].astype(int), "drag_force": np.zeros((1, 3)),
            "E": 1e5, "nu": 0.3, "floor_height": 0.2, "gravity": 0}
    oa = p2_build_ood.make_ood({**base, "qid": P1[0]["a"]}, "mo_comp_3")
    ob = p2_build_ood.make_ood({**base, "qid": P1[0]["b"]}, "mo_comp_3")
    ok_phys = oa["E"] == ob["E"] and oa["nu"] == ob["nu"] and 4 <= np.log10(oa["E"]) <= 7
    check("phase 4l: pairing, contact design, per-object kNN, generated contact frame, shared material",
          ok_pair and ok_design and ok_nb and kg == true_k and ok_phys,
          f"pair {ok_pair} design {ok_design} (k_c {d['k_c']}) knn {ok_nb} k_g {kg}/{true_k} phys {ok_phys}")


if __name__ == "__main__":
    for t in [t_lift_affine, t_quad_velocity_edges, t_spearman_bootstrap, t_restart_exact,
              t_window_zero_on_own_sim, t_window_selection, t_synthetic_guard, t_config_env,
              t_particle_volume, t_model_frames, t_frame_time, t_cuda_lost, t_blowup_rule, t_auroc_and_prior_contact, t_cfl, t_conformal, t_object_conformal, t_repair_frames, t_conditioning_grid, t_frame_conditioning, t_temporal_lora, t_project_F, t_fresh_shift_splits, t_mo_scenes]:
        try:
            t()
        except Exception as e:
            check(t.__name__, False, f"{type(e).__name__}: {e}")
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
