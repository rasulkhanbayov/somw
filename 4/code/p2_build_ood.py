"""Phase 2 (prep): build out-of-distribution elastic queries.

Each OOD query re-simulates a released elastic test object (its frame-0
particles and volumes) with physics outside PhysCtrl's training range, using
the same Warp MPM settings that generated the training data. Ground truth is
therefore produced by the very simulator the probes call.

Splits (configs/p2_ood.yaml):
  ood_E_low         Young's modulus 10^3.5 Pa   (training range 10^4 - 10^7)
  ood_E_high        Young's modulus 10^7.5 Pa
  ood_force_x3      drag force x3               (training coeff 0.02 - 0.2)
  ood_gravity_drop  no drag, gravity on, floor just under the object -> contact.
                    Elastic training data never has gravity, so this is the
                    split where contact events exist.
  ood2_E_low, ood2_E_high, ood2_force, ood2_drop  (Phase 4m, Oct 14, 2026) FRESH random shifts
                    seeded by qid: log10 E ~ U(3.0, 3.9) / U(7.1, 7.8); drag coeff ~ U(0.3, 0.8);
                    gravity drop with the floor U(0.1, 0.5) below the object. See PLAN.md.
  mo_comp_<j>       (Phase 4l, Oct 15, 2026) one object of a two-object contact scene, alone, with
                    the scene's shared (E, nu) and its own in-distribution drag (mo_scenes.py).
  aug_<j>           (Phase 14a, Oct 22, 2026) training candidates for simulator distillation, seeded by
                    (qid, split): log10 E ~ U(3.0, 8.0), nu ~ U(0.05, 0.45), drag coeff ~ U(0.02, 0.8) x base;
                    the drag point is a uniformly drawn particle with PhysCtrl's 0.4 drag box as the mask
                    (as generate_mpm_data.py), direction from the hemisphere around the outward radial
                    direction. No gravity (base has no gravity input).
  id_new_<j>        (Phase 4a calibration set, added Oct 11, 2026) fresh IN-distribution
                    physics for the same object: log10 E ~ U(4, 7), nu ~ U(0.05, 0.45), one
                    drag force at the stored drag point with coefficient ~ U(0.02, 0.2) x the
                    PhysCtrl base coefficient, as in PhysCtrl's generate_mpm_data.py. The
                    direction is drawn from the hemisphere around the outward radial
                    direction at the drag point; PhysCtrl uses the mesh normal, which the
                    test set does not store. Seeded by (qid, split), so it is reproducible.

Writes PhysCtrl-format HDF5 files (all 49 frames of x, v, F, C) plus
results/p2_build_ood/shard_<i>/ood_manifest.json. Unstable runs are recorded,
not written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, resume_rows, save_json, shard_items, start_run)
from backends import make_backend, testset_manifest_path, N_SIM_FRAMES  # noqa: E402
from regimes import _base_drag_coeff  # noqa: E402


def _hemisphere(normal: np.ndarray, rng) -> np.ndarray:
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    return d if d @ normal >= 0 else -d


# Phase 4l: {component qid: scene record}, set by main() from the FULL object list before sharding
# (pairing needs every object), or by a test.
MO_CTX: dict = {}


def _drag(o: dict, q: dict, rng, c_lo: float, c_hi: float) -> None:
    """One drag force at the stored drag point: coefficient ~ U(c_lo, c_hi) x PhysCtrl's base
    coefficient, direction from the hemisphere around the outward radial direction."""
    x0 = np.asarray(q["x0"], dtype=np.float64)
    point = np.asarray(q["drag_point"], dtype=np.float64).reshape(-1, 3)[0]
    mask = np.asarray(q["drag_mask"]).reshape(-1, len(x0))[0]
    radial = point - x0.mean(0)
    radial /= max(np.linalg.norm(radial), 1e-9)
    coeff = rng.uniform(c_lo, c_hi) * _base_drag_coeff(np.asarray(q["vol"]), mask)
    o["drag_force"] = (_hemisphere(radial, rng) * coeff)[None, :]
    o["drag_point"] = point[None, :]
    o["drag_mask"] = mask[None, :]


def make_ood(q: dict, split: str) -> dict:
    o = dict(q)
    o["split"] = split
    o["qid"] = f"{q['qid']}__{split}"
    if split == "ood_E_low":
        o["E"] = 10 ** 3.5
    elif split == "ood_E_high":
        o["E"] = 10 ** 7.5
    elif split == "ood_force_x3":
        o["drag_force"] = np.asarray(q["drag_force"]) * 3.0
    elif split == "ood_gravity_drop":
        n = len(q["x0"])
        o["drag_point"] = np.zeros((0, 3))
        o["drag_mask"] = np.zeros((0, n))
        o["drag_force"] = np.zeros((0, 3))
        o["gravity"] = 1
        o["floor_height"] = float(np.asarray(q["x0"])[:, 1].min() - 0.3)
    elif split.startswith("id_new_"):
        import zlib
        rng = np.random.default_rng(zlib.crc32(o["qid"].encode()))
        o["E"] = float(10 ** rng.uniform(4, 7))
        o["nu"] = float(rng.uniform(0.05, 0.45))
        _drag(o, q, rng, 0.02, 0.2)
    elif split.startswith("ood2_") or split.startswith("ood3_"):
        # Phase 4m (pre-registered Oct 14, 2026): FRESH random shifts, seeded by qid. Same draw
        # order as id_new_ (E, nu, then drag), with one parameter moved outside the training range.
        # Phase 9 (Oct 19, 2026): ood3_<family> = the same distributions, new draws (the split name seeds them).
        import zlib
        family = split[5:]
        rng = np.random.default_rng(zlib.crc32(o["qid"].encode()))
        lo_hi = {"E_low": (3.0, 3.9), "E_high": (7.1, 7.8)}.get(family, (4.0, 7.0))
        o["E"] = float(10 ** rng.uniform(*lo_hi))
        o["nu"] = float(rng.uniform(0.05, 0.45))
        if family in ("E_low", "E_high"):
            _drag(o, q, rng, 0.02, 0.2)
        elif family == "force":
            _drag(o, q, rng, 0.3, 0.8)
        elif family == "drop":
            n = len(q["x0"])
            o["drag_point"] = np.zeros((0, 3))
            o["drag_mask"] = np.zeros((0, n))
            o["drag_force"] = np.zeros((0, 3))
            o["gravity"] = 1
            o["floor_height"] = float(np.asarray(q["x0"])[:, 1].min() - rng.uniform(0.1, 0.5))
        else:
            raise ValueError(split)
    elif split.startswith("stiff_"):
        # Pilot E1 (Oct 18, 2026; exploratory): in-distribution drag and nu, Young's modulus 10^U(7.5, 9.0),
        # i.e. stiffer than PhysCtrl's training range (10^4-10^7). Same draw order as id_new_.
        import zlib
        rng = np.random.default_rng(zlib.crc32(o["qid"].encode()))
        o["E"] = float(10 ** rng.uniform(7.5, 9.0))
        o["nu"] = float(rng.uniform(0.05, 0.45))
        _drag(o, q, rng, 0.02, 0.2)
    elif split.startswith("aug_"):
        # Phase 14a (pre-registered Oct 22, 2026): candidates over a wider physics range, random drag point.
        import zlib
        rng = np.random.default_rng(zlib.crc32(o["qid"].encode()))
        o["E"] = float(10 ** rng.uniform(3.0, 8.0))
        o["nu"] = float(rng.uniform(0.05, 0.45))
        x0 = np.asarray(q["x0"], dtype=np.float64)
        point = x0[rng.integers(len(x0))]
        mask = (np.abs(x0 - point) < 0.4).all(axis=-1).astype(np.int64)
        radial = point - x0.mean(0)
        radial /= max(np.linalg.norm(radial), 1e-9)
        coeff = rng.uniform(0.02, 0.8) * _base_drag_coeff(np.asarray(q["vol"]), mask)
        o["drag_force"] = (_hemisphere(radial, rng) * coeff)[None, :]
        o["drag_point"] = point[None, :]
        o["drag_mask"] = mask[None, :]
    elif split.startswith("mo_comp_"):
        # Phase 4l (pre-registered Oct 15, 2026): one object of a two-object scene, simulated
        # alone with the scene's shared (E, nu) and its own in-distribution drag.
        import zlib
        from mo_scenes import scene_physics
        if o["qid"] not in MO_CTX:
            raise KeyError(f"{o['qid']}: no Phase 4l pairing context (set p2_build_ood.MO_CTX)")
        o["E"], o["nu"] = scene_physics(MO_CTX[o["qid"]]["scene"])
        _drag(o, q, np.random.default_rng(zlib.crc32(o["qid"].encode())), 0.02, 0.2)
    else:
        raise ValueError(split)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p2_build_ood"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)

    if args.dry_run:
        print("[p2_build_ood] dry run: the synthetic backend generates its OOD splits on the fly.")
        save_json([], run.path("ood_manifest.json"))
        run.finish("ok")
        return

    from physctrl_bridge import is_unstable, write_h5

    out_dir = Path(cfg["paths"]["ood_dir"])
    # base_manifest/base_split (Phase 4g): build from another object set than the test set,
    # e.g. PhysCtrl's sample/ objects for training data. Default: the released test set.
    base = cfg.get("base_manifest") or testset_manifest_path(False)
    entries = backend.manifest(base, split=cfg.get("base_split", "test_id"))[: cfg["n_base_queries"]]
    mo_splits = [sp for sp in cfg["splits"] if sp.startswith("mo_comp_")]
    if mo_splits:
        from mo_scenes import component_context
        MO_CTX.update(component_context([e["qid"] for e in entries], mo_splits))
    entries = shard_items(entries, shard, args.num_shards)
    # Failures are appended as they happen so a restart after a CUDA crash keeps them and
    # does not retry an object that already blew up.
    fail_log = run.path("failures.jsonl")
    failures = resume_rows(fail_log, args.resume)
    failed = {f["qid"] for f in failures}
    manifest = []
    for e in entries:
        q = backend.load(e)
        for split in cfg["splits"]:
            o = make_ood(q, split)
            path = out_dir / split / f"{o['qid']}.h5"
            if o["qid"] in failed:
                continue
            if path.exists():
                manifest.append({"qid": o["qid"], "path": str(path), "split": split, "mat_type": 0,
                                 "base_qid": q["qid"]})
                continue
            try:
                sim = backend.make_sim(o, timing_mode=cfg["mode"])
                xs, vs, Fs, Cs = [], [], [], []
                st = sim.export_state()
                xs.append(st["x"]); vs.append(st["v"]); Fs.append(st["F"]); Cs.append(st["C"])
                for _ in range(N_SIM_FRAMES):
                    sim.step_frames(1, record=False)
                    st = sim.export_state()
                    xs.append(st["x"]); vs.append(st["v"]); Fs.append(st["F"]); Cs.append(st["C"])
                x = np.stack(xs, 0)
                if is_unstable(x):
                    failures.append({"qid": o["qid"], "reason": "unstable"})
                    append_jsonl(failures[-1], fail_log)
                    continue
                write_h5(path, {
                    "x": x, "v": np.stack(vs, 0), "F": np.stack(Fs, 0), "C": np.stack(Cs, 0),
                    "vol": o["vol"], "floor_height": o["floor_height"],
                    "drag_point": o["drag_point"] if np.asarray(o["drag_force"]).size else [],
                    "drag_mask": o["drag_mask"] if np.asarray(o["drag_force"]).size else [],
                    "drag_force": o["drag_force"] if np.asarray(o["drag_force"]).size else [],
                    "E": o["E"], "nu": o["nu"], "mat_type": 0, "gravity": o["gravity"],
                })
                manifest.append({"qid": o["qid"], "path": str(path), "split": split, "mat_type": 0,
                                 "base_qid": q["qid"]})
            except Exception as ex:
                failures.append({"qid": o["qid"], "reason": f"{type(ex).__name__}: {ex}"})
                if cuda_context_lost(ex):
                    failures[-1]["reason"] = "crashed (CUDA context lost): " + failures[-1]["reason"]
                    append_jsonl(failures[-1], fail_log)
                    exit_cuda_lost(f"{o['qid']}", ex)
                append_jsonl(failures[-1], fail_log)
        print(f"[p2_build_ood] {q['qid']} done")

    save_json(manifest, run.path("ood_manifest.json"))
    save_json(failures, run.path("failures.json"))
    print(f"[p2_build_ood] {len(manifest)} written, {len(failures)} failed")
    run.finish("ok", n_written=len(manifest), n_failed=len(failures))


if __name__ == "__main__":
    main()
