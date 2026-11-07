"""Phase 1a: wall-clock cost of generation vs simulation, on THIS node's GPU.

Run once on an A100 node and once on an H200 node; ratios are only compared
within the same hardware.

Generation: PhysCtrl base (4 DDIM steps) and large (25 steps), batch sizes from
the config, after warm-up, device-synchronised. Model loading is excluded and
reported separately.

Simulation: for each regime (regimes.py) and each timing mode
  as_is   PhysCtrl's solver unchanged (per-substep device syncs included)
  nosync  syncs removed
  graph   nosync + one frame of substeps replayed as a CUDA graph
we report solver build time, first-frame time (includes Warp JIT if the kernel
cache is cold), and steady-state full-rollout time (48 frames). The best
successful mode is the simulator's cost; giving the simulator its best case is
what makes a "simulation is expensive" conclusion credible.

Probe cost: build + restart + run for windows of m model frames (2m simulator
frames), in the best mode.

Falsifier (evaluated in p1_summarize.py, not here): best-mode simulation is
less than `go_ratio` x the per-sample generation time in every realistic regime.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, resume_rows, start_run, timed)
from backends import make_backend, testset_manifest_path, N_SIM_FRAMES  # noqa: E402


def row_key(r: dict) -> tuple:
    if r["kind"] == "generation":
        return ("generation", r["model"], r.get("qid"), r["batch_size"])
    if r["kind"] == "simulation":
        return ("simulation", r["regime"], r.get("qi"), r["mode"])
    return ("probe", r.get("qid"), r["m_model_frames"])


def fatal_guard(where: str, fn, *a, **kw):
    """Run fn; on a sticky CUDA error exit for a --resume restart instead of continuing."""
    try:
        return fn(*a, **kw)
    except Exception as ex:
        if cuda_context_lost(ex):
            exit_cuda_lost(where, ex)
        raise


def time_generation(backend, query: dict, which: str, batch_sizes: list[int], reps: int,
                    warmup: int, out_path: Path, tag: str, done: set) -> None:
    if all(("generation", which, query["qid"], bs) in done for bs in batch_sizes):
        return
    t0 = time.perf_counter()
    if not backend.synthetic:
        backend.generator(which)
    load_s = time.perf_counter() - t0
    for bs in batch_sizes:
        if ("generation", which, query["qid"], bs) in done:
            continue
        for _ in range(warmup):
            backend.generate(which, [query], n_samples=bs, seed=0, max_rows=bs)
        times = []
        for r in range(reps):
            rec = {}
            with timed(rec, "t"):
                backend.generate(which, [query], n_samples=bs, seed=r, max_rows=bs)
            times.append(rec["t"])
        append_jsonl({"kind": "generation", "hw": tag, "model": which, "qid": query["qid"], "batch_size": bs,
                      "model_load_s": load_s, "times_s": times,
                      "per_sample_s_median": float(np.median(times) / bs),
                      "num_inference_steps": None if backend.synthetic else
                      backend.generator(which).num_inference_steps,
                      "n_params": None if backend.synthetic else backend.generator(which).n_params},
                     out_path)


def time_simulation(backend, query: dict, overrides: dict, regime: str, qi: int, modes: list[str],
                    out_path: Path, tag: str, n_frames: int, done: set) -> dict:
    from physctrl_bridge import is_unstable

    best = None
    for mode in modes:
        row = {"kind": "simulation", "hw": tag, "regime": regime, "qi": qi, "mode": mode,
               "n_particles": int(len(query["x0"])), "overrides": overrides}
        if row_key(row) in done:
            continue
        try:
            rec = {}
            with timed(rec, "build_s"):
                sim = backend.make_sim(query, overrides=overrides, timing_mode=mode)
            with timed(rec, "first_frame_s"):
                sim.step_frames(1, record=False)
            # fresh scene so the timed rollout starts from frame 0
            sim = backend.make_sim(query, overrides=overrides, timing_mode=mode)
            sim.step_frames(1, record=False)          # warm (JIT cached now, graph captured)
            sim = backend.make_sim(query, overrides=overrides, timing_mode=mode)
            with timed(rec, "rollout_s"):
                sim.step_frames(n_frames, record=False)
            end = sim.export_state()["x"]
            row.update(rec)
            row["substeps_per_frame"] = sim.substeps_per_frame
            row["substeps"] = sim.substeps_done
            row["per_substep_us"] = 1e6 * rec["rollout_s"] / max(sim.substeps_done, 1)
            row["stable"] = not is_unstable(end)
            if row["stable"] and (best is None or rec["rollout_s"] < best["rollout_s"]):
                best = row
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if cuda_context_lost(e):
                row.update(stable=False, crashed=True)
                append_jsonl(row, out_path)
                exit_cuda_lost(f"{regime} qi={qi} {mode}", e)
        append_jsonl(row, out_path)
    return best or {}


def time_probes(backend, query: dict, mode: str, ms: list[int], reps: int, out_path: Path, tag: str,
                done: set):
    from verifier import gt_frames

    if query.get("x") is None:
        return
    frames = gt_frames(query["x"])
    for m in ms:
        if ("probe", query["qid"], m) in done:
            continue
        times = []
        for r in range(reps):
            rec = {}
            with timed(rec, "t"):
                sim = backend.make_sim(query, timing_mode=mode)
                s = 2 * 5 + 1
                sim.set_state(frames[s], None, None, None, time=s * sim.frame_dt)
                sim.step_frames(2 * m, record=False)
            times.append(rec["t"])
        append_jsonl({"kind": "probe", "hw": tag, "qid": query["qid"], "mode": mode, "m_model_frames": m,
                      "sim_frames": 2 * m, "times_s": times, "median_s": float(np.median(times))},
                     out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--tag", required=True, help="hardware tag: A100 or H200")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    run = start_run(f"p1_cost_{args.tag}", cfg, synthetic=args.dry_run)
    backend = make_backend(cfg, args.dry_run)
    out = run.path("timings.jsonl")
    prev = resume_rows(out, args.resume)
    done = {row_key(r) for r in prev}

    entries = backend.manifest(testset_manifest_path(args.dry_run), split="test_id")
    elastic = entries[: cfg["n_queries"]]
    base_queries = [backend.load(e) for e in elastic]

    g = cfg["generation"]
    for which in g["models"]:
        for q in base_queries[: g["n_queries"]]:
            fatal_guard(f"generation {which} {q['qid']}", time_generation, backend, q, which,
                        g["batch_sizes"], g["reps"], g["warmup"], out, args.tag, done)

    s = cfg["simulation"]
    # On --resume, recover the fastest stable physctrl mode from rows already written.
    prior = [r for r in prev if r["kind"] == "simulation" and r["regime"] == "physctrl"
             and r.get("stable") and "rollout_s" in r]
    best_mode = min(prior, key=lambda r: r["rollout_s"])["mode"] if prior else None
    for regime in s["regimes"]:
        if args.dry_run and regime != "physctrl":
            continue  # regime builders need the MPM solver's grid; the spring stand-in has none
        from regimes import build_regime

        for qi in range(s["n_queries"]):
            rotated = base_queries[qi:] + base_queries[:qi]
            q, overrides = build_regime(regime, rotated)
            best = time_simulation(backend, q, overrides, regime, qi, s["modes"], out, args.tag,
                                   N_SIM_FRAMES, done)
            if regime == "physctrl" and best and best_mode is None:
                best_mode = best["mode"]

    p = cfg["probes"]
    for q in base_queries[: p["n_queries"]]:
        fatal_guard(f"probes {q['qid']}", time_probes, backend, q, best_mode or "as_is",
                    p["m_model_frames"], p["reps"], out, args.tag, done)

    print(f"[p1_cost] wrote {out}")
    run.finish("ok")


if __name__ == "__main__":
    main()
