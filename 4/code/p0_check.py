"""Phase 0: environment, data and reproduction checks. Run on A100 and on H200.

1. Imports and versions; GPU visible.
2. Checkpoints and test set present; writes the test-set manifest that every
   later phase reads (results/p0_check/testset_manifest.json).
3. Replay determinism: re-simulate stored elastic test trajectories from frame 0
   with our MPMSim and compare to the stored positions. If this fails, the
   window defect in Phase 2 would measure simulator mismatch, not model error.
   Also checks the three timing modes agree with each other.
4. Generator sanity: base and large checkpoints load with every key matched,
   and their error against ground truth on a few elastic queries is finite and
   reported (model-coordinate MSE is the number PhysCtrl's eval.py prints).
5. Writes filled eval configs so the Slurm script can also run PhysCtrl's own
   eval.py as an independent cross-check.

Falsifier: replay error above `replay_tol` on any query, or any missing
checkpoint key. Either one stops the project until fixed.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, load_config, save_json, start_run  # noqa: E402
from backends import make_backend, N_SIM_FRAMES  # noqa: E402


def check_imports() -> dict:
    out = {}
    for mod in ["torch", "warp", "taichi", "h5py", "omegaconf", "diffusers", "torch_cluster",
                "xformers", "scipy", "safetensors", "yaml"]:
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", None) or getattr(getattr(m, "config", None), "version", "ok")
        except Exception as e:  # report, do not hide
            out[mod] = f"IMPORT FAILED: {type(e).__name__}: {e}"
    try:
        import torch

        out["cuda_available"] = torch.cuda.is_available()
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["cuda_available"] = False
    return out


def build_manifest(testset_dir: Path) -> list[dict]:
    from physctrl_bridge import load_h5

    files = sorted(glob.glob(str(testset_dir / "**" / "*.h5"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .h5 files under {testset_dir}. Did the testset.zip unpack there?")
    entries = []
    for f in files:
        d = load_h5(f)
        forces = np.asarray(d.get("drag_force", []))
        entries.append({
            "qid": Path(f).stem, "path": f, "split": "test_id",
            # Absent in the released test set; 0 is PhysCtrl's own default (see backends.py).
            "mat_type": int(np.asarray(d.get("mat_type", 0))),
            "gravity": int(np.asarray(d.get("gravity", 0))),
            "E": float(np.asarray(d["E"])), "nu": float(np.asarray(d["nu"])),
            "n_forces": int(forces.reshape(-1, 3).shape[0]) if forces.size else 0,
            "n_frames": int(d["x"].shape[0]), "n_points": int(d["x"].shape[1]),
        })
    return entries


def replay(backend, queries: list[dict], modes: list[str], tol: float) -> list[dict]:
    rows = []
    for q in queries:
        finals = {}
        for mode in modes:
            row = {"qid": q["qid"], "mode": mode}
            try:
                t0 = time.perf_counter()
                sim = backend.make_sim(q, timing_mode=mode)
                traj = sim.step_frames(N_SIM_FRAMES)
                row["seconds"] = time.perf_counter() - t0
                err = np.linalg.norm(traj - q["x"][1:], axis=2).mean(axis=1)   # per frame
                row["err_mean"] = float(err.mean())
                row["err_final"] = float(err[-1])
                row["pass"] = bool(err.max() < tol)
                finals[mode] = traj[-1]
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["pass"] = False
            rows.append(row)
        if "as_is" in finals:
            for mode, fx in finals.items():
                if mode != "as_is":
                    rows.append({"qid": q["qid"], "mode": f"{mode}_vs_as_is",
                                 "err_final": float(np.linalg.norm(fx - finals["as_is"], axis=1).mean())})
    return rows


def generator_sanity(backend, queries: list[dict], models: list[str], seed: int) -> list[dict]:
    from physctrl_bridge import to_model_coords

    rows = []
    for which in models:
        try:
            gen = backend.generate(which, queries, n_samples=1, seed=seed)
        except Exception as e:
            rows.append({"model": which, "error": f"{type(e).__name__}: {e}"})
            continue
        for q, g in zip(queries, gen):
            gt = q["x"][1::2][:24]
            l2 = np.linalg.norm(g[0] - gt, axis=2)
            rows.append({
                "model": which, "qid": q["qid"],
                "l2_mean": float(l2.mean()), "l2_final": float(l2[-1].mean()),
                "mse_model_coords": float(((to_model_coords(g[0]) - to_model_coords(gt)) ** 2).mean()),
            })
    return rows


def write_eval_configs(cfg: dict, run, manifest: list[dict]) -> None:
    """Filled copies of PhysCtrl's eval yamls for their own eval.py."""
    from omegaconf import OmegaConf

    root = Path(os.environ["PHYSCTRL_ROOT"])
    elastic = [e for e in manifest if e["mat_type"] == 0][: cfg["eval_cross_check_n"]]
    folder = Path(elastic[0]["path"]).parent
    lst = run.path("physctrl_eval", "elastic_test_list.json")
    save_json([Path(e["path"]).name for e in elastic if Path(e["path"]).parent == folder], lst)
    for which in ("base", "large"):
        c = OmegaConf.load(root / "src" / "configs" / f"eval_{which}.yaml")
        c.train_dataset.dataset_path = str(folder)
        c.train_dataset.dataset_list = str(lst)
        c.resume = str(Path(cfg["paths"]["ckpt_dir"]) / f"physctrl_{which}.safetensors")
        c.vis_dir = str(run.path("physctrl_eval", f"vis_{which}", ".keep").parent)
        OmegaConf.save(c, run.path("physctrl_eval", f"eval_{which}_filled.yaml"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--tag", default="", help="suffix for the results folder, e.g. A100 or H200")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    run = start_run("p0_check" + (f"_{args.tag}" if args.tag else ""), cfg, synthetic=args.dry_run)
    backend = make_backend(cfg, args.dry_run)
    report = {"imports": check_imports()}

    if args.dry_run:
        manifest = backend.manifest(split="test_id")
    else:
        p = cfg["paths"]
        report["checkpoints"] = {w: (Path(p["ckpt_dir"]) / f"physctrl_{w}.safetensors").exists()
                                 for w in ("base", "large")}
        manifest = build_manifest(Path(p["testset_dir"]))
        save_json(manifest, run.path("testset_manifest.json"))
        counts = {}
        for e in manifest:
            counts[e["mat_type"]] = counts.get(e["mat_type"], 0) + 1
        report["testset_counts_by_mat_type"] = counts
        write_eval_configs(cfg, run, manifest)

    elastic = [e for e in manifest if int(e["mat_type"]) == 0]
    queries = [backend.load(e) for e in elastic[: cfg["replay_n"]]]
    report["replay"] = replay(backend, queries, cfg["timing_modes"], cfg["replay_tol"])
    report["replay_all_pass"] = all(r.get("pass", True) for r in report["replay"] if "pass" in r)

    gq = [backend.load(e) for e in elastic[: cfg["generator_n"]]]
    report["generator"] = generator_sanity(backend, gq, cfg["models"], cfg["seed"])
    report["generator_ok"] = all("error" not in r and np.isfinite(r["l2_mean"]) for r in report["generator"])

    save_json(report, run.path("p0_report.json"))
    ok = report["replay_all_pass"] and report["generator_ok"]
    print(f"[p0] replay_all_pass={report['replay_all_pass']} generator_ok={report['generator_ok']}")
    print(f"[p0] report: {run.path('p0_report.json')}")
    run.finish("ok" if ok else "checks_failed")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
