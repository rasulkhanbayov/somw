"""Shared utilities: config loading, run metadata, result writing, timing, statistics.

Every experiment script goes through `start_run` so that each results folder
carries the same provenance record (host, GPU, Slurm job, git commits, config).
Synthetic (dry-run) runs are written under a SYNTHETIC_ prefix and the
summarizers refuse to read them unless explicitly asked to.
"""
from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("SOMW_RESULTS", PROJECT_ROOT / "results"))
SYNTHETIC_PREFIX = "SYNTHETIC_"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path, dry_run: bool = False) -> dict:
    """Load a YAML config, expanding ${ENV_VAR} references in string values.

    Fails loudly if a referenced environment variable is unset, so a path never
    silently resolves to a literal '${DATA_ROOT}/...'. In dry-run mode the
    cluster variables get placeholder values, since the synthetic backend never
    touches those paths.
    """
    import yaml

    if dry_run:
        for var in ("DATA_ROOT", "PHYSCTRL_ROOT"):
            os.environ.setdefault(var, f"/dry-run-unset/{var}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def expand(v):
        if isinstance(v, str):
            out = os.path.expandvars(v)
            if "${" in out:
                raise RuntimeError(f"Unexpanded environment variable in config value: {v!r}")
            return out
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v

    return expand(cfg)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _gpu_info() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


class Run:
    """A results folder plus its provenance record."""

    def __init__(self, name: str, cfg: dict, synthetic: bool, shard: str | None = None):
        self.name = name
        self.synthetic = synthetic
        folder = (SYNTHETIC_PREFIX + name) if synthetic else name
        self.dir = RESULTS_ROOT / folder
        if shard is not None:
            self.dir = self.dir / f"shard_{shard}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {
            "run_name": name,
            "synthetic": synthetic,
            "shard": shard,
            "start_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "argv": sys.argv,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "gpus": _gpu_info(),
            "project_git_commit": _git_commit(PROJECT_ROOT),
            "physctrl_git_commit": _git_commit(Path(os.environ["PHYSCTRL_ROOT"]))
            if os.environ.get("PHYSCTRL_ROOT") else None,
            "config": cfg,
        }
        try:
            import torch
            self.meta["torch"] = torch.__version__
        except Exception:
            pass
        try:
            import warp as wp
            self.meta["warp"] = wp.config.version
        except Exception:
            pass
        save_json(self.meta, self.dir / "run_meta.json")

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def finish(self, status: str = "ok", **extra) -> None:
        self.meta["end_utc"] = datetime.now(timezone.utc).isoformat()
        self.meta["status"] = status
        self.meta.update(extra)
        save_json(self.meta, self.dir / "run_meta.json")


def start_run(name: str, cfg: dict, synthetic: bool, shard: str | None = None) -> Run:
    return Run(name, cfg, synthetic, shard)


def results_dir(name: str, allow_synthetic: bool = False) -> Path:
    """Locate a results folder for reading. Refuses synthetic folders by default."""
    real = RESULTS_ROOT / name
    if real.exists():
        return real
    syn = RESULTS_ROOT / (SYNTHETIC_PREFIX + name)
    if syn.exists():
        if allow_synthetic:
            return syn
        raise RuntimeError(
            f"Only a synthetic results folder exists for '{name}' ({syn}). "
            "Synthetic numbers must never reach paper tables; pass --allow-synthetic "
            "only for pipeline testing."
        )
    raise FileNotFoundError(f"No results folder for '{name}' under {RESULTS_ROOT}")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_to_jsonable(x) for x in o.tolist()]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    return o


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    os.replace(tmp, path)


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_to_jsonable(row)) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shard_items(items: list, shard: int, num_shards: int) -> list:
    require(0 <= shard < num_shards, f"bad shard {shard}/{num_shards}")
    return items[shard::num_shards]


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import warp as wp

        if wp.is_cuda_available():
            wp.synchronize()
    except Exception:
        pass


@contextmanager
def timed(record: dict, key: str):
    """Wall-clock a block with device synchronisation on both sides."""
    cuda_sync()
    t0 = time.perf_counter()
    yield
    cuda_sync()
    record[key] = time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(y)     # +inf ranks last: Phase 2 scores blow-ups as +inf
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def auroc(score, label) -> float:
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney).
    This is the single-threshold question a certificate asks; +inf scores rank highest."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(stat_fn, arrays: list[np.ndarray], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over the shared first axis of `arrays` (unit = query)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = stat_fn(*arrays)
    if n < 3:
        return point, float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(stat_fn(*[a[idx] for a in arrays]))
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(stat_fn, a_arrays, b_arrays, n_boot=2000, alpha=0.05, seed=0):
    """CI for stat(a) - stat(b), resampling the same query indices for both."""
    rng = np.random.default_rng(seed)
    n = len(a_arrays[0])
    point = stat_fn(*a_arrays) - stat_fn(*b_arrays)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(stat_fn(*[x[idx] for x in a_arrays]) - stat_fn(*[x[idx] for x in b_arrays]))
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def add_common_args(parser) -> None:
    parser.add_argument("--config", required=True, help="YAML config under configs/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run end-to-end on the tiny CPU synthetic backend. "
                             "Outputs go to results/SYNTHETIC_<name>/.")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (Slurm array task)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Keep rows already written and skip their units. Passed by "
                             "run_resumable (slurm/_common.sh) when restarting after exit 75.")


# --------------------------------------------------------------------------- #
# CUDA context loss
# --------------------------------------------------------------------------- #
# A rollout that blows up sends particles outside the MPM grid, and Warp's kernels then
# write out of bounds. That error is sticky: every later CUDA call in the process fails,
# including the generator's (seen in p1_fidelity on Oct 10, 2026, where one coarse-substep
# rollout turned the rest of the shard into error rows). The only clean recovery is a new
# process, so scripts record the crashed unit, exit with EXIT_CUDA_LOST, and the job
# wrapper restarts them with --resume.
EXIT_CUDA_LOST = 75
_STICKY_CUDA_ERRORS = ("illegal memory access", "unspecified launch failure",
                       "misaligned address", "illegal instruction", "device-side assert")


def cuda_context_lost(ex: BaseException) -> bool:
    msg = str(ex)
    return any(s in msg for s in _STICKY_CUDA_ERRORS)


def exit_cuda_lost(where: str, ex: BaseException):
    first = (str(ex).splitlines() or [""])[0]
    print(f"[cuda-lost] {where}: {type(ex).__name__}: {first}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(EXIT_CUDA_LOST)   # skip interpreter teardown: it would touch the dead context


def resume_rows(path: str | Path, resume: bool) -> list[dict]:
    """Fresh run: delete any old file. --resume: return the rows already written."""
    path = Path(path)
    if not resume:
        if path.exists():
            path.unlink()
        return []
    return read_jsonl(path) if path.exists() else []
