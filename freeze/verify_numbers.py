"""Re-derive every reported number from the stored data and compare it with the stored result (CPU, dev server).

Every results/<run>/SUMMARY.md written on a dev server (run_meta.json: no Slurm job) was produced by one command, kept
in run_meta.json "argv". This copies the project's code/configs/paper/slurm and results/ to a scratch folder, re-runs
each such command there in the original order (later verdicts read earlier ones), with SOMW_RESULTS and PROJECT_ROOT
pointing at the copy and the data read from DATA_ROOT / BIG_ROOT, and compares the new SUMMARY.md with the stored one
byte for byte. The tables the scripts write into paper/tab/ are compared too. Nothing in the real results/ changes.
Not re-run here, with the reason printed: GPU/cluster jobs (their outputs are the stored rows and trajectories that these
verdicts read), and the Phase 12 S0 gate (hours of PolyFEM on cpu-all; job 102).
Output: <work>/VERIFY.md (one line per verdict: IDENTICAL / DIFFERS / FAILED) and logs per command; a full run
also copies the report to freeze/VERIFY.md. Default work folder: $BIG_ROOT/freeze_verify (excluded from the archive).
Usage: python freeze/verify_numbers.py [--work DIR] [--only p9_confirm,p10c_summary] [--timeout 7200]
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from areas import BIG, PROJECT  # noqa: E402

EXTRA = {"p10d_summary": ["code/p10d_strands.py", "--summarize"]}          # runs made before run_meta covered them
SKIP = {"p12_s0": "the S0 gate itself (PolyFEM, hours on cpu-all: slurm/102_p12_s0_cpu.sh)"}


def commands():
    out = []
    for s in sorted((PROJECT / "results").glob("*/SUMMARY.md")):
        run = s.parent.name
        m = s.parent / "run_meta.json"
        if run in SKIP:
            out.append((run, None, SKIP[run], ""))
            continue
        if not m.exists():
            if run in EXTRA:
                out.append((run, EXTRA[run], None, "9999"))
            else:
                out.append((run, None, "no run_meta.json and no known command", ""))
            continue
        r = json.loads(m.read_text())
        if r.get("slurm_job_id"):
            out.append((run, None, f"written by Slurm job {r['slurm_job_id']}", r.get("start_utc", "")))
            continue
        out.append((run, r["argv"], None, r.get("start_utc", "")))
    return sorted(out, key=lambda x: x[3])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default=str(BIG / "freeze_verify"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    try:                                            # one run at a time: a second one would race on the copy
        lock = os.open(work / ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock, str(os.getpid()).encode())
    except FileExistsError:
        sys.exit(f"another verify_numbers.py is using {work} (remove {work / '.lock'} if it is not running)")
    proj = work / "4"
    if not (work / ".copy_complete").exists():      # copy to a temporary folder, renamed only when complete
        shutil.rmtree(proj, ignore_errors=True)
        tmp = work / "4.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        for d in ("code", "configs", "paper", "slurm", "results"):
            shutil.copytree(PROJECT / d, tmp / d, symlinks=True)
        tmp.rename(proj)
        (work / ".copy_complete").write_text("ok\n")
    env = dict(os.environ, SOMW_RESULTS=str(proj / "results"), PROJECT_ROOT=str(proj), PYTHONNOUSERSITE="1")
    keep = set(args.only.split(",")) if args.only else None
    lines = ["# Re-derivation of every stored verdict (freeze/verify_numbers.py)\n", f"Work folder: {work}\n",
             "| run | command | result | seconds |", "|---|---|---|---|"]
    counts = {}
    for run, argv, why, _ in commands():
        if keep and run not in keep:
            continue
        if argv is None:
            lines.append(f"| {run} | — | not re-run: {why} | |")
            counts["not re-run"] = counts.get("not re-run", 0) + 1
            continue
        t = time.time()
        log = work / "logs" / f"{run}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            p = subprocess.run([sys.executable] + argv, cwd=proj, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                               timeout=args.timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        new, old = proj / "results" / run / "SUMMARY.md", PROJECT / "results" / run / "SUMMARY.md"
        if rc != 0:
            res = f"FAILED (exit {rc}; see logs/{run}.log)"
        elif new.read_bytes() == old.read_bytes():
            res = "IDENTICAL"
        else:
            d = list(difflib.unified_diff(old.read_text().splitlines(), new.read_text().splitlines(), lineterm="", n=0))
            (work / "logs" / f"{run}.diff").write_text("\n".join(d) + "\n")
            res = f"DIFFERS ({sum(1 for x in d if x.startswith(('+', '-')) and not x.startswith(('+++', '---')))} changed lines; logs/{run}.diff)"
        key = res.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
        lines.append(f"| {run} | `{' '.join(argv)}` | {res} | {time.time() - t:.0f} |")
        print(f"[verify] {run}: {res}", flush=True)
        (work / "VERIFY.md").write_text("\n".join(lines) + "\n")
    tabs = []
    for f in sorted((PROJECT / "paper" / "tab").glob("*")):
        g = proj / "paper" / "tab" / f.name
        tabs.append(f"- paper/tab/{f.name}: " + ("IDENTICAL" if g.exists() and g.read_bytes() == f.read_bytes() else "DIFFERS"))
    lines += ["", "## Paper tables", ""] + tabs + ["", "## Totals", ""] + [f"- {k}: {v}" for k, v in sorted(counts.items())]
    (work / "VERIFY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(counts) - 1:]))
    if not keep:                                    # the full report travels with the code (git) and the results repo
        shutil.copyfile(work / "VERIFY.md", Path(__file__).resolve().parent / "VERIFY.md")
    print(f"[verify] report: {work / 'VERIFY.md'}")
    os.close(lock)
    (work / ".lock").unlink()


if __name__ == "__main__":
    main()
