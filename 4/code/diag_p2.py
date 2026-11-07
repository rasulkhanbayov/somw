"""Phase 2 diagnostics (CPU only): why placement, normalisation and the large model behave
as they do in results/p2_summary_v2. Reads existing generations, ground truth and probe rows;
runs no simulator. Exploratory analysis: nothing here decides a gate.

Per generated trajectory (sample 0) against ground truth it measures
  - the temporal profile of position error e(t) over the 24 model frames, and how much of
    the error *growth* falls in the single worst 1-step window (localisation; a uniform
    spread over the 23 steps gives 1/23 = 0.043);
  - onset: first frame where e(t) reaches half its final value;
  - error type: per-frame best-fit translation and affine maps from generated to true
    points, giving the share of error that is a rigid shift / any affine map / non-affine;
  - jitter: mean second-difference magnitude, generated over true;
  - gravity drop only: the first frame with floor contact in the ground truth, and
    whether the event placement put a window there.
Also reports Spearman(signal, true error) per split, which the pooled tables hide.

Outputs: results/diag_p2/{DIAG.md, per_traj.jsonl}
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append_jsonl, read_jsonl, spearman  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gt_path(qid: str, split: str, data_root: Path, ood_index: dict) -> Path:
    return data_root / "testset" / f"{qid}.h5" if split == "test_id" else Path(ood_index[qid])


def affine_residual(g: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Mean L2 left after the best translation, and after the best affine map g -> t."""
    shift = t.mean(0) - g.mean(0)
    r_trans = float(np.linalg.norm(g + shift - t, axis=1).mean())
    A = np.c_[g, np.ones(len(g))]
    M, *_ = np.linalg.lstsq(A, t, rcond=None)
    r_aff = float(np.linalg.norm(A @ M - t, axis=1).mean())
    return r_trans, r_aff


def analyse(g: np.ndarray, gt: np.ndarray, floor: float | None, windows: list | None) -> dict:
    e = np.linalg.norm(g - gt, axis=2).mean(axis=1)                    # (24,)
    growth = np.clip(np.diff(e), 0, None)                              # 23 one-step increments
    tot = growth.sum()
    loc = float(growth.max() / tot) if tot > 0 else float("nan")
    top3 = float(np.sort(growth)[-3:].sum() / tot) if tot > 0 else float("nan")
    onset = int(np.argmax(e >= 0.5 * e[-1])) if e[-1] > 0 else -1
    rt, ra = zip(*(affine_residual(g[k], gt[k]) for k in (5, 11, 17, 23)))
    e_s = e[[5, 11, 17, 23]]
    jit = lambda x: np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=2).mean()
    out = {"e_final": float(e[-1]), "e_mean": float(e.mean()), "e_first": float(e[0]),
           "loc_top1": loc, "loc_top3": top3, "onset": onset,
           "share_nonshift": float(np.mean(np.array(rt) / np.maximum(e_s, 1e-9))),
           "share_nonaffine": float(np.mean(np.array(ra) / np.maximum(e_s, 1e-9))),
           "jitter_ratio": float(jit(g) / max(jit(gt), 1e-12)),
           "worst_step": int(np.argmax(growth))}
    if floor is not None:
        gap = gt[..., 1].min(axis=1) - floor
        hit = np.nonzero(gap < 0.05)[0]
        out["contact_frame"] = int(hit[0]) if len(hit) else -1
        if windows and len(hit):
            # window (start_sim_frame s, length L) covers model frames (s-1)/2 .. (s-1+L)/2
            covered = set()
            for s, L, _a in windows:
                covered.update(range(max((s - 1) // 2, 0), (s - 1 + L) // 2 + 1))
            out["event_covers_contact"] = bool(int(hit[0]) in covered)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-run", default="p2b_probe_v2")
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()
    data_root = Path(args.data_root)
    res = ROOT / "results"
    out_dir = res / "diag_p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    per = out_dir / "per_traj.jsonl"
    if per.exists():
        per.unlink()

    ood_index = {e["qid"]: e["path"] for f in glob.glob(str(res / "p2_build_ood/shard_*/ood_manifest.json"))
                 for e in json.load(open(f))}
    probes = [r for f in sorted(glob.glob(str(res / args.probe_run / "shard_*/probes.jsonl")))
              for r in read_jsonl(f)]
    rows = []
    for r in probes:
        npz = data_root / "generations" / r["model"] / r["split"] / f"{r['qid']}.npz"
        g = np.load(npz)["samples"][0].astype(np.float64)
        with h5py.File(gt_path(r["qid"], r["split"], data_root, ood_index), "r") as h:
            gt = np.asarray(h["x"], dtype=np.float64)[1::2][:24]
            floor = float(np.asarray(h["floor_height"])) if r["split"] == "ood_gravity_drop" else None
        d = analyse(g, gt, floor, r.get("event__windows"))
        d.update({k: r[k] for k in ("qid", "model", "split", "err_mean", "seed_spread")})
        for s in ("uniform", "event", "oracle", "proposal"):
            d[f"{s}__raw_max"] = r.get(f"{s}__raw_max")
        rows.append(d)
        append_jsonl(d, per)

    md = ["# Phase 2 diagnostics (exploratory; does not decide any gate)\n",
          f"Source: `results/{args.probe_run}`, sample 0 of each generation vs ground truth. "
          "Uniform spread of error growth over 23 steps gives loc_top1 = 0.043, loc_top3 = 0.13.\n",
          "## 1. Where and what the error is (medians)\n",
          "| model | split | n | e first | e final | loc top1 | loc top3 | onset frame | "
          "non-shift share | non-affine share | jitter gen/true |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    med = lambda rs, k: float(np.nanmedian([x[k] for x in rs]))
    groups = defaultdict(list)
    for d in rows:
        groups[(d["model"], d["split"])].append(d)
    for (m, s), rs in sorted(groups.items()):
        md.append(f"| {m} | {s} | {len(rs)} | {med(rs, 'e_first'):.4f} | {med(rs, 'e_final'):.4f} | "
                  f"{med(rs, 'loc_top1'):.3f} | {med(rs, 'loc_top3'):.3f} | {med(rs, 'onset'):.0f} | "
                  f"{med(rs, 'share_nonshift'):.2f} | {med(rs, 'share_nonaffine'):.2f} | "
                  f"{med(rs, 'jitter_ratio'):.2f} |")

    md += ["\n## 2. Spearman with true error, per split\n",
           "| model | split | seed_spread | uniform | event | proposal | oracle |",
           "|---|---|---|---|---|---|---|"]
    for (m, s), rs in sorted(groups.items()):
        e = np.array([x["err_mean"] for x in rs])
        cells = []
        for k in ("seed_spread", "uniform__raw_max", "event__raw_max", "proposal__raw_max", "oracle__raw_max"):
            v = np.array([np.nan if x[k] is None else x[k] for x in rs], dtype=float)
            cells.append(f"{spearman(v, e):.2f}")
        md.append(f"| {m} | {s} | " + " | ".join(cells) + " |")

    gd = [d for d in rows if d["split"] == "ood_gravity_drop" and d.get("contact_frame", -1) >= 0]
    if gd:
        md += ["\n## 3. Gravity drop: contact vs error growth\n",
               "| model | n | median contact frame | median worst-growth step | "
               "share where worst step is within 1 frame of contact | event windows cover contact |",
               "|---|---|---|---|---|---|"]
        for m in sorted({d["model"] for d in gd}):
            rs = [d for d in gd if d["model"] == m]
            near = np.mean([abs(d["worst_step"] + 1 - d["contact_frame"]) <= 1 for d in rs])
            cov = np.mean([d.get("event_covers_contact", False) for d in rs])
            md.append(f"| {m} | {len(rs)} | {med(rs, 'contact_frame'):.0f} | {med(rs, 'worst_step'):.0f} | "
                      f"{near:.2f} | {cov:.2f} |")
    (out_dir / "DIAG.md").write_text("\n".join(md) + "\n")
    print(f"[diag_p2] wrote {out_dir / 'DIAG.md'} ({len(rows)} trajectories)")


if __name__ == "__main__":
    main()
