"""Phase 11 V2: human preference study verdict (pre-registered Oct 20, 2026; analysis rules fixed Oct 21 before any
response existed; PLAN.md "Phase 11"). CPU.

Input: the study page's responses exported from its db (one document per rater: {answers: {item: {q1, q2, left}}}),
answer codes sent by raters without a claude.ai account (--codes, one per line; amended Oct 22 before any response:
SOMW1-<rater>-<q1 q2 left per item in the page's item order>-<FNV-1a checksum>), and
results/p11_study_design/mapping.json (item -> case, which letter is ours). A rater present in both keeps the more
complete record; a code with a wrong checksum or length is rejected and reported.
V2: the share of answers preferring ours on Q1 (physical plausibility, asked before the reference is shown), over all
(rater, item) answers; 95% CI by bootstrap over raters (2,000 resamples). PASS if >= 10 raters answered at least one
item and the lower bound is above 0.5. No exclusions. Also reported: Q2 (closer to the reference), per item group
(Phase 11 images, where ours' video is the reference video; PhysCtrl's 3x examples, repaired mid-clip), per rater.
  --sandbox DIR: read DIR/responses.json and DIR/mapping.json, write DIR/SUMMARY.md (code test only)
Output: results/p11_study/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, save_json  # noqa: E402

B, MIN_RATERS = 2000, 10


def fnv(s):
    """The page's 32-bit FNV-1a hash (index.html hash())."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def parse_codes(lines, order):
    """Answer codes -> {rater: {answers: ...}}; bad codes are returned separately, never guessed."""
    docs, bad = {}, []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # By position, not by splitting: unanswered items are "-" and account ids may contain "-" too.
        n = 3 * len(order)
        core, chk = ln[:-9], ln[-8:]
        body, rater = core[-n:], core[len("SOMW1-"):-n - 1]
        if (not core.startswith("SOMW1-") or ln[-9] != "-" or core[-n - 1] != "-" or not rater
                or f"{fnv(core):08x}" != chk or set(body) - set("ab-")):
            bad.append(ln[:40])
            continue
        ans = {}
        for i, it in enumerate(order):
            q1, q2, left = body[3 * i:3 * i + 3]
            if q1 != "-":
                ans[it] = {"q1": q1, "q2": None if q2 == "-" else q2, "left": left}
        if len(ans) > len((docs.get(rater) or {}).get("answers", {})):
            docs[rater] = {"answers": ans}
    return docs, bad


def share(rows, q, raters):
    """Share of answers choosing ours on question q, with a rater-clustered bootstrap CI."""
    by = {r: [x[q] for x in rows if x["rater"] == r and x[q] is not None] for r in raters}
    by = {r: v for r, v in by.items() if v}
    if not by:
        return None
    ids = sorted(by)
    pt = float(np.mean(np.concatenate([by[r] for r in ids])))
    rng = np.random.default_rng(1)
    bs = [np.mean(np.concatenate([by[r] for r in rng.choice(ids, len(ids))])) for _ in range(B)]
    return pt, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), sum(len(v) for v in by.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", help="responses.json exported from the study db ({rater: document})")
    ap.add_argument("--codes", help="text file of answer codes, one per line")
    ap.add_argument("--items", default=None, help="the page's items.json (item order of the codes)")
    ap.add_argument("--sandbox", default=None)
    args = ap.parse_args()
    if args.sandbox:
        src, mp = Path(args.sandbox) / "responses.json", Path(args.sandbox) / "mapping.json"
    else:
        src = Path(args.responses) if args.responses else None
        mp = Path(RESULTS_ROOT) / "p11_study_design" / "mapping.json"
    M = json.loads(mp.read_text())["items"]
    docs = json.loads(src.read_text()) if src and src.exists() else {}
    bad = []
    if args.codes:
        order = [x["id"] for x in json.loads(Path(args.items).read_text())]
        cdocs, bad = parse_codes(Path(args.codes).read_text().splitlines(), order)
        for r, d in cdocs.items():
            if len(d["answers"]) > len((docs.get(r) or {}).get("answers") or {}):
                docs[r] = d
    rows = []
    for rater, d in docs.items():
        for item, a in (d.get("answers") or {}).items():
            if item not in M:
                raise RuntimeError(f"unknown item {item}")
            ours = M[item]["ours"]
            rows.append({"rater": rater, "item": item, "group": "physctrl_x3" if "force_x3" in M[item]["case"] else "phase11",
                         "q1": None if a.get("q1") is None else float(a["q1"] == ours),
                         "q2": None if a.get("q2") is None else float(a["q2"] == ours)})
    raters = sorted({x["rater"] for x in rows})
    f = lambda c: "—" if c is None else f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}] (n {c[3]})"         # noqa: E731
    res = {}
    md = ["# Phase 11 V2: human preference study (pre-registered Oct 20, 2026; analysis rules fixed Oct 21)\n",
          f"Raters with at least one answer: {len(raters)}; answers: {len(rows)}; items: {len(M)}; "
          f"rejected answer codes: {len(bad)} {bad}.\n",
          "| group | Q1 prefers ours (plausibility) | Q2 ours closer to the reference |", "|---|---|---|"]
    for g, sel in (("all", rows), ("Phase 11 images (ours = reference video)", [x for x in rows if x["group"] == "phase11"]),
                   ("PhysCtrl 3x examples (repaired)", [x for x in rows if x["group"] == "physctrl_x3"])):
        res[g] = {"q1": share(sel, "q1", raters), "q2": share(sel, "q2", raters)}
        md.append(f"| {g} | {f(res[g]['q1'])} | {f(res[g]['q2'])} |")
    q1 = res["all"]["q1"]
    v2 = q1 is not None and len(raters) >= MIN_RATERS and q1[1] > 0.5
    md += ["", f"- **V2 (share preferring ours on Q1 > 0.5, >= {MIN_RATERS} raters): "
               + ("PASS" if v2 else ("NOT RUN (fewer than 10 raters)" if len(raters) < MIN_RATERS else "FAIL")) + "**", "",
           "## Per rater (Q1 share preferring ours, answers)", ""]
    md += [f"- {r[:10]}…: {np.mean([x['q1'] for x in rows if x['rater'] == r and x['q1'] is not None]):.2f} "
           f"({sum(1 for x in rows if x['rater'] == r)})" for r in raters]
    out = Path(args.sandbox) if args.sandbox else Path(RESULTS_ROOT) / "p11_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"V2": v2, "n_raters": len(raters), "n_answers": len(rows), "res": res}, out / "summary.json")
    print(f"[p11 study] wrote {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
