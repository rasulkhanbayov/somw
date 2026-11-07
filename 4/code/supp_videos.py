"""Supplementary videos (Oct 20, 2026, after the paper review). CPU (system ffmpeg), existing renders only.

For every case with a Phase 9 record (case_v2.json) under the given render roots: PhysCtrl | ours | reference side by
side (720 x 480 each), each labelled with its name and its mean trajectory error from the case record; ours also states
the decision (accepted / repaired from frame k / simulated). Output: $BIG_ROOT/supp_videos/<case>.mp4 and an index.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def label(txt):
    t = txt.replace(":", "\\:").replace("'", "")
    return f"drawtext=fontfile={FONT}:text='{t}':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=0x14213Dcc:boxborderw=6"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=[os.path.expandvars("${BIG_ROOT}/renders")])
    ap.add_argument("--out", default=os.path.expandvars("${BIG_ROOT}/supp_videos"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for root in args.roots:
        for c in sorted(Path(root).glob("*/case_v2.json")):
            rec = json.loads(c.read_text())
            case, cd, e, dec = rec["case"], c.parent, rec["errors"], rec["decision"]
            act = "accepted (= PhysCtrl)" if dec["accepted"] else dec["action"].replace("repair from k=", "repaired from frame ")
            parts = [("generation", f"PhysCtrl  error {e['generation']:.3f}"),
                     ("certified_v2", f"ours  error {e['certified_v2']:.3f}  ({act})"),
                     ("ground_truth", "reference simulation")]
            vids = [cd / "variants" / v / "result.mp4" for v, _ in parts]
            if not all(v.exists() for v in vids):
                print(f"[supp] {case}: missing render, skipped", flush=True)
                continue
            filt = ";".join(f"[{i}:v]{label(t)}[v{i}]" for i, (_, t) in enumerate(parts)) + ";[v0][v1][v2]hstack=inputs=3[out]"
            dst = out / f"{case}.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error"] + sum((["-i", str(v)] for v in vids), []) + \
                  ["-filter_complex", filt, "-map", "[out]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)]
            subprocess.run(cmd, check=True)
            index.append(f"{case}\t{e['generation']:.4f}\t{e['certified_v2']:.4f}\t{act}")
            print(f"[supp] {dst.name}", flush=True)
    (out / "index.txt").write_text("case\tPhysCtrl error\tours error\tdecision\n" + "\n".join(index) + "\n")


if __name__ == "__main__":
    main()
