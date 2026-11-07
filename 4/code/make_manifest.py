"""Write a query manifest (the testset_manifest.json format) for a folder of PhysCtrl-format
elastic HDF5 trajectories. CPU only. Used for the Phase 4g training objects (PhysCtrl's
released sample/ set), so p2_build_ood.py can re-simulate them with new physics draws.

    python code/make_manifest.py --folder $DATA_ROOT/sample --split train_obj --out results/p4g_sample_manifest/manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = []
    for p in sorted(Path(args.folder).glob("*.h5")):
        with h5py.File(p, "r") as h:
            out.append({"qid": p.stem, "path": str(p), "split": args.split,
                        # Absent keys default to 0, as in PhysCtrl's TrajDataset (see backends.py).
                        "mat_type": int(np.asarray(h["mat_type"])) if "mat_type" in h else 0,
                        "gravity": int(np.asarray(h["gravity"])) if "gravity" in h else 0,
                        "E": float(np.asarray(h["E"])), "nu": float(np.asarray(h["nu"])),
                        "n_forces": int(np.asarray(h["drag_force"]).reshape(-1, 3).shape[0]),
                        "n_frames": int(h["x"].shape[0]), "n_points": int(h["x"].shape[1])})
    save_json(out, args.out)
    print(f"[make_manifest] {len(out)} entries -> {args.out}")


if __name__ == "__main__":
    main()
