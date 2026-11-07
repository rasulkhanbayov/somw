"""Phase 12 (pre-registered Oct 22, 2026; PLAN.md "Phase 12"): the independent second simulator S2 = PolyFEM
(fixed-corotated, implicit Newmark), built from source at commit 008a1aea ($BIG_ROOT/tools/polyfem). CPU only; runs in
the separate conda env `somw-fem` (numpy, scipy, h5py, meshio, pyyaml).

Per query (a PhysCtrl-format HDF5):
  mesh   (amended Oct 22 before any query was simulated; PLAN "Phase 12") the grid cells (dx = 10/125) occupied by
         the particles, i.e. the body the MPM actually simulates: PhysCtrl's points are surface samples, so its MPM
         object is a one-cell-thick shell and sum(vol) = n_cells * dx^3 exactly (checked on development queries).
         Each cell is split into 6 tetrahedra (Kuhn); particles are embedded barycentrically in their cell. Excluded
         if the largest face-connected set of cells holds < 99% of the particles. Density 1000 (mass matches exactly).
  loads  gravity (gravity == 1): 9.81 downward, initial acceleration = gravity.
         drag: a body force on the drag box (half-size 0.4 around drag_point, which equals the stored drag_mask, checked
         on development queries), sized so the total force equals the MPM's n_masked * f; the same field is the
         initial acceleration (PolyFEM's integrator otherwise starts from zero acceleration; smoke test Oct 22).
         Amended Oct 24 after S0 (job 432388) failed S0c: the box is snapped to the grid cells whose centre lies inside
         it, so the field is constant on every tetrahedron (PolyFEM's quadrature integrates it exactly) and the force
         is normalised by that mesh volume. The first version normalised by the masked particles' volume, which differs
         from the mesh volume inside the box by up to 5% (correlation 0.85 with the S0c errors).
  output node positions at simulator frames 1, 3, ..., 47 (= the stored ground truth x[1::2][:24]).
PolyFEM's rhs moves the body opposite to its sign (smoke test), so rhs = -(body acceleration).
CLI: run a list of queries -> <out>/<qid>.npz (positions, info) and <out>/rows.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append_jsonl  # noqa: E402

BIN = Path(os.path.expandvars("${BIG_ROOT}/tools/polyfem/build_conda/PolyFEM_bin"))   # conda toolchain, glibc 2.17 sysroot, -march=x86-64-v2 (job 431597: the host-built binary needed glibc 2.33)
FRAME = 417 * 1e-4            # PhysCtrl's simulator frame (s)
OUT_FRAMES = list(range(1, 48, 2))
DENSITY, BOX = 1000.0, 0.4


def load_query(path):
    import h5py
    with h5py.File(path, "r") as h:
        q = {k: np.asarray(h[k]) for k in ("x", "vol", "E", "nu", "drag_point", "drag_mask", "drag_force", "floor_height")
             if k in h}
        q["gravity"] = int(np.asarray(h["gravity"])) if "gravity" in h else 0
    q["x0"] = q["x"][0].astype(np.float64)
    q["gt"] = q["x"][1::2][:24].astype(np.float64)
    q["qid"] = Path(path).stem
    return q


DX = 10.0 / 125
KUHN = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def mesh_from_points(x0, vol):
    """Occupied-cell tetrahedral mesh; returns (vertices, tets, embedding (node ids (N,4), weights (N,4)), info)."""
    ijk = np.floor(x0 / DX).astype(np.int64)
    cells, inv = np.unique(ijk, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    # largest face-connected component of occupied cells
    index = {tuple(c): i for i, c in enumerate(cells)}
    comp = -np.ones(len(cells), np.int64)
    nc = 0
    for i in range(len(cells)):
        if comp[i] >= 0:
            continue
        stack, comp[i] = [i], nc
        while stack:
            a = cells[stack.pop()]
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                j = index.get((a[0] + d[0], a[1] + d[1], a[2] + d[2]))
                if j is not None and comp[j] < 0:
                    comp[j] = nc
                    stack.append(j)
        nc += 1
    big = np.bincount(comp[inv]).argmax()
    share = float(np.mean(comp[inv] == big))
    # cell corners -> shared vertices; 6 Kuhn tetrahedra per cell
    corners = (cells[:, None, :] + np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])[None]).reshape(-1, 3)
    verts, vinv = np.unique(corners, axis=0, return_inverse=True)
    vid = vinv.reshape(len(cells), 8)
    def corner(cell_vids, off):
        return cell_vids[:, off[0] * 4 + off[1] * 2 + off[2]]
    tets, eye = [], np.eye(3, dtype=int)
    for a, b, c in KUHN:
        o0, o1 = np.zeros(3, int), eye[a]
        o2, o3 = eye[a] + eye[b], np.ones(3, int)
        tets.append(np.stack([corner(vid, o0), corner(vid, o1), corner(vid, o2), corner(vid, o3)], 1))
    T = np.concatenate(tets, 0)
    V = verts.astype(np.float64) * DX
    v6 = np.einsum("ij,ij->i", np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), V[T[:, 3]] - V[T[:, 0]])
    T[v6 < 0] = T[v6 < 0][:, [0, 2, 1, 3]]
    # barycentric embedding: sort local coordinates descending -> Kuhn tetrahedron [000, e_a, e_a + e_b, 111]
    u = x0 / DX - ijk
    order = np.argsort(-u, axis=1)
    us = np.take_along_axis(u, order, 1)
    w = np.stack([1 - us[:, 0], us[:, 0] - us[:, 1], us[:, 1] - us[:, 2], us[:, 2]], 1)
    cv = vid[inv]
    off = np.zeros((len(x0), 4, 3), int)
    for r in range(len(x0)):
        a, b = order[r, 0], order[r, 1]
        off[r, 1, a] = 1
        off[r, 2, a] = off[r, 2, b] = 1
        off[r, 3] = 1
    nid = np.stack([cv[np.arange(len(x0)), off[:, k, 0] * 4 + off[:, k, 1] * 2 + off[:, k, 2]] for k in range(4)], 1)
    emb_err = float(np.abs((w[..., None] * V[nid]).sum(1) - x0).max())
    vmesh = float(len(cells) * DX ** 3)
    info = {"n_cells": int(len(cells)), "n_vertices": int(len(V)), "n_tets": int(len(T)), "mesh_volume": vmesh,
            "particle_volume": float(np.sum(vol)), "volume_ratio": vmesh / float(np.sum(vol)),
            "largest_component_share": share, "n_components": int(nc), "embedding_error": emb_err}
    info["valid"] = bool(share >= 0.99 and emb_err < 1e-9)
    return V, T.astype(np.int64), (nid, w), info


def snapped_box(x0, point):
    """The drag box snapped to whole grid cells (cells whose centre is inside it): (centre, half-widths, mesh volume)."""
    lo = np.ceil((point - BOX) / DX - 0.5).astype(np.int64)
    hi = np.floor((point + BOX) / DX - 0.5).astype(np.int64)
    cells = np.unique(np.floor(x0 / DX).astype(np.int64), axis=0)
    n_in = int(((cells >= lo) & (cells <= hi)).all(1).sum())
    if n_in == 0:
        raise RuntimeError("no occupied cell inside the drag box")
    return (lo + hi + 1) * DX / 2, (hi + 1 - lo) * DX / 2, n_in * DX ** 3


def box_expr(centre, half, value):
    """value inside the box, 0 outside. PolyFEM's tinyexpr: if(c, a, b) = a when c > 0."""
    d = [f"abs({ax}-({c:.9g}))/{h:.9g}" for ax, c, h in zip("xyz", centre, half)]
    return f"if(1-max(max({d[0]},{d[1]}),{d[2]}), {value:.9g}, 0)"


def run_query(q, n_div, workdir, threads=8, gravity_only=False, tend_frames=47, keep=False):
    """S2 rollout of one query; returns (positions (len(frames), N, 3) at OUT_FRAMES within tend_frames, info)."""
    import meshio
    from scipy.spatial import cKDTree
    x0, vol = q["x0"], np.asarray(q["vol"], np.float64).reshape(-1)
    V, T, (nid, w), info = mesh_from_points(x0, vol)
    if not info["valid"]:
        return None, info
    wd = Path(workdir)
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    meshio.write(str(wd / "mesh.msh"), meshio.Mesh(V, [("tetra", T)]), file_format="gmsh22")
    rho = DENSITY * info["particle_volume"] / info["mesh_volume"]            # = 1000: the cells are the MPM body
    grav = gravity_only or int(q["gravity"]) == 1
    if grav:
        acc = [0.0, -9.81, 0.0]
        rhs = [-a for a in acc]
        a0 = acc
    else:
        m = np.asarray(q["drag_mask"]).reshape(-1).astype(bool)
        f = np.asarray(q["drag_force"], np.float64).reshape(-1)[:3]
        dp = np.asarray(q["drag_point"], np.float64).reshape(-1)[:3]
        centre, half, vbox = snapped_box(x0, dp)
        b = m.sum() * f / (rho * vbox)                      # body acceleration on the box: total force = n_masked * f
        rhs = [box_expr(centre, half, -bi) for bi in b]
        a0 = [box_expr(centre, half, bi) for bi in b]
        info.update(box_acc=b.tolist(), box_volume=vbox, box_volume_ratio=vbox / float(vol[m].sum()))
    dt = FRAME / n_div
    st = {"geometry": [{"mesh": str(wd / "mesh.msh")}],
          "materials": {"type": "FixedCorotational", "E": float(q["E"]), "nu": float(q["nu"]), "rho": float(rho)},
          "time": {"tend": tend_frames * FRAME, "dt": dt, "integrator": {"type": "ImplicitNewmark"}},
          "initial_conditions": {"acceleration": [{"id": 0, "value": a0}]},
          "boundary_conditions": {"rhs": rhs},
          "output": {"directory": str(wd), "paraview": {"file_name": "sim.vtu", "skip_frame": n_div, "volume": True,
                                                        "surface": False, "wireframe": False, "vismesh_rel_area": 1e10},
                     "advanced": {"save_time_sequence": True}, "log": {"level": "warning"}}}
    (wd / "run.json").write_text(json.dumps(st, indent=1))
    t = time.time()
    r = subprocess.run([str(BIN), "--json", str(wd / "run.json"), "--max_threads", str(threads)],
                       capture_output=True, text=True)
    info["seconds"] = time.time() - t
    info["returncode"] = r.returncode
    files = {int(p.stem.split("_")[-1]): p for p in wd.glob("step_*.vtu")}
    if r.returncode != 0 or not files:
        info["stderr"] = r.stderr[-2000:]
        raise RuntimeError(f"{q['qid']}: PolyFEM failed (rc {r.returncode}): {r.stderr[-400:]}")
    first = meshio.read(str(files[min(files)]))
    tree = cKDTree(first.points)
    dist, idx = tree.query(V)
    if dist.max() > 1e-6:
        raise RuntimeError(f"{q['qid']}: output points do not contain the mesh vertices (max {dist.max():.2e})")
    out, mcom = [], []
    tv = np.abs(np.einsum("ij,ij->i", np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), V[T[:, 3]] - V[T[:, 0]]))
    mesh_com = lambda P: (tv[:, None] * P[T].mean(1)).sum(0) / tv.sum()     # noqa: E731  (exact for linear tets)
    info["mesh_com0"] = mesh_com(V).tolist()
    for fr in [f for f in OUT_FRAMES if f <= tend_frames]:
        mf = meshio.read(str(files[fr * n_div]))              # files are named by time-step index
        vpos = mf.points[idx] + mf.point_data["solution"][idx]
        out.append((w[..., None] * vpos[nid]).sum(1))           # particles carried by their cell's tetrahedron
        mcom.append(mesh_com(vpos).tolist())
    info["mesh_com"] = mcom
    if not keep:
        shutil.rmtree(wd, ignore_errors=True)
    return np.stack(out, 0), info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", nargs="+", required=True, help="HDF5 query files")
    ap.add_argument("--n-div", type=int, required=True, help="dt = frame / n_div")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--work", default=os.path.expandvars("${BIG_ROOT}/tmp/p12_work"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.jsonl"
    done = {json.loads(line)["qid"] for line in rows.read_text().splitlines()} if rows.exists() else set()
    for path in args.queries:
        q = load_query(path)
        if q["qid"] in done:
            continue
        pos, info = run_query(q, args.n_div, Path(args.work) / q["qid"], args.threads)
        if pos is not None:
            np.savez_compressed(out / f"{q['qid']}.npz", s2=pos.astype(np.float32))
        append_jsonl({"qid": q["qid"], "path": str(path), "n_div": args.n_div, **info}, rows)
        print(f"[p12] {q['qid']}: {'ok' if pos is not None else 'excluded'} {info.get('seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
