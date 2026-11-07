"""EXPLORATORY (Oct 14, 2026, after the Phase 4k verdict): gated V2 on the Phase 2c queries,
including OOD, from the 4h + 4i rows. Development queries for V2, so not a test. CPU only.
Run from the project root: python code/explore_gated_2c.py"""
import sys, json, glob, numpy as np
sys.path.insert(0, 'code')
from pathlib import Path
from common import load_config, read_jsonl, results_dir
from p2b_generate import gen_dir
from p2c_select import starts_for
from p2c_summarize import query_inputs
from p4a_certificate import ood_index_from
from p4d_pipeline import gen_costs
cfg = load_config('configs/p4i_state.yaml'); spec = cfg['cert_spec']; tau = 0.02614; eps = 0.1
g, c_ref = gen_costs(); cg = g['base']['best']; share = lambda s: c_ref * s / 20016.
h4 = [r for f in sorted(results_dir('p4h_suffix').glob('shard_*/suffix.jsonl')) for r in read_jsonl(f)]
v = {r['qid']: r for f in sorted(results_dir('p4i_state').glob('shard_*/state.jsonl')) for r in read_jsonl(f)}
scan = {r['qid']: r for f in sorted(results_dir('p2c_scan').glob('shard_*/scan.jsonl')) for r in read_jsonl(f)
        if r['source'] == 'base' and r['fidelity'] == 'coarse_cfl' and int(r['m']) == 4}
d4 = {r['qid']: r for f in sorted(results_dir('p4d_fullsim').glob('shard_*/fullsim.jsonl')) for r in read_jsonl(f)}
rates = [r['rollout_s'] / r['substeps'] for r in d4.values() if r.get('stable') and r.get('substeps') and r['split'].startswith('id_new_')]
rate = float(np.median(rates)); oi = ood_index_from(['p2_build_ood']); root, groot = Path(cfg['paths']['data_root']), gen_dir(cfg, False)
chain = [0, 7, 15, 23, 31, 39]
out = {}
for r in h4:
    sr = scan[r['qid']]; q = query_inputs(r['qid'], r['split'], root, oi)
    st = starts_for(spec['placement'], np.load(groot / 'base' / r['split'] / f"{r['qid']}.npz")['samples'], q, 4)
    win = {s: (np.inf if (b or c or raw is None) else raw) for s, raw, b, c in zip(sr['starts'], sr['raw'], sr['blown'], sr['crashed'])}
    sec = {s: (t or 0.0) for s, t in zip(sr['starts'], sr['seconds'])}
    acc = max(win[s] for s in st) < tau; eg = float(np.mean(r['gen_err_t'])); cp = cg + sum(sec[s] for s in st)
    ec, cc = (eg, cp) if acc else (0.0, cp + c_ref)
    if acc: eG, cG = eg, cp
    else:
        j = r['j']; last = 5 if j is None else j
        pg = sum(sec[s] for s in chain[:last + 1] if s not in st)
        if r['mode'] == 'suffix':
            x = v[r['qid']]; eG = float(np.mean(x['v2_err_t'])); cG = cp + pg + rate * x['v2_shadow_substeps'] + share(x['v2_substeps'])
        else: eG, cG = 0.0, cp + pg + c_ref
    d = d4.get(r['qid']); eC = float(np.mean(d['err_t'])) if d and d.get('err_t') else np.inf; cC = d.get('rollout_s', np.nan) if d else np.nan
    out.setdefault(r['split'], []).append((ec, cc, eG, cG, eC, cC, eg))
print('EXPLORATORY gated V2 on Phase 2c (tau 0.02614); cost / c_ref')
print('split            n   cert err/risk/cost      gated err/risk/cost     coarse-only err/risk/cost   gen risk')
allr = []
for sp, L in sorted(out.items()):
    A = np.array(L); allr += L
    f = lambda e, c: f"{np.mean(np.minimum(A[:, e], 10)):.4f}/{np.mean(A[:, e] > eps):.2f}/{np.nanmean(A[:, c]) / c_ref:.2f}"
    print(f"{sp:16s} {len(L):3d}  {f(0,1):22s}  {f(2,3):22s}  {f(4,5):22s}  {np.mean(A[:, 6] > eps):.2f}")
A = np.array(allr); f = lambda e, c: f"{np.mean(np.minimum(A[:, e], 10)):.4f}/{np.mean(A[:, e] > eps):.3f}/{np.nanmean(A[:, c]) / c_ref:.2f}"
print(f"{'all':16s} {len(A):3d}  {f(0,1):22s}  {f(2,3):22s}  {f(4,5):22s}  {np.mean(A[:, 6] > eps):.3f}")
