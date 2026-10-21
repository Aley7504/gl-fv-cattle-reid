# -*- coding: utf-8 -*-
"""Absolute Rank-1 for Global / Local / Fusion with identity-cluster bootstrap CIs
and per-seed point estimates. Read-only: frozen ranks, frozen metric functions.
"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(r'D:\H2025_clean_restart')
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location("h2025_stage_j_scores", ROOT / 'scripts' / 'stage_j_scores.py')
SJ = importlib.util.module_from_spec(spec); sys.modules['h2025_stage_j_scores'] = SJ
spec.loader.exec_module(SJ)

b = SJ._load_holdout_bundle()
ids = list(b['identities']); q = np.array(b['q_labels'])
real = SJ.FROZEN['gallery_realization_seeds']; seeds = list(SJ.FROZEN['mobilenet_seeds'])
B_BOOT = 10000
SEED = 20260917          # frozen BOOTSTRAP_SEED used by the H2 inference
rng = np.random.default_rng(SEED)

# query index groups per identity (for cluster bootstrap)
groups = {c: np.where(q == c)[0] for c in ids}
print(f'identities={len(ids)} queries={len(q)} boot={B_BOOT} seed={SEED}')

out = {'bootstrap': {'unit': 'identity cluster', 'n_clusters': len(ids), 'n_boot': B_BOOT, 'seed': SEED,
                     'note': 'CI of the ABSOLUTE Rank-1 of each condition, resampling the 16 holdout identities'},
       'regimes': {}}

for regime in ('g1', 'g3', 'full'):
    A = {s: [] for s in seeds}; C = {s: [] for s in seeds}; Bl = []
    for rs in real:
        r = SJ.holdout_ranks(regime, rs)
        Bl.append(r['B'])
        for s in seeds:
            A[s].append(r['A_seed%d' % s]); C[s].append(r['C_seed%d' % s])
    condA = [x for s in seeds for x in A[s]]      # 30
    condC = [x for s in seeds for x in C[s]]      # 30
    condB = Bl                                     # 10 (seed-independent)

    def pt(conds):
        return float(np.mean([np.mean(c == 1) for c in conds]))
    pA, pC, pB = pt(condA), pt(condC), pt(condB)
    per_seed = {'Global': {int(s): round(pt(A[s]), 4) for s in seeds},
                'Fusion': {int(s): round(pt(C[s]), 4) for s in seeds}}

    bA, bC, bB = [], [], []
    for _ in range(B_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        bA.append(np.mean([np.mean(c[idx] == 1) for c in condA]))
        bC.append(np.mean([np.mean(c[idx] == 1) for c in condC]))
        bB.append(np.mean([np.mean(c[idx] == 1) for c in condB]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    out['regimes'][regime] = {
        'Global': {'rank1': round(pA, 4), 'ci95': ci(bA), 'per_seed': per_seed['Global'], 'n_conditions': len(condA)},
        'Local':  {'rank1': round(pB, 4), 'ci95': ci(bB), 'per_seed': 'seed-independent', 'n_conditions': len(condB)},
        'Fusion': {'rank1': round(pC, 4), 'ci95': ci(bC), 'per_seed': per_seed['Fusion'], 'n_conditions': len(condC)},
    }
    print(f"\n[{regime}]")
    for k in ('Global', 'Local', 'Fusion'):
        v = out['regimes'][regime][k]
        print(f"  {k:7} R1 {v['rank1']:.4f}  CI {v['ci95']}  per-seed {v['per_seed']}")

dst = Path(r'D:\CowReID_Paper\analysis\absolute_r1_bootstrap_ci.json')
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nWROTE', dst)
