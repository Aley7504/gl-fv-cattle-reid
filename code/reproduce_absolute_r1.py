# -*- coding: utf-8 -*-
"""Recover ABSOLUTE Rank-1 / Macro-Rank-1 / MRR for the H2025 arms.

Read-only: rebuilds per-query ranks with the frozen evaluation program
(stage_j_scores.holdout_ranks) and applies the frozen metric functions.
No model, no feature, no split is modified; nothing is written to the project.
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
q_lab = np.array(b['q_labels']); ids = list(b['identities'])
real = SJ.FROZEN['gallery_realization_seeds']
seeds = SJ.FROZEN['mobilenet_seeds']
print(f'identities={len(ids)} queries={len(q_lab)} realizations={len(real)} seeds={list(seeds)}')

def metrics(rank):
    m = SJ.macro_rank1_identity_balanced(rank, q_lab, ids)
    return (SJ.rank1_query_weighted(rank), m['macro_rank1'], SJ.identity_level_map(rank))

out = {'protocol': {'regimes': ['g1', 'g3', 'full'], 'n_identities': len(ids),
                    'n_queries': int(len(q_lab)), 'realizations': len(real), 'seeds': list(seeds),
                    'source': 'stage_j_scores.holdout_ranks (frozen per-query ranks)',
                    'metric_fns': 'rank1_query_weighted / macro_rank1_identity_balanced / identity_level_map'},
       'regimes': {}}

for regime in ('g1', 'g3', 'full'):
    per_arm = {'A': [], 'B': [], 'C': {s: [] for s in seeds}}
    for r in real:
        ranks = SJ.holdout_ranks(regime, r)
        per_arm['B'].append(metrics(ranks['B']))
        for s in seeds:
            per_arm['A'].append(metrics(ranks['A_seed%d' % s]))
            per_arm['C'][s].append(metrics(ranks['C_seed%d' % s]))
    def agg(v):
        a = np.array(v, dtype=float)
        return {'rank1': float(a[:,0].mean()), 'macro_rank1': float(a[:,1].mean()),
                'mrr': float(a[:,2].mean()), 'n_conditions': int(len(a)),
                'rank1_sd': float(a[:,0].std(ddof=1))}
    # B is seed-independent: expand the 10 realization values to the 30-condition basis
    bl = [x for x in per_arm['B'] for _ in seeds]
    A = agg(per_arm['A']); C = agg([m for s in seeds for m in per_arm['C'][s]])
    Cper = {int(s): agg(per_arm['C'][s]) for s in seeds}
    B = agg(bl)
    out['regimes'][regime] = {
        'Global': A, 'Local': B, 'Fusion': C,
        'Fusion_per_seed': {str(k): v for k, v in Cper.items()},
        'delta_rank1_pp': round(100 * (C['rank1'] - A['rank1']), 4),
        'delta_macro_pp': round(100 * (C['macro_rank1'] - A['macro_rank1']), 4),
        'delta_mrr_pp': round(100 * (C['mrr'] - A['mrr']), 4),
    }
    g = out['regimes'][regime]
    print(f"\n[{regime}] n_conditions={A['n_conditions']}")
    print(f"  Global : R1 {A['rank1']:.4f}  Macro {A['macro_rank1']:.4f}  MRR {A['mrr']:.4f}")
    print(f"  Local  : R1 {B['rank1']:.4f}  Macro {B['macro_rank1']:.4f}  MRR {B['mrr']:.4f}")
    print(f"  Fusion : R1 {C['rank1']:.4f}  Macro {C['macro_rank1']:.4f}  MRR {C['mrr']:.4f}")
    print(f"  Delta  : R1 {g['delta_rank1_pp']:+.2f} pp | Macro {g['delta_macro_pp']:+.2f} | MRR {g['delta_mrr_pp']:+.2f}")

dst = Path(r'D:\CowReID_Paper\analysis\absolute_r1_from_frozen.json')
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nWROTE', dst)
print('CHECK g1 delta vs frozen +5.8923 pp ->', out['regimes']['g1']['delta_rank1_pp'])
