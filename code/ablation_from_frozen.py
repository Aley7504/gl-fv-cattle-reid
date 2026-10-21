# -*- coding: utf-8 -*-
"""Post-hoc ablations computed ONLY from frozen artefacts (no training, no refitting).

Q1 FV necessity        : FV (frozen codebook) vs mean-pooled HardNet (128-D) vs PCA32-mean-pool (32-D)
Q2 Local discriminability: Local-only Rank-1 vs chance (1 / n_identities)
Q3 Alpha sensitivity   : fusion score = a*sG + (1-a)*sL, a in [0,1] step 0.05
Q4 Backbone dependence : ConvNeXt-Tiny arms (A_p_seed, C_p_seed) at g1
Q5 Coverage stability  : g1 / g3 / full, spread across the 10 gallery realizations
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
ids = list(b['identities']); g_lab = np.array(b['g_labels']); q_lab = np.array(b['q_labels'])
real = SJ.FROZEN['gallery_realization_seeds']; seeds = list(SJ.FROZEN['mobilenet_seeds'])
valid_q, valid_g = b['valid_q'], b['valid_g']

# ---------- local encodings from frozen Top64 HardNet descriptors ----------
F = ROOT / 'features' / 'local'
zq = np.load(F / 'holdout_query_descriptors.npz', allow_pickle=True)
zg = np.load(F / 'holdout_gallery_descriptors.npz', allow_pickle=True)
dq = [np.asarray(x, dtype=np.float64) for x in zq['descriptors']]
dg = [np.asarray(x, dtype=np.float64) for x in zg['descriptors']]

cb = np.load(ROOT / 'codebooks' / 'local_fv_pca32_gmm16_nowhiten_pool256_shared.npz', allow_pickle=False)
CENTER, COMP = cb['center'], cb['components']

def pool(descs, pca=False):
    out = np.zeros((len(descs), 32 if pca else 128), dtype=np.float64)
    for i, d in enumerate(descs):
        if d.size == 0:
            continue
        z = (d - CENTER) @ COMP.T if pca else d
        v = z.mean(axis=0)
        n = np.linalg.norm(v)
        out[i] = v / n if n > 0 else 0.0
    return out

mp_q, mp_g = pool(dq), pool(dg)
pp_q, pp_g = pool(dq, pca=True), pool(dg, pca=True)

def cos_safe(A, B):
    an = np.linalg.norm(A, axis=1, keepdims=True); bn = np.linalg.norm(B, axis=1, keepdims=True)
    return (A / np.maximum(an, 1e-12)) @ (B / np.maximum(bn, 1e-12)).T

def metrics_from_scores(S, glab):
    S_qc = SJ.identity_max(S, glab, ids)
    r = SJ.rank_of_true(S_qc, q_lab, ids)
    m = SJ.macro_rank1_identity_balanced(r, q_lab, ids)
    return SJ.rank1_query_weighted(r), m['macro_rank1'], SJ.identity_level_map(r)

def fuse(sg, sl, a, vg):
    both = np.asarray(valid_q, bool).reshape(-1, 1) & np.asarray(vg, bool).reshape(1, -1)
    return np.where(both, a * sg + (1 - a) * sl, sg)

out = {'caveat': ('Post-hoc exploratory ablations computed from frozen artefacts only: no model, feature, '
                  'split or realization was altered, and no parameter was re-selected for the main claim.'),
       'n_identities': len(ids), 'n_queries': int(len(q_lab)),
       'chance_rank1': round(1.0 / len(ids), 4), 'regimes': {}}

for regime in ('g1', 'g3', 'full'):
    A, B, C = [], [], []
    MP, PP = [], []
    CN_A, CN_C = [], []
    sweep = {round(a, 2): [] for a in np.arange(0, 1.0001, 0.05)}
    for rs in real:
        arms, glab, sel = SJ.holdout_arm_scores(regime, rs)
        vg = np.asarray(valid_g)[sel]
        sg_by = {s: arms['A_seed%d' % s] for s in seeds}
        sl = arms['B']
        S_mp = cos_safe(mp_q, mp_g[sel]); S_pp = cos_safe(pp_q, pp_g[sel])
        for s in seeds:
            sg = sg_by[s]
            A.append(metrics_from_scores(sg, glab))
            C.append(metrics_from_scores(arms['C_seed%d' % s], glab))
            CN_A.append(metrics_from_scores(arms['A_p_seed%d' % s], glab))
            CN_C.append(metrics_from_scores(arms['C_p_seed%d' % s], glab))
            for a, acc in sweep.items():
                acc.append(metrics_from_scores(fuse(sg, sl, a, vg), glab)[0])
        B.append(metrics_from_scores(sl, glab))
        MP.append(metrics_from_scores(S_mp, glab))
        PP.append(metrics_from_scores(S_pp, glab))

    def agg(v):
        a = np.array(v, float)
        return {'rank1': round(float(a[:,0].mean()), 4), 'macro_rank1': round(float(a[:,1].mean()), 4),
                'mrr': round(float(a[:,2].mean()), 4), 'rank1_sd': round(float(a[:,0].std(ddof=1)), 4),
                'n': int(len(a))}
    reg = {
        'naming': 'A=Global-only, B=Local-only (Fisher Vector), C=Fusion(alpha=0.5)',
        'Global': agg(A), 'Local_FV': agg(B), 'Fusion': agg(C),
        'Local_MeanPool128': agg(MP), 'Local_PCA32_MeanPool': agg(PP),
        'ConvNeXt_Global': agg(CN_A), 'ConvNeXt_Fusion': agg(CN_C),
        'alpha_sweep_rank1': {str(k): round(float(np.mean(v)), 4) for k, v in sweep.items()},
    }
    out['regimes'][regime] = reg
    print(f"\n=== {regime} (chance {out['chance_rank1']}) ===")
    for k in ('Global','Local_FV','Local_MeanPool128','Local_PCA32_MeanPool','Fusion','ConvNeXt_Global','ConvNeXt_Fusion'):
        v = reg[k]; print(f"  {k:22} R1 {v['rank1']:.4f} (sd {v['rank1_sd']:.4f})  Macro {v['macro_rank1']:.4f}  MRR {v['mrr']:.4f}")
    best = max(reg['alpha_sweep_rank1'].items(), key=lambda x: x[1])
    print(f"  alpha sweep: R1@0.5={reg['alpha_sweep_rank1']['0.5']:.4f} | best a={best[0]} ({best[1]:.4f}) | "
          f"a=0 ({reg['alpha_sweep_rank1']['0.0']:.4f}) a=1 ({reg['alpha_sweep_rank1']['1.0']:.4f})")

dst = Path(r'D:\CowReID_Paper\analysis\ablation_from_frozen.json')
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nWROTE', dst)
