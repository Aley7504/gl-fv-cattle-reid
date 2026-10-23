# -*- coding: utf-8 -*-
"""Export the H2025 per-query ranks so the primary numbers can be recomputed
from a file instead of by re-running the frozen evaluation program.

The frozen program returns the ranks in memory only (stage_j_scores.holdout_ranks),
so the release package could not be checked without running the code.  This
export writes one row per (regime, realization, train seed, query) with the rank of
the true identity for the global and the fused arm, and is validated by
recomputing the primary contrast and its bootstrap interval from the exported file.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\H2025_clean_restart")
WORK = Path(r"D:\CowReID_Paper\strengthening_20261023")
OUT = WORK / "results_raw" / "stage_h2025_per_query"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sj = load_module("stage_j_scores")
h2si = load_module("h2_statistical_inference")

REGIMES = ("g1", "g3", "full")
REALIZATION_SEEDS = list(sj.FROZEN["gallery_realization_seeds"])
TRAIN_SEEDS = (42, 43, 44)

b = sj._load_holdout_bundle()
q_paths = [str(p).replace("\\", "/") for p in b["q_paths"]]
q_ids = [str(x) for x in b["q_labels"]]

rows = []
for regime in REGIMES:
    for r_seed in REALIZATION_SEEDS:
        ranks = sj.holdout_ranks(regime, r_seed)
        for s in TRAIN_SEEDS:
            rg = ranks["A_seed%d" % s]
            rf = ranks["C_seed%d" % s]
            for i, (qp, qi) in enumerate(zip(q_paths, q_ids)):
                rows.append({"regime": regime, "realization": r_seed, "train_seed": s,
                             "query_path": qp, "query_identity": qi,
                             "rank_global": int(rg[i]), "rank_fusion": int(rf[i]),
                             "hit_global": int(rg[i] == 1), "hit_fusion": int(rf[i] == 1)})

out_csv = OUT / "h2025_per_query_ranks.csv"
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", out_csv, "rows", len(rows))

# ---- validation: recompute the primary contrast and its CI from the FILE ----
out = {"source": str(out_csv), "n_rows": len(rows), "regimes": {}}
for regime in REGIMES:
    packed = []
    for r_seed in REALIZATION_SEEDS:
        for s in TRAIN_SEEDS:
            sub = [r for r in rows if r["regime"] == regime and r["realization"] == r_seed
                   and r["train_seed"] == s]
            packed.append({
                "packed": {
                    "C_seed": h2si.pack_repeat([r["query_identity"] for r in sub],
                                               np.array([r["rank_fusion"] for r in sub]),
                                               b["identities"]),
                    "A_seed": h2si.pack_repeat([r["query_identity"] for r in sub],
                                               np.array([r["rank_global"] for r in sub]),
                                               b["identities"]),
                },
                "meta": {"regime": regime, "seed": s, "realization": r_seed}})
    rec = {}
    for metric in ("rank1_query_weighted", "macro_rank1_identity_balanced", "identity_level_map_mrr"):
        pt = h2si.delta_point_estimate(packed, metric, "C_seed", "A_seed")
        bt = h2si.identity_cluster_bootstrap(packed, metric, "C_seed", "A_seed")
        rec[metric] = {"delta": pt["delta_point_estimate_full_sample"],
                       "ci95": [bt["ci95_lower"], bt["ci95_upper"]],
                       "n_repeats": pt["n_repeats"],
                       "n_positive": pt["directional_consistency"]["n_positive"],
                       "n_negative": pt["directional_consistency"]["n_negative"]}
    out["regimes"][regime] = rec
    m = rec["rank1_query_weighted"]
    print("%-5s delta %+.4f pp  CI [%+.4f, %+.4f]  (n_pos %d / n_neg %d)" %
          (regime, 100 * m["delta"], 100 * m["ci95"][0], 100 * m["ci95"][1],
           m["n_positive"], m["n_negative"]))

json.dump(out, open(OUT / "h2025_per_query_validation.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("wrote", OUT / "h2025_per_query_validation.json")
