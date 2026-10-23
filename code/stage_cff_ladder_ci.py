# -*- coding: utf-8 -*-
"""CFF coverage ladder with the bootstrap that the ladder plan declared but never ran.

The plan (CFF_COVERAGE_LADDER_PLAN.json) specifies an identity-cluster bootstrap
(B = 10,000, seed 20260922) but the original ladder script never executed it, so
the published rungs carry point estimates only.  This script recomputes the rungs
from the same frozen inputs and adds:

  (a) a pooled identity-cluster bootstrap over all 256 outer-fold identity
      clusters - the same estimator the paper's primary CFF contrast uses, so the
      intervals are directly comparable;
  (b) the per-fold within-fold intervals, for transparency.

Validation: the k = 3 rung must reproduce the sealed primary contrast
(+3.74 pp pooled, +3.75 pp fold-mean).
"""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\H2025_clean_restart")
OUT = Path(r"D:\CowReID_Paper\strengthening_20261023\results_raw\stage_cff_ladder_ci")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)
B = 10000
BOOT_SEED = 20260922


def cos(A, Bm):
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bm = Bm / np.maximum(np.linalg.norm(Bm, axis=1, keepdims=True), 1e-12)
    return A @ Bm.T


def identity_scores(S, gid_list, ids):
    """Identity-level max aggregation; returns (n_query, n_identity) with -inf for empty."""
    M = np.full((S.shape[0], len(ids)), -np.inf)
    for b in range(S.shape[1]):
        col = gid_list[b]
        M[:, col] = np.maximum(M[:, col], S[:, b])
    return M


def hits_and_rr(M, q_ids, ids):
    """Optimistic tie rule: a query is a hit when the true identity is within 1e-12 of the max."""
    r1 = np.zeros(len(q_ids), dtype=np.float64)
    rr = np.zeros(len(q_ids), dtype=np.float64)
    for k, t in enumerate(q_ids):
        s = M[k]
        if t not in ids:
            continue
        st = s[ids.index(t)]
        r1[k] = 1.0 if st >= s.max() - 1e-12 else 0.0
        rr[k] = 1.0 / (1 + int(np.sum(s > st + 1e-12)))
    return r1, rr


def main():
    # query path sets per outer fold, from the sealed P3-C per-query outcomes
    qset = {}
    with open(ROOT / "analysis" / "stage_p3c" / "CFF_EXTERNAL_PRIMARY_PER_QUERY_OUTCOMES.csv",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qset.setdefault(int(row["fold"]), set()).add(row["query_path"].replace("\\", "/"))

    folds = []
    for f in range(5):
        E = {s: np.load(ROOT / f"analysis/stage_p3b/embeddings/fold{f}_s{s}_outer_eval.npz",
                        allow_pickle=True) for s in SEEDS}
        Fv = np.load(ROOT / f"analysis/stage_p3b/fv/fold{f}_outer_eval.npz", allow_pickle=True)
        rel = np.array([r.replace("\\", "/") for r in E[42]["rel"]])
        canon = np.array(E[42]["canonical"])
        keep = np.array([i for i, r in enumerate(rel) if r not in qset.get(f, set())])
        qidx = np.array([i for i, r in enumerate(rel) if r in qset.get(f, set())])
        ids = sorted(set(canon[keep]))
        pos = {c: i for i, c in enumerate(ids)}
        gal = {}
        for i in keep:
            gal.setdefault(canon[i], []).append(i)
        if {len(v) for v in gal.values()} != {3}:
            print("fold%d: gallery sizes not 3 -> skipped" % f)
            continue
        q_ids = [canon[i] for i in qidx]
        per_k = {}
        for k in (1, 2, 3):
            recs = []
            for sub in itertools.combinations(range(3), k):
                gidx = [gal[c][j] for c in ids for j in sub]
                gid_list = [pos[canon[i]] for i in gidx]
                for s in SEEDS:
                    SG = cos(E[s]["emb"][qidx], E[s]["emb"][gidx])
                    SL = cos(Fv["fv"][qidx], Fv["fv"][gidx])
                    out = {}
                    for tag, S in (("Global", SG), ("Fusion", 0.5 * SG + 0.5 * SL)):
                        M = identity_scores(S, gid_list, ids)
                        r1, rr = hits_and_rr(M, q_ids, ids)
                        out[tag] = (r1, rr)
                    out["Local"] = hits_and_rr(identity_scores(SL, gid_list, ids), q_ids, ids)
                    recs.append(out)
            per_k[k] = recs
        folds.append({"fold": f, "ids": ids, "q_ids": q_ids, "per_k": per_k,
                      "n_query": len(q_ids), "n_identity": len(ids)})
        print("fold%d: %d queries, %d identities" % (f, len(q_ids), len(ids)), flush=True)

    # ---------------- point estimates ----------------
    summary = {"protocol": {
        "inputs": "frozen CFF embeddings (stage_p3b) and frozen per-fold Fisher vectors",
        "rungs": "exhaustive enumeration of all C(3,k) gallery subsets; no random draw",
        "fusion": "alpha = 0.5, identity-level max, optimistic tie rule",
        "bootstrap": "identity cluster, B = %d, seed = %d; pooled over the outer folds (256 clusters) "
                     "and reported per fold as well" % (B, BOOT_SEED),
    }, "rungs": {}}

    for k in (1, 2, 3):
        # per (fold, subset, seed) -> per-query hit vectors, averaged over subsets and seeds
        pooled = {"Global": [], "Fusion": [], "Local": []}
        cluster_of_query = []      # global cluster id per pooled query row
        rr_pooled = {"Global": [], "Fusion": [], "Local": []}
        per_fold_delta = []
        for fd in folds:
            recs = fd["per_k"][k]
            n = len(recs[0]["Global"][0])
            for tag in ("Global", "Fusion", "Local"):
                r1 = np.mean([r[tag][0] for r in recs], axis=0)
                rr = np.mean([r[tag][1] for r in recs], axis=0)
                pooled[tag].append(r1)
                rr_pooled[tag].append(rr)
            per_fold_delta.append(float(np.mean(pooled["Fusion"][-1]) - np.mean(pooled["Global"][-1])))
            cluster_of_query.extend([(fd["fold"], c) for c in fd["q_ids"]])
        cat = {t: np.concatenate(pooled[t]) for t in pooled}
        cat_rr = {t: np.concatenate(rr_pooled[t]) for t in rr_pooled}
        clusters = sorted(set(cluster_of_query))
        cindex = {c: i for i, c in enumerate(clusters)}
        slot = np.array([cindex[c] for c in cluster_of_query])
        n_c = len(clusters)

        rng = np.random.RandomState(BOOT_SEED)
        deltas = np.empty(B)
        for b in range(B):
            pick = rng.randint(0, n_c, size=n_c)
            m = np.zeros(len(slot), dtype=bool)
            sel = pick
            # count multiplicity of each cluster in the draw
            mult = np.bincount(sel, minlength=n_c).astype(np.float64)
            w = mult[slot]
            denom = w.sum()
            deltas[b] = (np.sum(w * cat["Fusion"]) - np.sum(w * cat["Global"])) / denom
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        # per-fold intervals
        fold_ci = []
        for fd in folds:
            recs = fd["per_k"][k]
            r1f = np.mean([r["Fusion"][0] for r in recs], axis=0)
            r1g = np.mean([r["Global"][0] for r in recs], axis=0)
            qids = fd["q_ids"]
            cs = sorted(set(qids))
            ci = {c: i for i, c in enumerate(cs)}
            sl = np.array([ci[c] for c in qids])
            nc = len(cs)
            rg = np.random.RandomState(BOOT_SEED + fd["fold"])
            d = np.empty(B)
            for b in range(B):
                pick = rg.randint(0, nc, size=nc)
                mult = np.bincount(pick, minlength=nc).astype(np.float64)
                w = mult[sl]
                d[b] = (np.sum(w * r1f) - np.sum(w * r1g)) / w.sum()
            flo, fhi = np.percentile(d, [2.5, 97.5])
            fold_ci.append({"fold": fd["fold"], "delta_pp": round(100 * float(r1f.mean() - r1g.mean()), 4),
                            "ci95_pp": [round(100 * float(flo), 4), round(100 * float(fhi), 4)]})

        summary["rungs"]["k=%d" % k] = {
            "global_rank1": round(float(cat["Global"].mean()), 6),
            "local_rank1": round(float(cat["Local"].mean()), 6),
            "fusion_rank1": round(float(cat["Fusion"].mean()), 6),
            "delta_pp_pooled": round(100 * float(cat["Fusion"].mean() - cat["Global"].mean()), 4),
            "delta_ci95_pp_pooled": [round(100 * float(lo), 4), round(100 * float(hi), 4)],
            "delta_pp_fold_mean": round(100 * float(np.mean(per_fold_delta)), 4),
            "delta_pp_fold_sd": round(100 * float(np.std(per_fold_delta, ddof=1)), 4),
            "per_fold": fold_ci,
            "n_identity_clusters_pooled": n_c,
            "n_query_pooled": int(len(slot)),
        }
        r = summary["rungs"]["k=%d" % k]
        print("k=%d  global %.4f  local %.4f  fusion %.4f  delta %+.2f pp  CI %s (pooled) | "
              "fold-mean %+.2f pp (sd %.2f)" %
              (k, r["global_rank1"], r["local_rank1"], r["fusion_rank1"], r["delta_pp_pooled"],
               r["delta_ci95_pp_pooled"], r["delta_pp_fold_mean"], r["delta_pp_fold_sd"]), flush=True)

    json.dump(summary, open(OUT / "cff_ladder_ci.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", OUT / "cff_ladder_ci.json")


if __name__ == "__main__":
    main()
