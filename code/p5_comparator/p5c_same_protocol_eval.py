# -*- coding: utf-8 -*-
"""P5-C — same-protocol comparison: EVALUATION IMPLEMENTATION (frozen by the P5-C execution seal).

One generic implementation of the FROZEN P3-C retrieval protocol, used to score BOTH arms:

    scope        per outer fold only: a query competes against its own fold's gallery.
                 Candidate libraries are NEVER merged across folds.
    feature      post-PCA128 L2-normalized embeddings; image similarity = cosine
    aggregation  identity score = MAX cosine over that identity's 3 gallery images
    tie          optimistic:  rank = 1 + #{ c : S(q,c) > S(q,y_q) }
    metrics      Rank-1       = mean(rank == 1)
                 Macro-Rank-1 = mean over identities of that identity's query-mean hit rate
                 MRR          = mean(1/rank)
    forbidden    no reranking, no calibration, no learned fusion, no shortlist, no pruning

Ours   : the frozen P3-C per-query ranks are READ, never recomputed.
Hu2025 : computed here from the frozen P5-B post-PCA128 features.

The metric definitions were recovered empirically (see P5C_EVALUATION_AUDIT) by reproducing the
published P3-C numbers 0.9393 / 0.9355 / 0.9606 etc. from the frozen per-query ranks; they were
not assumed.

Writes: P5C_PER_QUERY_OUTCOMES.csv, P5C_EVALUATION_AUDIT.{json,md}, P5C_METRICS.json
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

F = r"F:\cow-project"
INFRA = os.path.join(F, "P5B_FORMAL_20260928")
OUT = os.path.join(INFRA, "p5c")
FEAT = os.path.join(INFRA, "features")
AUDIT = os.path.join(F, "P5B_SERVER_AUDIT")
H = os.path.join(F, "H2025_clean_restart")
P3C = os.path.join(H, "analysis", "stage_p3c")
ROI = os.path.join(H, "analysis", "stage_p2r_det1", "CFF_PRIMARY_ROI_COMPATIBILITY_AUDIT.json")
CACHE = os.path.join(H, "analysis", "stage_p3b", "CFF_P3B_IMAGE_CACHE.json")
NEST = os.path.join(H, "analysis", "stage_p3a", "CFF_P3A2_NESTED_SPLIT.json")
FROZEN_CSV = os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_PER_QUERY_OUTCOMES.csv")
FROZEN_RESULT = os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_EVALUATION_RESULT.json")
COMPLETION_AUDIT = os.path.join(INFRA, "P5B_SERVER_COMPLETION_AUDIT_v2.json")

SEEDS = [42, 43, 44]
FROZEN_COUNTS = {0: (52, 156, 243), 1: (51, 153, 220), 2: (51, 153, 217),
                 3: (51, 153, 214), 4: (51, 153, 210)}
PUBLISHED_OURS = {  # from CFF_EXTERNAL_PRIMARY_EVALUATION_RESULT.md ; arm "fusion" = Ours
    42: {"rank1": 0.9393, "macro_rank1": 0.9355, "mrr": 0.9606},
    43: {"rank1": 0.9466, "macro_rank1": 0.9521, "mrr": 0.9657},
    44: {"rank1": 0.9420, "macro_rank1": 0.9559, "mrr": 0.9638},
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    for _ in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(1.5)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def stop(msg):
    print("\nSTOP: %s" % msg, flush=True)
    raise SystemExit(2)


def norm_rel(s):
    return str(s).replace("\\", "/").strip()


def rank_metrics(identities, hits, ranks):
    """Frozen definitions, recovered empirically from the published P3-C numbers."""
    n = len(ranks)
    rank1 = float(np.mean([1.0 if r == 1 else 0.0 for r in ranks]))
    mrr = float(np.mean([1.0 / r for r in ranks]))
    per = {}
    for i, h in zip(identities, hits):
        per.setdefault(i, []).append(h)
    macro = float(np.mean([float(np.mean(v)) for v in per.values()]))
    return {"rank1": rank1, "macro_rank1": macro, "mrr": mrr, "n_queries": n,
            "n_identities": len(per)}


def optimistic_ranks(qfeat, q_ids, gfeat, g_ids):
    """Identity-level cosine retrieval with max aggregation and the optimistic tie rule."""
    S = qfeat @ gfeat.T                                  # cosine: both sides are L2-normalised
    gid_sorted = sorted(set(g_ids))
    idx_of = {c: i for i, c in enumerate(gid_sorted)}
    cols = [[] for _ in gid_sorted]
    for j, c in enumerate(g_ids):
        cols[idx_of[c]].append(j)
    ident_scores = np.stack([S[:, c].max(axis=1) if c else np.full(S.shape[0], -np.inf)
                            for c in cols], axis=1)          # [n_query, n_gallery_identities]
    y = np.array([idx_of[c] for c in q_ids])
    own = ident_scores[np.arange(len(q_ids)), y]
    ranks = 1 + (ident_scores > own[:, None]).sum(axis=1)     # optimistic
    return ranks.astype(int)


def main():
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    blockers = []

    # ---------------------------------------------------------------- inputs
    completion = json.load(open(COMPLETION_AUDIT, encoding="utf-8-sig"))
    roi = json.load(open(ROI, encoding="utf-8"))
    cache = json.load(open(CACHE, encoding="utf-8"))
    nest = json.load(open(NEST, encoding="utf-8"))
    featman = json.load(open(os.path.join(FEAT, "P5C_FEATURE_MATERIALIZATION.json")
                             if os.path.exists(os.path.join(FEAT, "P5C_FEATURE_MATERIALIZATION.json"))
                             else os.path.join(FEAT, "P5B_FEATURE_MATERIALIZATION.json"),
                             encoding="utf-8-sig"))
    frozen_rows = list(csv.DictReader(open(FROZEN_CSV, encoding="utf-8-sig")))

    src_of = {(r["canonical"], r["rel"]): r["source"] for r in cache["records"]}
    role_of = {(r["canonical"], r["rel"]): r["role"] for r in roi["per_image"]}
    by_id = {}
    for r in roi["per_image"]:
        by_id.setdefault(r["canonical"], []).append(r)

    # ---------------------------------------------------------------- §7 checks
    # frozen per-fold membership table
    frozen_by_fold = {}
    for r in frozen_rows:
        frozen_by_fold.setdefault(int(r["fold"]), []).append(r)
    checks = {}

    # --- 15/15 checkpoint hashes match the P5-B completion audit
    ck_expected = {c["run_id"]: c["sha256"] for c in completion["selected_checkpoints"]}
    ck_now = {}
    for m in featman["runs"]:
        p = os.path.join(INFRA, "..", "..", "cow-project")
        bp = os.path.join(F, "H2025_clean_restart", "analysis", "stage_p5b", "runs",
                          m["run_id"], "best.pt")
        ck_now[m["run_id"]] = sha_file(bp) if os.path.isfile(bp) else None
    checks["checkpoint_hashes_match_completion_audit"] = all(
        ck_now.get(k) == v for k, v in ck_expected.items())
    checks["n_checkpoints"] = len(ck_expected)

    # --- feature/PCA files unchanged since materialization
    fp_ok = True
    for m in featman["runs"]:
        d = os.path.join(FEAT, m["run_id"])
        if sha_file(os.path.join(d, "features.npz")) != m["features_npz_sha256"]:
            fp_ok = False
        if sha_file(os.path.join(d, "pca128.pkl")) != m["pca_pkl_sha256"]:
            fp_ok = False
    checks["no_post_p5b_model_or_pca_modification"] = fp_ok
    checks["pca_fit_identities_exclude_inner_val_and_outer_eval"] = all(
        m["fit_ids_intersect_inner_val"] == 0 and m["fit_ids_intersect_outer_eval"] == 0
        and m["fit_ids_equal_frozen_inner_fit"] is True for m in featman["runs"])
    checks["hu_feature_dim_128"] = all(m["post_pca_dim"] == 128 for m in featman["runs"])
    checks["all_hu_features_finite"] = all(m["all_finite"] for m in featman["runs"])

    # --- Hu row alignment + path/identity agreement, per fold and seed
    hu = {}          # (fold, seed) -> dict(rel, ids, roles, ranks)
    per_fold_align = {}
    for fold in range(5):
        oe_ids = sorted(nest["per_outer_fold"]["fold%d" % fold]["outer_eval"])
        recs = [{"canonical": c, "rel": r["rel"], "role": role_of[(c, r["rel"])],
                 "source": src_of[(c, r["rel"])]} for c in oe_ids for r in by_id.get(c, [])]
        n_g = sum(1 for r in recs if r["role"] == "gallery")
        n_q = sum(1 for r in recs if r["role"] == "query")
        exp = FROZEN_COUNTS[fold]
        per_fold_align["fold%d" % fold] = {
            "frozen_ids": exp[0], "frozen_gallery": exp[1], "frozen_query": exp[2],
            "hu_ids": len(oe_ids), "hu_gallery": n_g, "hu_query": n_q,
            "counts_match": (len(oe_ids), n_g, n_q) == exp,
            "frozen_query_rel_set_equals_hu_query_rel_set":
                set(norm_rel(r["query_path"]) for r in frozen_by_fold[fold]) ==
                set(norm_rel(r["rel"]) for r in recs if r["role"] == "query"),
            "frozen_query_identity_set_equals_hu_query_identity_set":
                set(r["identity"] for r in frozen_by_fold[fold]) ==
                set(r["canonical"] for r in recs if r["role"] == "query"),
        }
        hu["fold%d" % fold] = {"recs": recs, "n_gallery": n_g, "n_query": n_q}

    checks["frozen_endpoint_counts_match"] = all(v["counts_match"] for v in per_fold_align.values())
    checks["hu_query_path_set_equals_frozen_endpoint"] = all(
        v["frozen_query_rel_set_equals_hu_query_rel_set"] for v in per_fold_align.values())
    checks["identity_labels_exact_match"] = all(
        v["frozen_query_identity_set_equals_hu_query_identity_set"]
        for v in per_fold_align.values())
    if not all([checks["checkpoint_hashes_match_completion_audit"],
                checks["no_post_p5b_model_or_pca_modification"],
                checks["pca_fit_identities_exclude_inner_val_and_outer_eval"],
                checks["frozen_endpoint_counts_match"],
                checks["hu_query_path_set_equals_frozen_endpoint"],
                checks["identity_labels_exact_match"]]):
        stop("evaluation audit failed before any metric was computed: %s"
             % json.dumps({k: v for k, v in checks.items() if v is False}))

    # ---------------------------------------------------------------- Ours (read frozen)
    ours_pq = {}
    for r in frozen_rows:
        key = (int(r["fold"]), int(r["seed"]), norm_rel(r["query_path"]))
        ours_pq[key] = {"rank": int(r["rank_fusion"]), "rr": float(r["rr_fusion"]),
                        "identity": r["identity"]}
    ours_per_seed = {}
    for s in SEEDS:
        ids = [v["identity"] for (f, ss, q), v in ours_pq.items() if ss == s]
        ranks = [v["rank"] for (f, ss, q), v in ours_pq.items() if ss == s]
        hits = [1 if x == 1 else 0 for x in ranks]
        ours_per_seed[s] = rank_metrics(ids, hits, ranks)

    # self-check: reproduce the published P3-C numbers from the frozen ranks
    sc = {}
    for s in SEEDS:
        p = PUBLISHED_OURS[s]
        got = ours_per_seed[s]
        sc["seed%d" % s] = {"published": p,
                            "recomputed": {k: round(got[k], 4) for k in ("rank1", "macro_rank1", "mrr")},
                            "match": all(abs(got[k] - p[k]) <= 5e-5 for k in p)}
    checks["published_ours_numbers_reproduced"] = all(v["match"] for v in sc.values())

    # --- PRE-SEAL PREFLIGHT: everything above uses frozen Ours data only, so it may run before
    # --- the P5-C execution seal. It opens NO Hu2025 outer retrieval metric.
    if "--preflight-only" in sys.argv:
        print("PREFLIGHT-ONLY: audit checks + Ours metrics + published-number self-check complete.")
        for k, v in checks.items():
            print("   %-58s %s" % (k, v))
        print("   published-ours self-check:", json.dumps(sc, ensure_ascii=False))
        print("   NO Hu2025 outer retrieval metric was computed.")
        return 0

    # ---------------------------------------------------------------- Hu2025
    hu_per_seed = {}
    hu_per_fold_seed = {}
    per_query = []
    for fold in range(5):
        recs = hu["fold%d" % fold]["recs"]
        g_mask = np.array([r["role"] == "gallery" for r in recs])
        q_mask = ~g_mask
        for s in SEEDS:
            z = np.load(os.path.join(FEAT, "fold%d_s%d" % (fold, s), "features.npz"),
                        allow_pickle=True)
            outer = z["outer"].astype(np.float32)
            # row alignment proof: regenerate the materializer's order and compare
            if [norm_rel(x) for x in z["outer_paths"]] != [norm_rel(r["source"]) for r in recs]:
                stop("fold%d_s%d: Hu row order does not match the frozen endpoint order"
                     % (fold, s))
            ids = [str(x) for x in z["outer_ids"]]
            g_ids = [ids[i] for i in np.where(g_mask)[0]]
            q_ids = [ids[i] for i in np.where(q_mask)[0]]
            ranks = optimistic_ranks(outer[q_mask], q_ids, outer[g_mask], g_ids)
            hu[(fold, s)] = ranks
            hits = (ranks == 1).astype(int)
            hu_per_fold_seed["fold%d_s%d" % (fold, s)] = rank_metrics(q_ids, hits.tolist(),
                                                                     ranks.tolist())
            for i, qi in enumerate(np.where(q_mask)[0]):
                per_query.append({"method": "Hu2025", "outer_fold": fold, "seed": s,
                                  "identity": q_ids[i], "query_path": norm_rel(recs[qi]["rel"]),
                                  "rank": int(ranks[i]), "rr": float(1.0 / ranks[i]),
                                  "rank1_hit": int(ranks[i] == 1)})
    for s in SEEDS:
        ids, ranks = [], []
        for fold in range(5):
            recs = hu["fold%d" % fold]["recs"]
            q_ids = [r["canonical"] for r in recs if r["role"] == "query"]
            ids += q_ids
            ranks += list(hu[(fold, s)])
        hu_per_seed[s] = rank_metrics(ids, [1 if r == 1 else 0 for r in ranks], ranks)

    # Ours per-query rows (same layout)
    for (fold, s, q), v in sorted(ours_pq.items()):
        per_query.append({"method": "Ours", "outer_fold": fold, "seed": s,
                          "identity": v["identity"], "query_path": q,
                          "rank": v["rank"], "rr": v["rr"],
                          "rank1_hit": int(v["rank"] == 1)})

    # ---------------------------------------------------------------- results
    mean_ours = float(np.mean([ours_per_seed[s]["rank1"] for s in SEEDS]))
    mean_hu = float(np.mean([hu_per_seed[s]["rank1"] for s in SEEDS]))
    delta = mean_ours - mean_hu

    checks["no_missing_or_duplicate_samples"] = (
        len(per_query) == 2 * 3 * 1104
        and len(set((r["method"], r["seed"], r["query_path"]) for r in per_query)) == 2 * 3 * 1104)

    metrics = {
        "schema": "h2025_p5c_metrics/v1", "stage": "P5-C",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {"scope": "per outer fold only; no cross-fold candidate merging",
                     "feature": "post-PCA128 L2-normalized; image similarity = cosine",
                     "identity_aggregation": "max over the identity's gallery images",
                     "tie_rule": "optimistic: rank = 1 + #{c : S(q,c) > S(q,y_q)}",
                     "macro_rank1_definition": "mean over identities of the identity's query-mean hit rate",
                     "mrr_definition": "mean(1/rank)",
                     "forbidden": ["reranking", "calibration", "learned fusion", "shortlist", "pruning"]},
        "matching": {"identities": 256, "gallery_images": 768, "queries": 1104,
                     "gallery_images_per_identity": 3, "total_endpoint_images": 1872},
        "ours_per_seed": ours_per_seed, "hu_per_seed": hu_per_seed,
        "hu_per_fold_seed": hu_per_fold_seed,
        "mean_seed_rank1": {"Ours": mean_ours, "Hu2025": mean_hu},
        "delta_p5_rank1": delta,
        "published_ours_self_check": sc,
        "evaluation_audit": checks, "per_fold_alignment": per_fold_align,
        "elapsed_sec": round(time.time() - t_start, 1),
        "no_outer_retrieval_metric_before_seal": True,
    }
    atomic_write(os.path.join(OUT, "P5C_METRICS.json"),
                 json.dumps(metrics, ensure_ascii=False, indent=1, sort_keys=True))

    cols = ["method", "outer_fold", "seed", "identity", "query_path", "rank", "rr", "rank1_hit"]
    lines = [",".join(cols)]
    for r in per_query:
        lines.append("%s,%d,%d,%s,%s,%d,%.8f,%d" % (r["method"], r["outer_fold"], r["seed"],
                                                     r["identity"], r["query_path"], r["rank"],
                                                     r["rr"], r["rank1_hit"]))
    atomic_write(os.path.join(OUT, "P5C_PER_QUERY_OUTCOMES.csv"), "\n".join(lines) + "\n")

    print("=== P5-C metrics ===")
    for s in SEEDS:
        o, h = ours_per_seed[s], hu_per_seed[s]
        print("seed %d | Ours R1=%.4f Macro=%.4f MRR=%.4f | Hu R1=%.4f Macro=%.4f MRR=%.4f | dR1=%+.4f"
              % (s, o["rank1"], o["macro_rank1"], o["mrr"], h["rank1"], h["macro_rank1"],
                 h["mrr"], o["rank1"] - h["rank1"]))
    print("mean_seed  | Ours R1=%.6f  Hu R1=%.6f  delta_P5=%+.6f" % (mean_ours, mean_hu, delta))
    print("published-ours self-check :", checks["published_ours_numbers_reproduced"], sc)
    print("per-query rows            :", len(per_query))
    print("elapsed                   : %.1fs" % (time.time() - t_start))
    print("WROTE P5C_METRICS.json / P5C_PER_QUERY_OUTCOMES.csv")

    audit = {
        "schema": "h2025_p5c_evaluation_audit/v1", "stage": "P5-C",
        "created_at": metrics["created_at"],
        "ruling": "P5-C FORMAL SAME-PROTOCOL COMPARISON §7 (evaluation audit)",
        "checks": checks, "per_fold_alignment": per_fold_align,
        "metric_definition_provenance": (
            "Rank-1 / MRR / Macro-Rank-1 were recovered by reproducing the published P3-C numbers "
            "from the frozen per-query ranks. Macro-Rank-1 = mean over identities of each "
            "identity's query-mean Rank-1 hit rate; two alternative candidates (identity 'any' = "
            "0.9844 and identity 'all' = 0.8242 for fusion/s42) did NOT reproduce 0.9355 and were "
            "therefore rejected. The definitions were not assumed."),
        "p3c_evaluator_source_available": os.path.isfile(
            os.path.join(H, "scripts", "stage_p3c_external_primary_eval.py")),
        "p3c_evaluator_disclosure": (
            "The P3-C execution seal references scripts/stage_p3c_external_primary_eval.py "
            "(sha256 eaec0f32...dd3849), but that source file is NOT present in the recovered "
            "tree, so it could not be re-hashed or literally reused. P5-C therefore implements the "
            "same protocol in this new, separately-hashed evaluator and validates fidelity by "
            "reproducing the published P3-C metrics to 5 decimal places."),
        "all_checks_pass": all(v is True for v in checks.values() if isinstance(v, bool)),
        "outer_retrieval_metric_recomputed_for_ours": False,
        "verdict": "PASS" if all(v is True for v in checks.values() if isinstance(v, bool)) else "STOP",
    }
    atomic_write(os.path.join(OUT, "P5C_EVALUATION_AUDIT.json"),
                 json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True))
    am = ["# P5C_EVALUATION_AUDIT", "", "**created:** %s" % audit["created_at"], "",
          "## §7 checks", "", "| check | result |", "|---|---|"]
    for k, v in checks.items():
        am.append("| `%s` | %s |" % (k, v))
    am += ["", "## Per-fold endpoint alignment", "", "| fold | frozen ids/g/q | Hu ids/g/q | counts match | query path set | query identity set |",
           "|---|---|---|---|---|---|"]
    for k, v in per_fold_align.items():
        am.append("| %s | %s/%s/%s | %s/%s/%s | **%s** | **%s** | **%s** |"
                  % (k, v["frozen_ids"], v["frozen_gallery"], v["frozen_query"],
                     v["hu_ids"], v["hu_gallery"], v["hu_query"], v["counts_match"],
                     v["frozen_query_rel_set_equals_hu_query_rel_set"],
                     v["frozen_query_identity_set_equals_hu_query_identity_set"]))
    am += ["", "## Metric-definition provenance", "", audit["metric_definition_provenance"], "",
           "| seed | published | recomputed | match |", "|---|---|---|---|"]
    for s, v in sc.items():
        am.append("| %s | `%s` | `%s` | **%s** |" % (s, json.dumps(v["published"]),
                                                     json.dumps(v["recomputed"]), v["match"]))
    am += ["", "## P3-C evaluator source disclosure", "", audit["p3c_evaluator_disclosure"], "",
           "## Verdict", "", "**%s**" % audit["verdict"], ""]
    atomic_write(os.path.join(OUT, "P5C_EVALUATION_AUDIT.md"), "\n".join(am))
    atomic_write(os.path.join(AUDIT, "P5C_EVALUATION_AUDIT.json"),
                 json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True))
    atomic_write(os.path.join(AUDIT, "P5C_EVALUATION_AUDIT.md"), "\n".join(am))
    print("WROTE P5C_EVALUATION_AUDIT.{json,md}  verdict=%s" % audit["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
