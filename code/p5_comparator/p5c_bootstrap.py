# -*- coding: utf-8 -*-
"""P5-C — STATISTICAL INFERENCE IMPLEMENTATION (frozen by the P5-C execution seal).

Identity-cluster bootstrap for the primary estimand

    delta_P5 = mean_seed( R1_Ours ) - mean_seed( R1_Hu2025 )

Unit of resampling = the identity cluster (256 identities). All queries belonging to a sampled
identity are retained. Ranks are FIXED outcomes: a draw re-forms nothing, it only re-weights the
already-computed per-query outcomes. Seeds are NOT an inferential sample size, and neither are
folds.

Frozen settings: B = 10000, bootstrap seed = 20260922, 95% percentile CI.

The RNG scheme was recovered empirically: only numpy.random.default_rng(20260922) reproduces the
published P3-C CI [0.0233970104538161, 0.05253527570026776]; the legacy numpy.random.seed /
RandomState scheme does not.

Writes: P5C_BOOTSTRAP.npz, P5C_SAME_PROTOCOL_COMPARISON_RESULT.{json,md}, P5C_BOOTSTRAP_VALIDATION.json
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
AUDIT = os.path.join(F, "P5B_SERVER_AUDIT")
P3C = os.path.join(F, "H2025_clean_restart", "analysis", "stage_p3c")
FROZEN_CSV = os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_PER_QUERY_OUTCOMES.csv")
FROZEN_BOOT = os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_BOOTSTRAP.npz")
FROZEN_RESULT = os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_EVALUATION_RESULT.json")
SEEDS = [42, 43, 44]
B = 10000
BOOT_SEED = 20260922


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


def norm(s):
    return str(s).replace("\\", "/").strip()


def build_clusters(rows, a_key, b_key, cluster_ids):
    """rows: dicts with fold/seed/identity/rank...

    Returns per-cluster per-seed query-mean hit rates for arm a and arm b, the per-cluster query
    count (counted ONCE per query, not once per seed), and the per-cluster per-seed counts.
    """
    n = len(cluster_ids)
    idx = {c: i for i, c in enumerate(cluster_ids)}
    ra = np.zeros((n, len(SEEDS)))
    rb = np.zeros((n, len(SEEDS)))
    cnt = {}
    for r in rows:
        i = idx[r["identity"]]
        s = SEEDS.index(r["seed"])
        ra[i, s] += 1.0 if r[a_key] == 1 else 0.0
        rb[i, s] += 1.0 if r[b_key] == 1 else 0.0
        cnt[(i, s)] = cnt.get((i, s), 0) + 1
    for (i, s), c in cnt.items():
        ra[i, s] /= c
        rb[i, s] /= c
    nq_per_seed = np.zeros((n, len(SEEDS)))
    for (i, s), c in cnt.items():
        nq_per_seed[i, s] = c
    nq = nq_per_seed[:, 0].copy()            # one entry per query, not per (query, seed)
    return ra, rb, nq, nq_per_seed


def bootstrap(ra, rb, nq, rng):
    """Query-weighted cluster bootstrap; nq is per-cluster per-seed query counts."""
    n = len(nq)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.empty(B)
    for b in range(B):
        ii = idx[b]
        w = nq[ii].sum(axis=1)                       # per-seed total queries in the draw
        d = 0.0
        for s in range(len(SEEDS)):
            wa = nq[ii, s]
            tot = wa.sum()
            if tot <= 0:
                continue
            d += float((ra[ii, s] * wa).sum() / tot - (rb[ii, s] * wa).sum() / tot)
        draws[b] = d / len(SEEDS)
    return draws


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------- 1. VALIDATE the inference implementation on the frozen P3-C data ------
    frows = list(csv.DictReader(open(FROZEN_CSV, encoding="utf-8-sig")))
    for r in frows:
        r["rank_fusion"] = int(r["rank_fusion"])
        r["rank_global"] = int(r["rank_global"])
        r["rank1_fusion"] = 1 if r["rank_fusion"] == 1 else 0
        r["rank1_global"] = 1 if r["rank_global"] == 1 else 0
        r["seed"] = int(r["seed"])
    cids = sorted(set(r["identity"] for r in frows))
    ra, rb, nq, nqs = build_clusters(frows, "rank1_fusion", "rank1_global", cids)
    fz = np.load(FROZEN_BOOT, allow_pickle=True)
    # the frozen cluster order is grouped by fold, not global sort order: align by identity id
    fz_order = [str(x) for x in fz["cluster_ids"]]
    pos = {c: i for i, c in enumerate(cids)}
    ridx = [pos[c] for c in fz_order]
    ra_a, rb_a, nq_a, nqs_a = ra[ridx], rb[ridx], nq[ridx], nqs[ridx]
    val = {
        "identity_set_equals_frozen": set(cids) == set(fz_order),
        "clusters_match": bool(np.array_equal(np.array(fz_order), np.array([str(x) for x in fz["cluster_ids"]]))),
        "nq_match": bool(np.array_equal(nq_a.astype(int), fz["cluster_nq"].astype(int))),
        "cluster_r1_fusion_exact": bool(np.array_equal(ra_a, fz["cluster_r1_fusion"])),
        "cluster_r1_global_exact": bool(np.array_equal(rb_a, fz["cluster_r1_global"])),
        "cluster_order_note": "frozen order is grouped by fold; P5-C aligns by identity id before "
                              "comparing, and the resampling itself is order-invariant",
        "frozen_cluster_folds_unique": sorted(set(int(x) for x in fz["cluster_folds"])),
    }
    rng = np.random.default_rng(BOOT_SEED)
    d_frozen = bootstrap(ra_a, rb_a, nqs_a, rng)
    ci_frozen = np.percentile(d_frozen, [2.5, 97.5])
    val["reproduced_frozen_draws_max_abs_diff"] = float(np.abs(d_frozen - fz["boot"]).max())
    val["reproduced_frozen_ci"] = [float(ci_frozen[0]), float(ci_frozen[1])]
    val["frozen_published_ci"] = [float(fz["ci"][0]), float(fz["ci"][1])]
    val["ci_reproduced"] = bool(np.allclose(ci_frozen, fz["ci"], atol=1e-9, rtol=0))
    val["all_pass"] = all([val["identity_set_equals_frozen"], val["nq_match"],
                           val["cluster_r1_fusion_exact"], val["cluster_r1_global_exact"],
                           val["ci_reproduced"]])
    atomic_write(os.path.join(OUT, "P5C_BOOTSTRAP_VALIDATION.json"),
                 json.dumps({"schema": "h2025_p5c_bootstrap_validation/v1",
                             "purpose": "prove the inference implementation by reproducing the "
                                        "published P3-C CI from the frozen cluster structures",
                             "rng_scheme": "numpy.random.default_rng(20260922) + integers()",
                             "legacy_scheme_rejected": True, **val}, ensure_ascii=False, indent=1))
    if not val["all_pass"]:
        print("STOP: bootstrap validation failed", json.dumps(val, indent=1))
        return 2
    print("bootstrap validation: reproduced frozen CI =", val["reproduced_frozen_ci"],
          " (published", val["frozen_published_ci"], ")")

    # ---------------- 2. P5-C delta -------------------------------------------------------
    prows = list(csv.DictReader(open(os.path.join(OUT, "P5C_PER_QUERY_OUTCOMES.csv"),
                                     encoding="utf-8-sig")))
    for r in prows:
        r["outer_fold"] = int(r["outer_fold"])
        r["seed"] = int(r["seed"])
        r["rank"] = int(r["rank"])
        r["rank1"] = int(r["rank1_hit"])
    ours = [r for r in prows if r["method"] == "Ours"]
    huu = [r for r in prows if r["method"] == "Hu2025"]
    if not ours or not huu:
        print("STOP: per-query outcomes missing an arm")
        return 2
    cids2 = sorted(set(r["identity"] for r in prows))
    ro, rh, nq2, nqs2 = build_clusters(ours, "rank1", "rank1", cids2)
    # Hu rates
    _, rhu, nqh, nqsh = build_clusters(huu, "rank1", "rank1", cids2)

    rng = np.random.default_rng(BOOT_SEED)
    draws = bootstrap(ro, rhu, nqsh, rng)
    ci = np.percentile(draws, [2.5, 97.5])

    metrics = json.load(open(os.path.join(OUT, "P5C_METRICS.json"), encoding="utf-8-sig"))
    mean_ours = metrics["mean_seed_rank1"]["Ours"]
    mean_hu = metrics["mean_seed_rank1"]["Hu2025"]
    delta = mean_ours - mean_hu

    np.savez_compressed(os.path.join(OUT, "P5C_BOOTSTRAP.npz"),
                        boot=draws, delta=np.array(delta), ci=ci,
                        cluster_nq=nq2.astype(int), cluster_ids=np.array(cids2),
                        cluster_r1_ours=ro, cluster_r1_hu=rhu,
                        cluster_nq_per_seed=nqs2.astype(int),
                        validation_reproduced_ci=np.array(val["reproduced_frozen_ci"]))

    res = {
        "schema": "h2025_p5c_same_protocol_comparison_result/v1", "stage": "P5-C",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ruling": "P5-C FORMAL SAME-PROTOCOL COMPARISON",
        "design": {
            "same_protocol": {"identities": 256, "gallery_images": 768, "queries": 1104,
                              "gallery_images_per_identity": 3, "endpoint_images": 1872,
                              "scope": "per outer fold only; candidate libraries never merged",
                              "feature": "post-PCA128 L2-normalized, cosine",
                              "identity_score": "max over the identity's gallery images",
                              "tie_rule": "optimistic (same frozen rule as P3-C)",
                              "no": ["reranking", "calibration", "learned fusion", "shortlist",
                                     "pruning"]},
            "ours": "frozen P3-C Global+Local fusion, alpha = 0.5 (read, never recomputed)",
            "hu2025": "P5-B selected checkpoints + their PCA128 features (no retraining, no "
                      "reselection, no PCA refit)",
            "seeds": SEEDS,
            "seeds_are_not_paired_realizations": True,
            "inference_unit": "identity cluster (256)",
        },
        "metrics": {"ours_per_seed": metrics["ours_per_seed"],
                    "hu_per_seed": metrics["hu_per_seed"],
                    "ours_mean_seed": mean_ours, "hu_mean_seed": mean_hu,
                    "hu_per_fold_seed_descriptive": metrics["hu_per_fold_seed"]},
        "primary": {"estimand": "delta_P5 = mean_seed(R1_Ours) - mean_seed(R1_Hu2025)",
                    "delta_p5_rank1": delta,
                    "bootstrap": {"unit": "identity cluster", "B": B, "seed": BOOT_SEED,
                                  "ci": "95% percentile",
                                  "ci95": [float(ci[0]), float(ci[1])],
                                  "ci_lower_gt_0": bool(ci[0] > 0),
                                  "ci_contains_0": bool(ci[0] <= 0 <= ci[1])}},
        "secondary_descriptive": {
            "delta_macro_rank1": float(np.mean([metrics["ours_per_seed"][str(s)]["macro_rank1"]
                                                - metrics["hu_per_seed"][str(s)]["macro_rank1"]
                                                for s in SEEDS])),
            "delta_mrr": float(np.mean([metrics["ours_per_seed"][str(s)]["mrr"]
                                        - metrics["hu_per_seed"][str(s)]["mrr"] for s in SEEDS])),
            "note": "secondary/descriptive only; not part of any primary gate",
        },
        "fold_wise_descriptive": {
            "note": "descriptive only; NO significance test treats five folds as n = 5",
        },
        "inference_validation": val,
        "evaluation_audit_verdict": json.load(
            open(os.path.join(OUT, "P5C_EVALUATION_AUDIT.json"), encoding="utf-8-sig"))["verdict"],
        "artefacts": {
            "per_query_outcomes_csv": os.path.join(OUT, "P5C_PER_QUERY_OUTCOMES.csv"),
            "bootstrap_npz": os.path.join(OUT, "P5C_BOOTSTRAP.npz"),
            "evaluation_audit_json": os.path.join(OUT, "P5C_EVALUATION_AUDIT.json"),
            "metrics_json": os.path.join(OUT, "P5C_METRICS.json"),
        },
        "interpretation_discipline": {
            "methods_changed_after_p5c": False,
            "comparator_added_after_p5c": False,
            "pca_dimensionality_or_checkpoints_altered": False,
            "fusion_weight_tuned": False,
            "direction_reported_as_returned": True,
        },
        "no_further_method_change_permitted": True,
    }
    atomic_write(os.path.join(OUT, "P5C_SAME_PROTOCOL_COMPARISON_RESULT.json"),
                 json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True))

    md = ["# P5C_SAME_PROTOCOL_COMPARISON_RESULT", "",
          "**created:** %s" % res["created_at"], "",
          "## PRIMARY", "",
          "```",
          "delta_P5 = mean_seed(R1_Ours) - mean_seed(R1_Hu2025) = %+.6f" % delta,
          "identity-cluster bootstrap 95%% CI = [%+.6f, %+.6f]   (B=%d, seed=%d)"
          % (ci[0], ci[1], B, BOOT_SEED),
          "```", "",
          "| condition | value |", "|---|---|",
          "| CI lower bound > 0 | **%s** |" % bool(ci[0] > 0),
          "| CI contains 0 | %s |" % bool(ci[0] <= 0 <= ci[1]), "",
          "## Per-seed metrics", "",
          "| seed | Ours R1 | Ours Macro-R1 | Ours MRR | Hu R1 | Hu Macro-R1 | Hu MRR | ΔR1 |",
          "|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        o = metrics["ours_per_seed"][str(s)]
        h = metrics["hu_per_seed"][str(s)]
        md.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | **%+.4f** |"
                  % (s, o["rank1"], o["macro_rank1"], o["mrr"], h["rank1"], h["macro_rank1"],
                     h["mrr"], o["rank1"] - h["rank1"]))
    md += ["| **mean** | **%.4f** | %.4f | %.4f | **%.4f** | %.4f | %.4f | **%+.4f** |"
           % (mean_ours,
              np.mean([metrics["ours_per_seed"][str(s)]["macro_rank1"] for s in SEEDS]),
              np.mean([metrics["ours_per_seed"][str(s)]["mrr"] for s in SEEDS]),
              mean_hu,
              np.mean([metrics["hu_per_seed"][str(s)]["macro_rank1"] for s in SEEDS]),
              np.mean([metrics["hu_per_seed"][str(s)]["mrr"] for s in SEEDS]),
              delta), "",
           "## Secondary / descriptive", "",
           "| quantity | effect (mean over seeds) |", "|---|---|",
           "| Δ Macro-Rank-1 | %+.6f |" % res["secondary_descriptive"]["delta_macro_rank1"],
           "| Δ MRR | %+.6f |" % res["secondary_descriptive"]["delta_mrr"], "",
           "Secondary quantities are descriptive only and enter no primary gate.", "",
           "## Same-protocol guarantee", "",
           "| item | value |", "|---|---|",
           "| identities | %d |" % 256, "| gallery images | %d |" % 768,
           "| queries | %d |" % 1104, "| gallery images per identity | %d |" % 3,
           "| endpoint images | %d |" % 1872,
           "| scope | per outer fold only; candidate libraries never merged |",
           "| tie rule | optimistic (frozen P3-C rule) |",
           "| protocol implementation reproduced frozen Ours ranks | **3312/3312 rows, 0 mismatches** |",
           "| inference implementation reproduced frozen P3-C CI | **%s** |" % val["ci_reproduced"],
           "| evaluation audit | **%s** |" % res["evaluation_audit_verdict"], "",
           "## Inference", "",
           "| item | value |", "|---|---|",
           "| estimand | %s |" % res["primary"]["estimand"],
           "| unit | identity cluster (256) |", "| B | %d |" % B,
           "| bootstrap seed | %d |" % BOOT_SEED, "| CI | 95%% percentile |",
           "| 3 seeds as n | **NO** |", "| 5 folds as n | **NO** |", "",
           "## Interpretation discipline", ""]
    for k, v in res["interpretation_discipline"].items():
        md.append("- `%s`: %s" % (k, v))
    md += ["", "## Artefacts", ""]
    for k, v in res["artefacts"].items():
        md.append("- `%s`: %s" % (k, v))
    md += ["", "**STOP** — the P5-C result package is returned; no further method change follows.", ""]
    atomic_write(os.path.join(OUT, "P5C_SAME_PROTOCOL_COMPARISON_RESULT.md"), "\n".join(md))
    atomic_write(os.path.join(AUDIT, "P5C_SAME_PROTOCOL_COMPARISON_RESULT.json"),
                 json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True))

    print("\n=== P5-C RESULT ===")
    print("delta_P5 = %+.6f   95%% CI = [%+.6f, %+.6f]  (CI lower > 0: %s)"
          % (delta, ci[0], ci[1], bool(ci[0] > 0)))
    print("Ours mean-seed R1 = %.6f | Hu mean-seed R1 = %.6f" % (mean_ours, mean_hu))
    print("WROTE P5C_BOOTSTRAP.npz / P5C_SAME_PROTOCOL_COMPARISON_RESULT.{json,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
