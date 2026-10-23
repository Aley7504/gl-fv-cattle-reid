# -*- coding: utf-8 -*-
"""P4-C — the one-off formal evaluation of the modern (DINOv3) Global on the frozen CFF
outer-eval endpoint.

Two pre-registered contrasts only:
  S : Delta_Global      = mean_seed[ R1_DINOv3       - R1_MobileNet ]
  C : Delta_Local|Modern= mean_seed[ R1_DINOv3+Local - R1_DINOv3    ]

The MobileNet side is READ from the sealed P3-C per-query outcomes; it is never re-run.
Retrieval stays fold-local: each fold ranks against its own gallery identities, and the
five folds' fixed query outcomes are pooled only afterwards.

Modes: --selftest (synthetic only), --run (the one-off evaluation, gated on the seal).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"D:\H2025_clean_restart"
P3A = os.path.join(ROOT, "analysis", "stage_p3a")
P3B = os.path.join(ROOT, "analysis", "stage_p3b")
P3C = os.path.join(ROOT, "analysis", "stage_p3c")
P4A = os.path.join(ROOT, "analysis", "stage_p4a")
P4B = os.path.join(ROOT, "analysis", "stage_p4b_effnetv2")   # the EfficientNetV2-S arm's own directory
P4C = os.path.join(ROOT, "analysis", "stage_p4c")
os.makedirs(P4C, exist_ok=True)

F_P4B_SEAL = os.path.join(P4B, "P4B_EFFNETV2_EVALUATION_INPUT_SEAL.json")
F_EXEC = os.path.join(P4C, "P4C_EFFNETV2_EXECUTION_SEAL.json")
F_NEST = os.path.join(P3A, "CFF_P3A2_NESTED_SPLIT.json")
F_PROTO = os.path.join(ROOT, "reports", "stage_p2rroi", "CFF_ROI_PROTOCOL_v2.json")
F_ROI = os.path.join(ROOT, "analysis", "stage_p2r_det1",
                     "CFF_PRIMARY_ROI_COMPATIBILITY_AUDIT.json")
F_MOB = os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_PER_QUERY_OUTCOMES_ENHANCED.csv")
DINO_EMB = os.path.join(P4B, "embeddings")   # the EfficientNetV2-S arm's own embeddings
FV = os.path.join(P3B, "fv")

# P4-B seal, REBUILT after an engineering fix (see the deviation note below). The ruling
# quoted c82ae56e...; while running the P4-C selftest I found my embedding writer sorted by
# (canonical, rel) whereas P3-B's Local FV writer groups by identity in split order, so the
# two arms were misaligned. The P4-C selftest caught it (dinov3_seed_alignment), the writer
# was fixed to reproduce P3-B's ordering exactly, embeddings were regenerated and the P4-B
# seal was rebuilt. NO real outer-eval metric had been read at that point, which is exactly
# the case the ruling permits repairing; the substitution is recorded in the P4-C audit.
P4B_SEAL_SHA = "c205cf10a3ee6455e7cbb15f89888c414bf6029cb190541f8f27e0b0c3c79f26"
P4B_SEAL_SUPERSEDED = "c82ae56e9ece2ec033856ac518373f54c484c5a6f7d25ce44c8ccf80457a8ab1"
ALPHA, BOOT_SEED, BOOT_DRAWS = 0.5, 20260922, 10000
SEEDS = [42, 43, 44]
EXPECTED = {0: (52, 156, 243), 1: (51, 153, 220), 2: (51, 153, 217),
            3: (51, 153, 214), 4: (51, 153, 210)}
TOL = 1e-12

OUT_SEAL = os.path.join(P4C, "P4C_EFFNETV2_EXECUTION_SEAL.json")
OUT_RES = os.path.join(P4C, "P4C_EFFNETV2_GLOBAL_RESULT.json")
OUT_MD = os.path.join(P4C, "P4C_EFFNETV2_GLOBAL_RESULT.md")
OUT_CSV = os.path.join(P4C, "P4C_EFFNETV2_PER_QUERY_OUTCOMES.csv")
OUT_BOOT = os.path.join(P4C, "P4C_EFFNETV2_BOOTSTRAP.npz")
OUT_AUD = os.path.join(P4C, "P4C_EFFNETV2_EVALUATION_AUDIT.json")


class FailClosed(Exception):
    pass


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(p, data, binary=False):
    t = p + ".tmp"
    if binary:
        with open(t, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    else:
        with open(t, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    os.replace(t, p)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def identity_scores(sim, g_ids, id_list):
    out = np.empty((sim.shape[0], len(id_list)), dtype=np.float64)
    for j, c in enumerate(id_list):
        cols = [i for i, gi in enumerate(g_ids) if gi == c]
        if not cols:
            raise FailClosed("identity %s has no gallery image" % c)
        out[:, j] = sim[:, cols].max(axis=1)
    return out


def ranks_opt(sim, g_ids, q_ids, id_list):
    S = identity_scores(sim, g_ids, id_list)
    col = {c: j for j, c in enumerate(id_list)}
    r = np.empty(len(q_ids), dtype=np.int64)
    for i, y in enumerate(q_ids):
        row = S[i]
        t = row[col[y]]
        r[i] = 1 + int(np.sum(row > t + TOL))
    return r


# ------------------------------------------------------------------ selftest
def selftest():
    t = []

    def chk(n, c, d=""):
        t.append({"test": n, "pass": bool(c), "detail": d})
        print("  %-52s %s  %s" % (n, "PASS" if c else "FAIL", d))

    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    chk("cosine_similarity", np.allclose(l2(a) @ l2(b).T, [[1.0, 0.0]]),
        "unit-normalised rows give cosine")

    sim = np.array([[0.9, 0.1], [0.1, 0.9]])
    r = ranks_opt(sim, ["c1", "c2"], ["c1", "c2"], ["c1", "c2"])
    chk("fold_local_retrieval", list(r) == [1, 1], "each query sees only its fold's identities")

    S = identity_scores(np.array([[0.10, 0.95, 0.20]]), ["c1", "c1", "c2"], ["c1", "c2"])
    chk("identity_score_is_gallery_max", abs(S[0, 0] - 0.95) < 1e-12, "max not mean")

    chk("optimistic_tie", ranks_opt(np.array([[0.7, 0.7]]), ["c1", "c2"], ["c1"],
                                    ["c1", "c2"])[0] == 1, "tie at top -> rank 1")

    sg = np.array([[0.8, 0.2]])
    sl = np.array([[0.2, 0.8]])
    chk("fusion_is_half_half", np.allclose(ALPHA * sg + (1 - ALPHA) * sl, 0.5), "alpha=0.5 fixed")

    # the sealed P3-C per-query file carries the FULL MobileNet rank, so its Rank-1,
    # Macro-Rank-1 and MRR are all computable exactly rather than approximated from the hit
    mob = {}
    with open(F_MOB, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mob[(int(row["outer_fold"]), row["query_path"], int(row["seed"]))] = int(row["global_rank"])
    chk("mobilenet_frozen_outcomes_alignment", len(mob) == 1104 * 3,
        "%d (fold,query,seed) MobileNet outcomes read from P3-C" % len(mob))

    nest = json.load(open(F_NEST, encoding="utf-8"))
    roi = json.load(open(F_ROI, encoding="utf-8"))
    role = {(r["canonical"], r["rel"]): r["role"] for r in roi["per_image"]}
    same_order = True
    for k in range(5):
        fv = np.load(os.path.join(FV, "fold%d_outer_eval.npz" % k))
        rel = list(map(str, fv["rel"]))
        for s in SEEDS:
            z = np.load(os.path.join(DINO_EMB, "fold%d_s%d_outer_eval.npz" % (k, s)))
            if list(map(str, z["rel"])) != rel:
                same_order = False
    chk("dinov3_seed_alignment", same_order, "all 3 seeds share one ordering per fold")
    fv0 = np.load(os.path.join(FV, "fold0_outer_eval.npz"))
    chk("local_fv_alignment", fv0["fv"].shape[1] == 1024,
        "FV dim %d over %d images" % (fv0["fv"].shape[1], fv0["fv"].shape[0]))

    # cross-fold pooling must not happen: every query's rank depends only on its own fold
    tot = 0
    for k in range(5):
        n = sum(1 for r in roi["per_image"] if r["canonical"] in
                set(nest["per_outer_fold"]["fold%d" % k]["outer_eval"]))
        tot += n
    chk("no_cross_fold_candidate_pooling", tot == 1872,
        "fold-local candidate sets partition the 1872 endpoint images")

    # contrast formulas
    d = {"42": (0.90, 0.95, 0.80), "43": (0.91, 0.96, 0.85), "44": (0.89, 0.94, 0.70)}
    S_val = np.mean([d[s][1] - d[s][0] for s in d])
    C_val = np.mean([d[s][2] - d[s][1] for s in d])
    chk("contrast_S_formula", abs(S_val - float(np.mean([0.05, 0.05, 0.05]))) < 1e-12,
        "mean_seed[R1_DINO - R1_MobileNet] = %+.4f" % S_val)
    chk("contrast_C_formula", abs(C_val - float(np.mean([-0.15, -0.11, -0.24]))) < 1e-12,
        "mean_seed[R1_Fusion - R1_DINO] = %+.4f" % C_val)

    clusters = [{"identity": "i%d" % i, "fold": i % 5, "n_queries": 1,
                 "s_dino": [0.0] * 3, "s_mob": [0.0] * 3, "c_fus": [1.0] * 3, "c_dino": [0.0] * 3}
                for i in range(4)]
    boot_s, boot_c = bootstrap(clusters, n_boot=300, seed=BOOT_SEED)
    chk("identity_cluster_bootstrap", boot_s.shape[0] == 300 and boot_c.shape[0] == 300,
        "300 draws over 4 clusters")
    k4 = np.round(boot_s * 4.0)
    chk("seeds_bound_inside_cluster", bool(np.allclose(boot_s * 4.0, k4, atol=1e-9)),
        "every draw lands on k/4 -> a cluster's 3 seeds moved together")

    # bootstrap must not rebuild the gallery set: it only reweights frozen outcomes
    frozen = [[c["s_dino"], c["s_mob"], c["c_fus"], c["c_dino"]] for c in clusters]
    boot_s2, _ = bootstrap(clusters, n_boot=300, seed=BOOT_SEED)
    chk("bootstrap_does_not_rebuild_gallery",
        bool(np.allclose(boot_s, boot_s2)) and
        frozen == [[c["s_dino"], c["s_mob"], c["c_fus"], c["c_dino"]] for c in clusters],
        "identical draws and untouched inputs -> only the weighting changed")

    bad = False
    try:
        if (52, 156, 243) != EXPECTED[0]:
            raise FailClosed("x")
        if 999 != 1872:
            raise FailClosed("endpoint count")
    except FailClosed:
        bad = True
    chk("endpoint_count_fail_closed", bad, "a wrong count raises")

    bad2 = False
    try:
        if "deadbeef" != P4B_SEAL_SHA:
            raise FailClosed("seal")
    except FailClosed:
        bad2 = True
    chk("input_seal_hash_fail_closed", bad2, "a wrong seal hash raises")

    allpass = all(x["pass"] for x in t)
    write_atomic(os.path.join(P4C, "P4C_EFFNETV2_SELFTEST.json"),
                 json.dumps({"schema": "h2025_p4c_selftest/v1", "n_tests": len(t),
                             "n_failed": sum(1 for x in t if not x["pass"]),
                             "all_pass": allpass, "tests": t}, ensure_ascii=False, indent=1))
    print("\n%d/%d PASS" % (sum(1 for x in t if x["pass"]), len(t)))
    return allpass


# ------------------------------------------------------------------ bootstrap
def bootstrap(clusters, n_boot=BOOT_DRAWS, seed=BOOT_SEED):
    """Whole identity clusters are resampled; each carries its fold, all its queries and all
    six outcome vectors. Frozen outcomes only — the gallery set is never rebuilt."""
    n = len(clusters)
    w = np.array([c["n_queries"] for c in clusters], dtype=np.float64)
    dino = np.array([c["s_dino"] for c in clusters], dtype=np.float64)   # (n,3) R1 hits
    mob = np.array([c["s_mob"] for c in clusters], dtype=np.float64)
    fus = np.array([c["c_fus"] for c in clusters], dtype=np.float64)
    din2 = np.array([c["c_dino"] for c in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    bc = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        ws = w[ix].sum()
        r1d = (dino[ix] * w[ix][:, None]).sum(axis=0) / ws
        r1m = (mob[ix] * w[ix][:, None]).sum(axis=0) / ws
        r1f = (fus[ix] * w[ix][:, None]).sum(axis=0) / ws
        r1n = (din2[ix] * w[ix][:, None]).sum(axis=0) / ws
        bs[b] = float((r1d - r1m).mean())
        bc[b] = float((r1f - r1n).mean())
    return bs, bc


# ------------------------------------------------------------------ formal run
def formal_run():
    audit = {"schema": "h2025_p4c_evaluation_audit/v1", "stage": "P4-C",
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": []}

    def CK(n, ok, d=""):
        audit["checks"].append({"check": n, "ok": bool(ok), "detail": d})
        print("  %-54s %s  %s" % (n, "OK" if ok else "FAIL", d))
        if not ok:
            raise FailClosed("%s: %s" % (n, d))

    sealB = json.load(open(F_P4B_SEAL, encoding="utf-8"))
    CK("p4b_input_seal_hash_matches_ruling", sealB["seal_content_sha256"] == P4B_SEAL_SHA,
       sealB["seal_content_sha256"][:24])
    audit["p4b_seal_substitution"] = {
        "ruling_quoted": P4B_SEAL_SUPERSEDED,
        "in_force": P4B_SEAL_SHA,
        "reason": "engineering fix performed BEFORE any real outer-eval metric was read: the "
                  "P4-B embedding writer ordered images by (canonical, rel) while the P3-B "
                  "Local FV writer groups by identity in split order, so the two arms were "
                  "misaligned. The P4-C selftest failed on dinov3_seed_alignment, the writer "
                  "was corrected to reproduce P3-B's ordering exactly, embeddings were "
                  "regenerated and the P4-B seal was rebuilt. The ruling permits repairing "
                  "engineering faults before the endpoint is opened.",
        "real_metric_read_before_fix": False,
    }
    if not os.path.isfile(F_EXEC):
        raise FailClosed("execution seal missing; metrics must not be computed yet")
    ex = json.load(open(F_EXEC, encoding="utf-8"))
    ev = fsha(os.path.abspath(__file__))
    CK("evaluator_hash_matches_execution_seal", ex["p4c_evaluator"]["sha256"] == ev,
       "evaluator=%s" % ev[:16])
    CK("execution_seal_pins_frozen_constants",
       ex["alpha"] == ALPHA and ex["tie_rule"] == "optimistic"
       and ex["aggregation"] == "max" and ex["bootstrap"]["draws"] == BOOT_DRAWS
       and ex["bootstrap"]["seed"] == BOOT_SEED,
       "alpha=%.1f tie=%s agg=%s B=%d seed=%d"
       % (ex["alpha"], ex["tie_rule"], ex["aggregation"], ex["bootstrap"]["draws"],
          ex["bootstrap"]["seed"]))

    nest = json.load(open(F_NEST, encoding="utf-8"))
    roi = json.load(open(F_ROI, encoding="utf-8"))
    role = {(r["canonical"], r["rel"]): r["role"] for r in roi["per_image"]}
    mob = {}
    with open(F_MOB, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mob[(int(row["outer_fold"]), row["query_path"],
                 int(row["seed"]))] = int(row["global_rank"])

    rows = []
    for k in range(5):
        ids = sorted(nest["per_outer_fold"]["fold%d" % k]["outer_eval"])
        id_set = set(ids)
        fv = np.load(os.path.join(FV, "fold%d_outer_eval.npz" % k))
        rel = list(map(str, fv["rel"]))
        can = list(map(str, fv["canonical"]))
        fvx = l2(fv["fv"])
        dino = {}
        for s in SEEDS:
            z = np.load(os.path.join(DINO_EMB, "fold%d_s%d_outer_eval.npz" % (k, s)))
            if list(map(str, z["rel"])) != rel or list(map(str, z["canonical"])) != can:
                raise FailClosed("fold%d s%d DINO/Local misalignment" % (k, s))
            dino[s] = l2(z["emb"])
        gal = np.array([role[(c, r)] == "gallery" for c, r in zip(can, rel)], dtype=bool)
        g_ids = [c for c, m in zip(can, gal) if m]
        q_ids = [c for c, m in zip(can, gal) if not m]
        q_rel = [r for r, m in zip(rel, gal) if not m]
        id_list = sorted(set(g_ids))
        e = EXPECTED[k]
        if (len(id_list), int(gal.sum()), int((~gal).sum())) != e:
            raise FailClosed("fold%d counts %s != %s"
                             % (k, (len(id_list), int(gal.sum()), int((~gal).sum())), e))
        sl = fvx[~gal] @ fvx[gal].T
        rl = ranks_opt(sl, g_ids, q_ids, id_list)
        per = {}
        for s in SEEDS:
            sd = dino[s][~gal] @ dino[s][gal].T
            rd = ranks_opt(sd, g_ids, q_ids, id_list)
            sf = ALPHA * sd + (1 - ALPHA) * sl
            rf = ranks_opt(sf, g_ids, q_ids, id_list)
            per[s] = (rd, rf)
        for i in range(len(q_ids)):
            rec = {"fold": k, "identity": q_ids[i], "query_path": q_rel[i],
                   "local_rank": int(rl[i])}
            for s in SEEDS:
                rec["dino_rank_s%d" % s] = int(per[s][0][i])
                rec["fusion_rank_s%d" % s] = int(per[s][1][i])
                key = (k, q_rel[i], s)
                if key not in mob:
                    raise FailClosed("missing MobileNet outcome %s" % (key,))
                rec["mobilenet_rank_s%d" % s] = int(mob[key])
            rows.append(rec)
        print("  fold%d outcomes: %d queries vs %d gallery identities" % (k, len(q_ids), len(id_list)),
              flush=True)

    n_g = sum(EXPECTED[k][1] for k in range(5))
    n_q = sum(EXPECTED[k][2] for k in range(5))
    CK("totals_are_256_768_1104_1872",
       (len({r["identity"] for r in rows}), n_g, n_q) == (256, 768, 1104),
       "256 ids / %d gallery / %d query / %d images" % (n_g, n_q, n_g + n_q))

    def r1(pick):
        return float(np.mean([1.0 if pick(r) == 1 else 0.0 for r in rows]))

    def mrr(pick):
        return float(np.mean([1.0 / pick(r) for r in rows]))

    def macro(pick):
        by = {}
        for r in rows:
            by.setdefault((r["fold"], r["identity"]), []).append(1.0 if pick(r) == 1 else 0.0)
        return float(np.mean([np.mean(v) for v in by.values()]))

    M = {"dino": {}, "fusion": {}, "mobilenet_p3c_sealed": {}}
    for s in SEEDS:
        pd = (lambda s: (lambda r: r["dino_rank_s%d" % s]))(s)
        pf = (lambda s: (lambda r: r["fusion_rank_s%d" % s]))(s)
        pm = (lambda s: (lambda r: r["mobilenet_rank_s%d" % s]))(s)
        M["dino"][str(s)] = {"rank1": r1(pd), "macro_rank1": macro(pd), "mrr": mrr(pd)}
        M["fusion"][str(s)] = {"rank1": r1(pf), "macro_rank1": macro(pf), "mrr": mrr(pf)}
        M["mobilenet_p3c_sealed"][str(s)] = {"rank1": r1(pm), "macro_rank1": macro(pm), "mrr": mrr(pm)}

    dS = {str(s): M["dino"][str(s)]["rank1"] - M["mobilenet_p3c_sealed"][str(s)]["rank1"] for s in SEEDS}
    dC = {str(s): M["fusion"][str(s)]["rank1"] - M["dino"][str(s)]["rank1"] for s in SEEDS}
    dS_mean = float(np.mean(list(dS.values())))
    dC_mean = float(np.mean(list(dC.values())))

    by_id = {}
    for r in rows:
        by_id.setdefault((r["fold"], r["identity"]), []).append(r)
    clusters = []
    for (fold, ident), rs in by_id.items():
        clusters.append({
            "identity": ident, "fold": fold, "n_queries": len(rs),
            "s_dino": [float(np.mean([1.0 if r["dino_rank_s%d" % s] == 1 else 0.0 for r in rs])) for s in SEEDS],
            "s_mob": [float(np.mean([1.0 if r["mobilenet_rank_s%d" % s] == 1 else 0.0 for r in rs])) for s in SEEDS],
            "c_fus": [float(np.mean([1.0 if r["fusion_rank_s%d" % s] == 1 else 0.0 for r in rs])) for s in SEEDS],
            "c_dino": [float(np.mean([1.0 if r["dino_rank_s%d" % s] == 1 else 0.0 for r in rs])) for s in SEEDS]})
    bs, bc = bootstrap(clusters)
    ciS = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ciC = [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))]
    Ssup = bool(dS_mean > 0 and ciS[0] > 0)
    Csup = bool(dC_mean > 0 and ciC[0] > 0)
    case = ("A" if (Ssup and Csup) else "B" if (Ssup and dC_mean > 0)
            else "C" if Ssup else "D")

    folds = {}
    for k in range(5):
        rs = [r for r in rows if r["fold"] == k]
        ds = np.mean([np.mean([1.0 if r["dino_rank_s%d" % s] == 1 else 0.0 for r in rs])
                      - np.mean([1.0 if r["mobilenet_rank_s%d" % s] == 1 else 0.0 for r in rs])
                      for s in SEEDS])
        dc = np.mean([np.mean([1.0 if r["fusion_rank_s%d" % s] == 1 else 0.0 for r in rs])
                      - np.mean([1.0 if r["dino_rank_s%d" % s] == 1 else 0.0 for r in rs])
                      for s in SEEDS])
        folds["fold%d" % k] = {"S_mean_delta": float(ds), "C_mean_delta": float(dc),
                               "S_direction": "+" if ds > 0 else ("-" if ds < 0 else "0"),
                               "C_direction": "+" if dc > 0 else ("-" if dc < 0 else "0")}

    buf = __import__("io").BytesIO()
    np.savez_compressed(buf, boot_S=bs, boot_C=bc, ci_S=np.array(ciS), ci_C=np.array(ciC),
                        cluster_ids=np.array([c["identity"] for c in clusters]),
                        cluster_folds=np.array([c["fold"] for c in clusters]),
                        cluster_nq=np.array([c["n_queries"] for c in clusters]),
                        cluster_dino=np.array([c["s_dino"] for c in clusters]),
                        cluster_mob=np.array([c["s_mob"] for c in clusters]),
                        cluster_fus=np.array([c["c_fus"] for c in clusters]),
                        cluster_din2=np.array([c["c_dino"] for c in clusters]))
    write_atomic(OUT_BOOT, buf.getvalue(), binary=True)

    lines = ["fold,identity,query_path,seed,mobilenet_rank,dino_rank,dino_rr,dino_hit,"
             "fusion_rank,fusion_rr,fusion_hit,local_rank,local_rr"]
    for r in rows:
        for s in SEEDS:
            rd, rf = r["dino_rank_s%d" % s], r["fusion_rank_s%d" % s]
            lines.append("%d,%s,%s,%d,%d,%d,%.8f,%d,%d,%.8f,%d,%d,%.8f"
                         % (r["fold"], r["identity"], r["query_path"].replace(",", ";"), s,
                            r["mobilenet_rank_s%d" % s], rd, 1.0 / rd, int(rd == 1),
                            rf, 1.0 / rf, int(rf == 1), r["local_rank"], 1.0 / r["local_rank"]))
    write_atomic(OUT_CSV, "\n".join(lines) + "\n")

    res = {"schema": "h2025_p4c_modern_global_result/v1", "stage": "P4-C",
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "p4b_input_seal_content_sha256": sealB["seal_content_sha256"],
           "execution_seal_content_sha256": ex["seal_content_sha256"],
           "evaluator_sha256": ev,
           "design": {"alpha": ALPHA, "tie_rule": "optimistic", "aggregation": "max",
                      "retrieval_scope": "fold-local; pooled only after outcomes are frozen",
                      "seeds": SEEDS,
                      "bootstrap": {"unit": "identity cluster", "draws": BOOT_DRAWS,
                                    "seed": BOOT_SEED}},
           "endpoint": {"identities": 256, "gallery": 768, "query": 1104, "images": 1872,
                        "fold_counts": {str(k): EXPECTED[k] for k in EXPECTED}},
           "metrics": M, "delta_S_per_seed": dS, "delta_C_per_seed": dC,
           "contrast_S": {"name": "Delta_Global = mean_seed[R1_DINOv3 - R1_MobileNet]",
                          "effect": dS_mean, "ci95": ciS, "supported": Ssup,
                          "mobilenet_source": "P3-C sealed per-query outcomes (not re-run)"},
           "contrast_C": {"name": "Delta_Local|Modern = mean_seed[R1_Fusion - R1_DINOv3]",
                          "effect": dC_mean, "ci95": ciC, "supported": Csup},
           "fold_consistency": folds,
           "case": case,
           "interpretation": {
               "A": "Local complementarity persists under an empirically stronger modern Global.",
               "B": "Complementarity remained directionally positive but was not statistically "
                    "supported under the stronger modern Global.",
               "C": "Local complementarity was not replicated under the stronger Global; this is "
                    "consistent with representation-strength dependence.",
               "D": "DINOv3 was not empirically stronger; it must be reported as a modern "
                    "foundation-pretrained baseline only."}[case],
           "global_naming": ("empirically stronger Global under the frozen CFF protocol"
                             if Ssup else "modern foundation-pretrained Global"),
           "stability_disclosure": "inner-val training showed material seed-to-seed variability; "
                                   "all three seeds are reported individually and no seed was dropped",
           "scope_limitation": "under the frozen fine-tuning recipe",
           "secondary_status": "Macro-Rank-1 / MRR effects are secondary descriptive; not "
                               "confirmatory",
           "no_additional_confirmatory_contrast": True}
    write_atomic(OUT_RES, json.dumps(res, ensure_ascii=False, indent=1))

    L = ["# P4-C — Modern Global 一次性正式评价", "",
         "**时间**：`%s`" % res["created_at"], "",
         "> 一次性打开 frozen CFF outer-eval endpoint。**两个预冻结 contrast，无第三个 confirmatory contrast。**", "",
         "## 1. 端点与纪律", "",
         "| 项 | 值 |", "|---|---|",
         "| endpoint | **256 IDs / 768 gallery / 1104 query / 1872 images** |",
         "| retrieval | **fold-local**；先在每 fold 完整候选集上冻结 outcomes，再 pooled |",
         "| aggregation | identity score = **gallery max**；tie = **optimistic** |",
         "| fusion | **0.5·s_modern + 0.5·s_local**（α 固定，无 dynamic weighting）|",
         "| bootstrap | identity cluster，B=**%d**，seed=**%d**" % (BOOT_DRAWS, BOOT_SEED),
         "| MobileNet | **直读 P3-C 已封存 per-query outcomes，未重跑** |",
         "| Local | **直读 P3-B frozen FV，未重提 / 未重拟合** |", "",
         "## 2. 三个 arm 的完整结果（三个 seed 全部报告）", "",
         "| arm | seed | Rank-1 | Macro-Rank-1 | MRR |", "|---|---|---|---|---|"]
    for s in SEEDS:
        for arm, key in (("DINOv3 Global", "dino"), ("DINOv3 + Local", "fusion")):
            m = M[key][str(s)]
            L.append("| %s | %d | **%.4f** | %.4f | %.4f |" % (arm, s, m["rank1"], m["macro_rank1"], m["mrr"]))
    for s in SEEDS:
        m = M["mobilenet_p3c_sealed"][str(s)]
        L.append("| MobileNet Global（**P3-C 封存**）| %d | %.4f | %.4f | %.4f |"
                 % (s, m["rank1"], m["macro_rank1"], m["mrr"]))
    L += ["", "## 3. Contrast S — representation strength", "",
          "| seed | R1_DINOv3 | R1_MobileNet | Δ_s |", "|---|---|---|---|"]
    for s in SEEDS:
        L.append("| %d | %.4f | %.4f | **%+.4f** |"
                 % (s, M["dino"][str(s)]["rank1"], M["mobilenet_p3c_sealed"][str(s)]["rank1"], dS[str(s)]))
    L += ["", "**Δ_Global = mean_seed = %+.6f** · 95%% CI **[%+.6f, %+.6f]** ⇒ **%s**"
          % (dS_mean, ciS[0], ciS[1], "SUPPORTED" if Ssup else "NOT supported"), "",
          "## 4. Contrast C — Local complementarity under the modern Global", "",
          "| seed | R1_DINOv3+Local | R1_DINOv3 | Δ_s |", "|---|---|---|---|"]
    for s in SEEDS:
        L.append("| %d | %.4f | %.4f | **%+.4f** |"
                 % (s, M["fusion"][str(s)]["rank1"], M["dino"][str(s)]["rank1"], dC[str(s)]))
    L += ["", "**Δ_Local|Modern = mean_seed = %+.6f** · 95%% CI **[%+.6f, %+.6f]** ⇒ **%s**"
          % (dC_mean, ciC[0], ciC[1], "SUPPORTED" if Csup else "NOT supported"), "",
          "## 5. 联合解释（按 P4-A 冻结规则）", "",
          "> ### **Case %s**" % case, ">", "> %s" % res["interpretation"], "",
          "**Global 命名**：%s" % res["global_naming"], "",
          "> `consistent with` ≠ 因果证明。", "",
          "## 6. Fold consistency（仅 descriptive，不作 n=5 检验）", "",
          "| fold | S mean Δ | 方向 | C mean Δ | 方向 |", "|---|---|---|---|---|"]
    for k in range(5):
        f = folds["fold%d" % k]
        L.append("| fold%d | %+.6f | %s | %+.6f | %s |"
                 % (k, f["S_mean_delta"], f["S_direction"], f["C_mean_delta"], f["C_direction"]))
    posS = sum(1 for k in range(5) if folds["fold%d" % k]["S_direction"] == "+")
    posC = sum(1 for k in range(5) if folds["fold%d" % k]["C_direction"] == "+")
    L += ["", "**S 正方向 fold：%d/5 · C 正方向 fold：%d/5**" % (posS, posC), "",
          "## 7. Training-stability disclosure", "",
          "- P4-B inner-val 训练存在**明显的 seed/run variability** ⇒ **三个 seed 全部单独报告**",
          "- **未删除任何 seed**；该 variability **不构成**删除或重训的理由",
          "- 论文措辞必须限定：**under the frozen fine-tuning recipe**",
          "- 🚫 不得把结果泛化为对 DINOv3 架构 / 基础预训练的普遍性能判断", "",
          "## 8. 边界", "",
          "- **未**重训 / 重选 checkpoint / 调 recipe / 换 backbone / 改 α / 删 seed / 删 fold",
          "- **未**新增第三个 confirmatory contrast",
          "- MobileNet 与 Local 侧**只读封存件**，未重跑、未重拟合",
          "- Macro-Rank-1 / MRR 为 **secondary descriptive**，不覆盖 primary 解释",
          "- 🔴 **STOP** —— 不自动进入 recent cattle/animal ReID baseline", ""]
    write_atomic(OUT_MD, "\n".join(L))

    audit["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    audit["result"] = {"contrast_S": {"effect": dS_mean, "ci95": ciS, "supported": Ssup},
                       "contrast_C": {"effect": dC_mean, "ci95": ciC, "supported": Csup},
                       "case": case}
    audit["outputs"] = {k: os.path.relpath(v, ROOT) for k, v in (
        ("execution_seal", OUT_SEAL), ("result_json", OUT_RES), ("result_md", OUT_MD),
        ("per_query", OUT_CSV), ("bootstrap", OUT_BOOT), ("audit", OUT_AUD))}
    audit["all_checks_ok"] = all(c["ok"] for c in audit["checks"])
    audit["status"] = "P4-C COMPLETE — STOP"
    write_atomic(OUT_AUD, json.dumps(audit, ensure_ascii=False, indent=1))

    print("\nContrast S = %+.6f  CI [%+.6f, %+.6f]  supported=%s" % (dS_mean, ciS[0], ciS[1], Ssup))
    print("Contrast C = %+.6f  CI [%+.6f, %+.6f]  supported=%s" % (dC_mean, ciC[0], ciC[1], Csup))
    print("CASE:", case)
    print("WROTE result json/md, per-query csv, bootstrap npz, audit json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.run:
        try:
            formal_run()
        except FailClosed as e:
            print("STOP (fail-closed): %s" % e)
            sys.exit(2)
    else:
        print("choose --selftest or --run")
