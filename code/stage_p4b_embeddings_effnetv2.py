# -*- coding: utf-8 -*-
"""P4-B — materialise inner-val and outer-eval DINOv3 Global embeddings from the selected
checkpoints, then run the alignment audit.

Deterministic inference only. This script computes NO retrieval metric on outer_eval:
no Rank-1, no Macro-R1, no MRR, no fusion, no contrast. It produces 128-D L2-normalised
embeddings plus the identity/path lists needed for the later one-off evaluation.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"D:\H2025_clean_restart"
P3A = os.path.join(ROOT, "analysis", "stage_p3a")
P3B = os.path.join(ROOT, "analysis", "stage_p3b")
P3C = os.path.join(ROOT, "analysis", "stage_p3c")
P4A = os.path.join(ROOT, "analysis", "stage_p4a")
P4B = os.path.join(ROOT, "analysis", "stage_p4b_effnetv2")
EMB = os.path.join(P4B, "embeddings")
RUNS = os.path.join(P4B, "runs")
os.makedirs(EMB, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import stage_p4b_train_effnetv2 as T                          # noqa: E402

F_REC = os.path.join(P4B, "P4B_EFFNETV2_GLOBAL_EMBEDDINGS_RECORD.json")
F_ALIGN = os.path.join(P4B, "P4B_EFFNETV2_ALIGNMENT_AUDIT.json")
F_NEST = os.path.join(P3A, "CFF_P3A2_NESTED_SPLIT.json")
F_ROI = os.path.join(ROOT, "analysis", "stage_p2r_det1",
                     "CFF_PRIMARY_ROI_COMPATIBILITY_AUDIT.json")
SEEDS = [42, 43, 44]


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(p, text):
    t = p + ".tmp"
    with open(t, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)


def main():
    T.verify_weights()
    nest = json.load(open(F_NEST, encoding="utf-8"))
    roi = json.load(open(F_ROI, encoding="utf-8"))
    role_of = {(r["canonical"], r["rel"]): r["role"] for r in roi["per_image"]}
    src_of = {}
    cache = json.load(open(os.path.join(P3B, "CFF_P3B_IMAGE_CACHE.json"), encoding="utf-8"))
    for r in cache["records"]:
        src_of[(r["canonical"], r["rel"])] = r["source"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, ev_tf = T.build_transforms()

    rec = {"schema": "h2025_p4b_global_embeddings/v1", "stage": "P4-B",
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "model": T.MODEL_ID, "embedding_dim": T.EMB,
           "source": "each run's selected best.pt; eval transform; frozen ROI",
           "no_retrieval_metric_computed": True, "folds": {}}
    align = {"schema": "h2025_p4b_alignment_audit/v1", "checks": [], "folds": {}}

    def CK(n, ok, d=""):
        align["checks"].append({"check": n, "ok": bool(ok), "detail": d})
        if not ok:
            raise SystemExit("STOP (alignment): %s -> %s" % (n, d))

    for k in range(5):
        e = nest["per_outer_fold"]["fold%d" % k]
        fold = {"inner_val": {}, "outer_eval": {}}
        for sub, ids in (("inner_val", e["inner_val"]), ("outer_eval", e["outer_eval"])):
            rad = role_of
            # Ordering must be byte-identical to the P3-B Local FV files, otherwise the two
            # arms would be misaligned. P3-B built them as "for c in ids: for r in cache[c]"
            # — i.e. grouped by identity in the split's order, cache order inside a group.
            # Sorting by (canonical, rel) here would silently break that, which the P4-C
            # selftest does catch (dinov3_seed_alignment).
            cache_by_id = {}
            for r in roi["per_image"]:
                cache_by_id.setdefault(r["canonical"], []).append(r)
            recs = []
            for c in ids:
                for r in cache_by_id.get(c, []):
                    recs.append({"canonical": r["canonical"], "rel": r["rel"],
                                 "role": rad[(r["canonical"], r["rel"])],
                                 "source": src_of[(r["canonical"], r["rel"])]})
            lm = {c: i for i, c in enumerate(sorted(ids))}
            for s in SEEDS:
                bp = os.path.join(RUNS, "fold%d_s%d" % (k, s), "best.pt")
                if not os.path.isfile(bp):
                    raise SystemExit("STOP: missing checkpoint %s" % bp)
                ck = torch.load(bp, map_location="cpu", weights_only=False)
                net = T.make_net(len(ck["class_to_idx"]), dev)
                net.load_state_dict(ck["state_dict"])
                emb = T.embed(net, recs, lm, ev_tf, dev)
                if emb.shape[1] != T.EMB or not np.isfinite(emb).all():
                    raise SystemExit("STOP: bad embedding for fold%d s%d %s" % (k, s, sub))
                fp = os.path.join(EMB, "fold%d_s%d_%s.npz" % (k, s, sub))
                tmp = fp + ".tmp.npz"
                np.savez_compressed(tmp, emb=emb.astype(np.float32),
                                    rel=np.array([r["rel"] for r in recs]),
                                    canonical=np.array([r["canonical"] for r in recs]),
                                    role=np.array([r["role"] for r in recs]))
                os.replace(tmp, fp)
                fold[sub]["s%d" % s] = {
                    "n_images": len(recs), "n_identities": len(set(r["canonical"] for r in recs)),
                    "file": os.path.relpath(fp, ROOT), "sha256": fsha(fp),
                    "best_pt_sha256": fsha(bp),
                    "emb_norm_mean": float(np.linalg.norm(emb, axis=1).mean()),
                    "n_zero_vectors": int((np.linalg.norm(emb, axis=1) < 1e-8).sum())}
                print("  fold%d s%d %-10s %s" % (k, s, sub, emb.shape), flush=True)
        rec["folds"]["fold%d" % k] = fold
        write_atomic(F_REC, json.dumps(rec, ensure_ascii=False, indent=1))

    # ---- alignment audit
    for k in range(5):
        e = nest["per_outer_fold"]["fold%d" % k]
        fv = np.load(os.path.join(P3B, "fv", "fold%d_outer_eval.npz" % k))
        fv_rel = set(map(str, fv["rel"]))
        fv_can = {}
        for r, c in zip(map(str, fv["rel"]), map(str, fv["canonical"])):
            fv_can[r] = c
        base_rel, base_can = None, None
        for s in SEEDS:
            z = np.load(os.path.join(EMB, "fold%d_s%d_outer_eval.npz" % (k, s)))
            rel = list(map(str, z["rel"]))
            can = list(map(str, z["canonical"]))
            CK("fold%d_s%d_no_duplicate_path" % (k, s), len(rel) == len(set(rel)),
               "%d paths" % len(rel))
            CK("fold%d_s%d_path_set_equals_local_fv" % (k, s), set(rel) == fv_rel,
               "|dino|=|fv|=%d" % len(fv_rel))
            CK("fold%d_s%d_identity_labels_match_fv" % (k, s),
               all(fv_can[r] == c for r, c in zip(rel, can)), "labels identical")
            if base_rel is None:
                base_rel, base_can = rel, can
            else:
                CK("fold%d_s%d_order_matches_seed42" % (k, s), rel == base_rel,
                   "identical ordering across seeds")
        CK("fold%d_outer_identity_set_matches_split" % k,
           {fv_can[r] for r in fv_rel} == set(e["outer_eval"]),
           "%d identities" % len(set(e["outer_eval"])))
        tot = align["folds"].setdefault("fold%d" % k, {})
        tot["n_gallery"] = sum(1 for r in fv_rel if
                               role_of.get((fv_can[r], r)) == "gallery")
        tot["n_query"] = len(fv_rel) - tot["n_gallery"]
        tot["n_ids"] = len(set(fv_can[r] for r in fv_rel))
        print("  align fold%d: %d ids / %d gallery / %d query"
              % (k, tot["n_ids"], tot["n_gallery"], tot["n_query"]), flush=True)

    # ---- global totals vs the frozen endpoint
    tot_ids = sum(v["n_ids"] for v in align["folds"].values())
    tot_g = sum(v["n_gallery"] for v in align["folds"].values())
    tot_q = sum(v["n_query"] for v in align["folds"].values())
    CK("totals_are_256_768_1104_1872", (tot_ids, tot_g, tot_q) == (256, 768, 1104),
       "%d / %d / %d / %d" % (tot_ids, tot_g, tot_q, tot_g + tot_q))

    # ---- P3-C / P3-B artifacts must be untouched
    res3 = json.load(open(os.path.join(P3C, "CFF_EXTERNAL_PRIMARY_EVALUATION_RESULT.json"),
                          encoding="utf-8"))
    CK("p3c_mobilenet_result_unchanged",
       abs(res3["delta_r1"] - 0.03743961352657007) < 1e-12
       and res3["evaluator_sha256"].startswith("eaec0f32"),
       "Delta_R1=%+.6f" % res3["delta_r1"])
    in_seal3 = json.load(open(os.path.join(P3B, "CFF_EXTERNAL_REID_EVALUATION_INPUT_SEAL.json"),
                              encoding="utf-8"))
    CK("p3b_local_seal_unchanged",
       in_seal3["seal_content_sha256"]
       == "62f9e2e4b03b9f1d629f08ad53b866b6e8732e6121968d8d89d7cc4b74d69162",
       in_seal3["seal_content_sha256"][:24])

    align["totals"] = {"identities": tot_ids, "gallery": tot_g, "query": tot_q,
                       "images": tot_g + tot_q}
    align["n_embedding_files"] = sum(len(v["inner_val"]) + len(v["outer_eval"])
                                     for v in rec["folds"].values())
    align["all_checks_ok"] = all(c["ok"] for c in align["checks"])
    align["no_outer_retrieval_metric_computed"] = True
    write_atomic(F_ALIGN, json.dumps(align, ensure_ascii=False, indent=1))
    write_atomic(F_REC, json.dumps(rec, ensure_ascii=False, indent=1))
    print("embedding files:", align["n_embedding_files"])
    print("WROTE", F_REC)
    print("WROTE", F_ALIGN)


if __name__ == "__main__":
    main()
