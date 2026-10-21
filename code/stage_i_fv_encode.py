#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage I —— improved Fisher Vector 编码（局部特征路线）。

方法出处（逐项对齐，不自行调参）
    ALFRE-ID / Norppa 参考实现的 **编码算法**：
      * `D:\\复现论文\\reproduction\\alfreid.py`  → Codebook.fit / fisher() / encode()
    详见 `reports/H2025_LOCAL_FV_REFERENCE_ALIGNMENT.md`。

    ⚠️ **码本容量**不再沿用参考工程的 PCA64/GMM1400。按导师裁决（2026-09-16）：
      * primary    : PCA 32 / GMM 16 / whiten **False** / FV dim **1024**
      * 容量匹配对照: PCA 64 / GMM  8 / whiten False / FV dim **1024**
      * ALFRE 参考 : PCA 64 / GMM 1400 / whiten True / FV dim 179200
                     → **禁止进入 H2 confirmatory matrix**，只能标
                       `ALFRE-reference exploratory replication`

设定位（写死）
    iFV : g1=Σ q·z, g2=Σ q·(z²−1)
          输出 [g1/(N·√w) ; g2/(N·√(2w))] 先均值后方差
          再做 signed square root（幂归一化）→ L2 → float32
          维度 = 2·K·D

零描述子处置（导师裁决 F，2026-09-16）—— **deterministic missing-local fallback**
    * 若某图 **0 个有效 local descriptor**：
        - FV = **1024 维全零**（primary 口径）
        - `local_valid = False`
      **不得**删除该图；**不得**复制 descriptor；**不得**补点。
    * 该规则是**确定性缺失兜底**，**不是** learned adaptive fusion /
      quality-aware weighting / dynamic alpha。

纪律
    * **码本只能在本工程的训练数据上拟合**（`train_pool`）。
      参考工程的负对照已量化跨域复用码本的代价，因此严禁复用海豹码本。
    * 本脚本**不计算任何检索指标**（Rank-1/mAP 属后续评估阶段），
      以免在导师批准正式评估之前产生任何性能数字。
    * 线程数固定，避免 GMM 拟合的线程相关不确定性。

用法
    # 1) 用 train_pool 的 Top64 描述子拟合 primary 码本
    python stage_i_fv_encode.py --fit-codebook --subset train_pool --role primary

    # 2) 编码 train_pool / inner_val（**不得**在本阶段编码 holdout）
    python stage_i_fv_encode.py --encode train_pool --role primary
    python stage_i_fv_encode.py --encode inner_val  --role primary
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# 参考实现同样设置：限制 OpenMP / BLAS 线程，减少拟合的非确定性
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2025_common as C

# --------------------------------------------------------------------------- #
# 角色 → 码本配置（导师裁决 D4，逐字落实）
ROLES = {
    "primary": {
        "pca": 32, "clusters": 16, "whiten": False,
        "file": "local_fv_pca32_gmm16_nowhiten_train_only.npz",
        "confirmatory": True,
        "label": "Primary Local codebook (H2 confirmatory, unique primary)",
    },
    "capacity_matched_control": {
        "pca": 64, "clusters": 8, "whiten": False,
        "file": "local_fv_pca64_gmm8_nowhiten_train_only.npz",
        "confirmatory": False,
        "label": "Secondary capacity-matched control (FV dim 1024, report-only)",
    },
    "alfre_reference": {
        "pca": 64, "clusters": 1400, "whiten": True,
        "file": "local_fv_alfreref_pca64_gmm1400_whiten.npz",
        "confirmatory": False,
        "label": ("ALFRE-reference exploratory replication "
                  "(MUST NOT enter H2 confirmatory matrix; MUST NOT affect H2 GO/STOP)"),
    },
}

GMM_REG_COVAR = 1e-6
GMM_MAX_ITER = 100
GMM_N_INIT = 1
SEED = 0                      # 参考实现 alfreid.py --seed 默认 0
FISHER_BATCH = 256            # 参考实现 fisher() 默认 batch_size

FEAT_DIR = C.ROOT / "features" / "local"
OUT_DIR = C.ROOT / "features" / "fv"
CODEBOOK_DIR = C.ROOT / "codebooks"

# 只有 fit 集合允许被用来拟合码本
ALLOWED_FIT_SUBSETS = {"train_pool"}

# 本阶段（导师裁决 J）不允许编码的集合：holdout 的一切编码须等最终批准
FORBIDDEN_ENCODE_SUBSETS_IN_STAGE_J = {"holdout", "holdout_gallery", "holdout_query"}


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fisher(x, means, variances, weights, batch_size=FISHER_BATCH):
    """improved Fisher Vector（含幂归一化 + L2）。

    与参考实现 `alfreid.py::fisher()` 逐行等价：
    工作空间 O(batch*K*D)；梯度排列为「先全部均值梯度，再全部方差梯度」。

    零描述子输入 → 返回**全零向量**（deterministic missing-local fallback；
    调用方须同时把 `local_valid` 记为 False）。
    """
    from scipy.special import logsumexp
    x = np.asarray(x, dtype=np.float64)
    k, d = means.shape
    if len(x) == 0:
        return np.zeros(2 * k * d, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != d or not np.isfinite(x).all():
        raise ValueError("Invalid descriptors for Fisher encoding")
    if np.any(variances <= 0) or np.any(weights <= 0):
        raise ValueError("GMM variances and weights must be positive")

    g1, g2 = np.zeros((k, d)), np.zeros((k, d))
    constant = (np.log(weights)
                - 0.5 * (d * np.log(2 * np.pi) + np.log(variances).sum(axis=1)))
    for start in range(0, len(x), batch_size):
        z = (x[start:start + batch_size, None, :] - means[None]) / np.sqrt(variances)[None]
        lp = constant[None] - 0.5 * (z * z).sum(axis=2)
        q = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
        g1 += np.einsum("bk,bkd->kd", q, z)
        g2 += np.einsum("bk,bkd->kd", q, z * z - 1)

    v = np.concatenate([(g1 / (len(x) * np.sqrt(weights[:, None]))).ravel(),
                        (g2 / (len(x) * np.sqrt(2 * weights[:, None]))).ravel()])
    return normalize(np.sign(v) * np.sqrt(np.abs(v))).astype(np.float32)


class Codebook:
    """PCA(+optional whiten) + 对角 GMM 码本。

    与参考实现 `alfreid.py::Codebook` 的**算法**等价；差别只在
    `whiten` 可配置（裁决要求 primary/对照取 False）。
    """

    def __init__(self):
        self.whiten = False

    def fit(self, x, dimensions=32, clusters=16, seed=SEED,
            max_iter=GMM_MAX_ITER, whiten=False):
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from threadpoolctl import threadpool_limits

        if min(x.shape) < dimensions or len(x) < clusters:
            raise ValueError(
                "Need at least max(dimensions, clusters) descriptors; got %s. "
                "Only shrink for smoke tests." % (x.shape,))
        pca = PCA(n_components=dimensions, whiten=bool(whiten), svd_solver="full")
        with threadpool_limits(limits=4):
            z = pca.fit_transform(x)
        with threadpool_limits(limits=1 if len(z) < 1024 else 4):
            gm = GaussianMixture(clusters, covariance_type="diag", random_state=seed,
                                 reg_covar=GMM_REG_COVAR, max_iter=max_iter,
                                 n_init=GMM_N_INIT).fit(z)
        self.center = pca.mean_
        self.components = pca.components_
        self.scale = np.sqrt(np.maximum(pca.explained_variance_, np.finfo(float).eps))
        self.explained_variance_ratio = pca.explained_variance_ratio_
        self.whiten = bool(whiten)
        self.means, self.variances, self.weights = gm.means_, gm.covariances_, gm.weights_
        self.converged = bool(gm.converged_)
        self.n_iter = int(gm.n_iter_)
        return self

    def transform(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.center) @ self.components.T
        if bool(self.whiten):
            z = z / self.scale
        return z

    def encode(self, x):
        return fisher(self.transform(x), self.means, self.variances, self.weights)

    def save(self, path):
        np.savez_compressed(
            path,
            **{k: v for k, v in vars(self).items() if k != "explained_variance_ratio"})

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                setattr(obj, key, data[key])
        obj.whiten = bool(obj.whiten)
        return obj


# --------------------------------------------------------------------------- #
def load_local_features(subset):
    p = FEAT_DIR / ("%s_descriptors.npz" % subset)
    if not p.is_file():
        raise SystemExit("missing Stage H features: %s (先跑 stage_h_local_descriptors.py)" % p)
    z = np.load(p, allow_pickle=True)
    paths = [str(x) for x in z["paths"]]
    descs = [np.asarray(d, dtype="float32") for d in z["descriptors"]]
    cap = int(z["num_features_cap"]) if "num_features_cap" in z.files else None
    meta = {}
    mp = FEAT_DIR / ("%s_meta.json" % subset)
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return paths, descs, meta, C.sha256_file(p), cap


def codebook_path(role):
    return CODEBOOK_DIR / ROLES[role]["file"]


def resolve_role(args):
    spec = ROLES[args.role]
    pca = args.pca if args.pca is not None else spec["pca"]
    clusters = args.clusters if args.clusters is not None else spec["clusters"]
    whiten = args.whiten if args.whiten is not None else spec["whiten"]
    return pca, clusters, bool(whiten)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-codebook", action="store_true")
    ap.add_argument("--subset", default="train_pool",
                    help="拟合码本用的 Stage H 特征子集（只允许 train_pool）")
    ap.add_argument("--encode", default=None,
                    choices=[None, "holdout", "inner_val", "holdout_gallery",
                             "holdout_query", "inner_val_gallery", "inner_val_query",
                             "train_pool"])
    ap.add_argument("--role", default="primary", choices=sorted(ROLES.keys()))
    ap.add_argument("--pca", type=int, default=None)
    ap.add_argument("--clusters", type=int, default=None)
    ap.add_argument("--whiten", dest="whiten", action="store_true", default=None)
    ap.add_argument("--no-whiten", dest="whiten", action="store_false")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-holdout-encode", action="store_true",
                    help="显式解除 J 阶段的 holdout 编码禁令（须导师最终批准后方可使用）")
    args = ap.parse_args()

    pca, clusters, whiten = resolve_role(args)
    fv_dim = 2 * pca * clusters

    C.set_seed(args.seed)
    prec = C.disable_tf32()
    C.ensure_dir(OUT_DIR)
    C.ensure_dir(CODEBOOK_DIR)
    cb_path = codebook_path(args.role)
    spec = ROLES[args.role]

    print("[role] %s  pca=%d clusters=%d whiten=%s fv_dim=%d confirmatory=%s"
          % (args.role, pca, clusters, whiten, fv_dim, spec["confirmatory"]), flush=True)

    if args.fit_codebook:
        if args.subset not in ALLOWED_FIT_SUBSETS:
            raise SystemExit(
                "拒绝：码本只允许在 %s 上拟合，收到 %s。跨域复用码本已被参考工程证明有害。"
                % (sorted(ALLOWED_FIT_SUBSETS), args.subset))
        paths, descs, hmeta, feat_sha, cap = load_local_features(args.subset)
        fit = np.concatenate([d for d in descs if len(d)], axis=0)
        eff_cap = int(np.max(cap)) if cap is not None else int(np.max([len(d) for d in descs]))
        print("[fit] subset=%s images=%d descriptors=%d (Top64 cap=%s)"
              % (args.subset, len(paths), len(fit), cap), flush=True)
        if cap is not None and cap != 64:
            raise SystemExit(
                "拒绝：码本必须拟合在 Top64 描述子上，Stage H 产物 cap=%s。" % cap)
        t0 = time.time()
        cb = Codebook().fit(fit, pca, clusters, args.seed, GMM_MAX_ITER, whiten=whiten)
        el = time.time() - t0
        cb.save(cb_path)
        prov = {
            "created_at": C.now_iso(),
            "role": args.role,
            "role_label": spec["label"],
            "enters_h2_confirmatory_matrix": bool(spec["confirmatory"]),
            "fit_subset": args.subset,
            "fit_images": len(paths),
            "fit_descriptors": int(len(fit)),
            "top64_cap": cap,
            "max_descriptors_per_image": eff_cap,
            "pca_dim": pca, "gmm_clusters": clusters,
            "whiten": bool(whiten), "fv_dim": int(fv_dim),
            "gmm_covariance": "diag", "gmm_reg_covar": GMM_REG_COVAR,
            "gmm_max_iter": GMM_MAX_ITER, "gmm_n_init": GMM_N_INIT,
            "seed": args.seed,
            "gmm_converged": cb.converged, "gmm_n_iter": cb.n_iter,
            "feature_file": str(FEAT_DIR / ("%s_descriptors.npz" % args.subset)),
            "feature_sha256": feat_sha,
            "stage_h_meta": hmeta,
            "codebook_file": str(cb_path),
            "codebook_sha256": C.sha256_file(cb_path),
            "elapsed_sec": round(el, 1),
            "precision": prec,
            "method_source": ("D:\\复现论文\\reproduction\\alfreid.py::Codebook/fisher "
                              "(algorithm only; capacity per H2025 D4 ruling)"),
            "ruling": ("D4: primary = PCA32/GMM16/whiten=False/FV1024; "
                       "capacity-matched control = PCA64/GMM8/whiten=False/FV1024; "
                       "ALFRE ref = PCA64/GMM1400/whiten=True/FV179200 "
                       "(exploratory only)"),
        }
        (OUT_DIR / ("%s_meta.json" % args.role)).write_text(
            json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[out] %s  (sha256 %s)" % (cb_path, prov["codebook_sha256"]), flush=True)
        print(json.dumps({k: prov[k] for k in
                          ("role", "fit_subset", "fit_images", "fit_descriptors",
                           "pca_dim", "gmm_clusters", "whiten", "fv_dim",
                           "gmm_converged", "gmm_n_iter", "elapsed_sec")},
                         ensure_ascii=False), flush=True)
        print("STAGE_I_CODEBOOK_OK %s" % args.role)
        return 0

    if args.encode:
        if args.encode in FORBIDDEN_ENCODE_SUBSETS_IN_STAGE_J and not args.allow_holdout_encode:
            raise SystemExit(
                "拒绝：J 阶段不得编码 holdout（%s）。\n"
                "holdout 的 FV 编码只能在导师最终批准正式评估后执行。" % args.encode)
        if not cb_path.is_file():
            raise SystemExit("missing codebook: %s (先跑 --fit-codebook --role %s)"
                             % (cb_path, args.role))
        cb = Codebook().load(cb_path)
        cb_sha = C.sha256_file(cb_path)
        paths, descs, hmeta, feat_sha, cap = load_local_features(args.encode)
        print("[encode] role=%s subset=%s images=%d (codebook %s)"
              % (args.role, args.encode, len(paths), cb_sha), flush=True)
        enc, counts, valid = [], [], []
        t0 = time.time()
        for i, d in enumerate(descs):
            v = cb.encode(d)
            n = int(len(d))
            if int(v.shape[0]) != fv_dim:
                raise RuntimeError("FV dim mismatch: got %d expected %d" % (v.shape[0], fv_dim))
            enc.append(v)
            counts.append(n)
            # deterministic missing-local fallback：0 个有效描述子 ⇒ 全零 FV + valid=False
            valid.append(n > 0)
            if (i + 1) % 25 == 0:
                print("  [%4d/%4d] %.1fs" % (i + 1, len(paths), time.time() - t0), flush=True)
        E = np.stack(enc).astype("float32")
        el = time.time() - t0
        valid = np.array(valid, dtype=bool)
        zero = ~valid
        # 断言：零描述子图的 FV 必须是**精确**全零（不得有任何非零残留）
        if zero.any() and not np.all(E[zero] == 0.0):
            raise RuntimeError("零描述子图的 FV 非全零 —— missing-local fallback 被破坏")
        if (~zero).any() and np.any(np.linalg.norm(E[~zero], axis=1) == 0.0):
            raise RuntimeError("有效描述子图的 FV 出现全零范数")
        dst = OUT_DIR / ("%s_fv_%s.npz" % (args.role, args.encode))
        np.savez_compressed(dst, paths=np.array(paths), fv=E,
                            counts=np.array(counts, "int32"),
                            local_valid=valid, role=args.role)
        meta = {
            "role": args.role, "role_label": spec["label"],
            "enters_h2_confirmatory_matrix": bool(spec["confirmatory"]),
            "subset": args.encode, "n_images": len(paths), "fv_dim": int(E.shape[1]),
            "pca_dim": pca, "gmm_clusters": clusters, "whiten": bool(whiten),
            "codebook_file": str(cb_path),
            "codebook_sha256": cb_sha,
            "feature_file": str(FEAT_DIR / ("%s_descriptors.npz" % args.encode)),
            "feature_sha256": feat_sha,
            "stage_h_cap": cap,
            "images_with_zero_descriptors": int(np.sum(zero)),
            "zero_descriptor_paths": [p for p, f in zip(paths, zero) if f],
            "missing_local_fallback": ("deterministic missing-local fallback: "
                                       "FV = all-zero, local_valid = false"),
            "n_local_valid": int(np.sum(valid)),
            "fv_finite": bool(np.isfinite(E).all()),
            "descriptor_count_min": int(np.min(counts)) if counts else 0,
            "descriptor_count_max": int(np.max(counts)) if counts else 0,
            "fv_sha256": hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest(),
            "elapsed_sec": round(el, 1),
            "output": str(dst), "output_sha256": C.sha256_file(dst),
            "created_at": C.now_iso(),
            "no_metrics_computed": True,
        }
        (OUT_DIR / ("%s_fv_%s_meta.json" % (args.role, args.encode))).write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[out] %s" % dst, flush=True)
        print(json.dumps(meta, ensure_ascii=False, indent=1), flush=True)
        print("STAGE_I_ENCODE_OK %s %s" % (args.role, args.encode))
        return 0

    ap.error("需要 --fit-codebook 或 --encode SUBSET")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
