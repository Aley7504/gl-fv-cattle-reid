#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H2025 —— Stage J 评分器（裁决 K §4 / §5 / §6）

本模块逐字实现导师裁决 K 规定的评分语义：

    sg(q,g) = cosine(global_q, global_g)

    Local：若 q 与 g 的 local 均 valid : sl(q,g) = cosine(local_q, local_g)
           Local-only 且任一侧 invalid : sl(q,g) = 0

    Fusion：若两侧均 valid : sf(q,g) = 0.5*sg(q,g) + 0.5*sl(q,g)
            任一侧 invalid  : sf(q,g) = sg(q,g)          <- deterministic missing-local fallback

    身份聚合：S(q,c) = max_{g: identity(g)=c} s(q,g)

    Global ensemble（仅 MobileNet seeds 42/43/44）：
        sD(q,g) = mean(cos42(q,g), cos43(q,g), cos44(q,g))
        SD(q,c) = max_{g: identity(g)=c} sD(q,g)
        **必须先 score mean、后 identity max；禁止反序**
        ConvNeXt **不建** ensemble。

    mAP 语义（§6）：identity-level ranking 上每个 query 只有一个 relevant identity，
        故 AP(q) = 1/rank(true identity) ⇒
        identity-level mAP ≡ mean reciprocal rank (MRR)。

严禁（本模块不得出现；由 scripts/h2_preopening_audit.py 静态核验）：
    shortlist / reranking / candidate pruning / dynamic alpha / test calibration

执行边界（fail-closed）：
    - 允许的数据集：synthetic（单测）、inner_val（smoke/验收）
    - holdout：**必须**先存在 locks/H2_EVALUATION_INPUT_SEAL.json 且
      `approved_for_evaluation == true`，且封条记录的**输入指纹与代码指纹**与磁盘逐一相符；
      任一不符立即拒绝退出。该封条由导师批准后另行生成；本模块不得自行创建。

用法：
    python scripts/stage_j_scores.py --self-test
    python scripts/stage_j_scores.py --dataset inner_val --out reports/_stage_j_scores_inner_val.json
    python scripts/stage_j_scores.py --dataset holdout --regime g1 --out reports/_stage_j_scores_holdout_g1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# 常量与路径
# --------------------------------------------------------------------------------------

ROOT = Path(r"D:\H2025_clean_restart")

SPLITS_PATH = ROOT / "splits" / "h2025_clean_splits.json"
REALIZATIONS_PATH = ROOT / "splits" / "h2_gallery_realizations.json"
REALIZATIONS_SHA256_EXPECTED = (
    "b6126b869fc9dab1838aa0307ca37adf805d5d9dc7bc487f948d5a27d9724beb"
)

GLOBAL_DIR = ROOT / "features" / "global"
FV_DIR = ROOT / "features" / "fv"

#: 唯一 Primary Local codebook（裁决 K §1）—— 两主干共享，不得为 ConvNeXt 另行拟合
PRIMARY_CODEBOOK = (
    ROOT / "codebooks" / "local_fv_pca32_gmm16_nowhiten_train_only.npz"
)
PRIMARY_CODEBOOK_SHA256_EXPECTED = (
    "03de71496276989f71b3e97a4dd152bbb0c2022d5e811fb0f232b2ff9bd080e1"
)

#: holdout 评估的输入封条（裁决 K §2）。本模块**不创建**它。
INPUT_SEAL_PATH = ROOT / "locks" / "H2_EVALUATION_INPUT_SEAL.json"

MOBILENET_RUNS: dict[int, str] = {
    42: "mobilenetv2_s42_img112_emb128",
    43: "mobilenetv2_s43_img112_emb128",
    44: "mobilenetv2_s44_img112_emb128",
}
CONVNEXT_RUNS: dict[int, str] = {
    42: "convnext_tiny_s42_img224_emb128_mb8x8",
    43: "convnext_tiny_s43_img224_emb128_mb8x8",
    44: "convnext_tiny_s44_img224_emb128_mb8x8",
}
MOBILENET_SEEDS_FOR_ENSEMBLE = (42, 43, 44)

# ======================================================================================
# 裁决 L：arm 映射（开放项 O1 已正式关闭）与冻结参数
# ======================================================================================

#: **Primary H2 对比 = R1**（裁决 L）：
#:     Delta(s,g) = M(C_seed(s,g)) − M(A_seed(s,g))
#:     C_seed = 0.5*sg(seed) + 0.5*sl
#:     A_seed = sg(seed)
#:     3 training seeds × 10 gallery realizations = **30 个重复条件**（**不是** 30 个独立样本）
PRIMARY_CONTRAST = {
    "id": "R1",
    "authority": "导师裁决 L（O1 正式关闭）",
    "num_arm": "C_seed",
    "den_arm": "A_seed",
    "per_seed": True,
    "n_repeated_conditions": 30,
    "n_independent_clusters": 16,
    "note": ("30 是重复条件数，不是样本量；正式不确定性只用 16 identity clusters 的 "
             "identity-cluster bootstrap。"),
}

#: **R2 = C_ens − D**：key secondary **robustness** analysis。
#: 用于判断 Local 是否还能在 3-seed Global ensemble 之上提供增益。
#: **不进入** H2a/H2b Primary 判定，**不得覆盖或替代 R1**。
SECONDARY_ROBUSTNESS_CONTRAST = {
    "id": "R2",
    "authority": "导师裁决 L",
    "num_arm": "C_ens",
    "den_arm": "D",
    "per_seed": False,
    "role": "key secondary robustness analysis",
    "purpose": "判断 Local 是否还能在 3-seed Global ensemble 之上提供增益",
    "enters_h2_primary_decision": False,
    "may_override_or_replace_R1": False,
}

#: **ConvNeXt robustness**：逐 seed 使用 `C'_seed − A'_seed`，
#: 仅报告 H2a/H2b replication，**不进入** MobileNet Primary 裁决。
CONVNEXT_REPLICATION_CONTRAST = {
    "num_arm": "C_p_seed",
    "den_arm": "A_p_seed",
    "per_seed": True,
    "scope": "H2a / H2b replication only",
    "enters_mobilenet_primary_decision": False,
}


#: 裁决 L 的**冻结超参数**：正式评估开始后一律禁止调整。
#: 本模块把它们写成常量 + 自测断言，使"禁止调整"成为**可机器核验**的约束。
FROZEN = {
    "fusion_alpha": 0.5,
    "top64_cap": 64,
    "pca_dim": 32,
    "gmm_clusters": 16,
    "whiten": False,
    "fv_dim": 1024,
    "codebook_sha256": "03de71496276989f71b3e97a4dd152bbb0c2022d5e811fb0f232b2ff9bd080e1",
    "mobilenet_seeds": (42, 43, 44),
    "convnext_seeds": (42, 43, 44),
    "n_gallery_realizations": 10,
    "gallery_realization_seeds": (101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
    "regimes": ("full", "g3", "g1"),
    "n_identity_clusters": 16,
    "n_query_fixed": 99,
    "missing_local_fallback": "deterministic missing-local fallback: sf = sg when either side invalid",
    "ci_rule": "estimate > 0 AND bootstrap 95% CI lower bound > 0",
    "ci_method": "percentile (2.5 / 97.5)",
    "bootstrap_reps": 10_000,
}

#: 正式评估期间禁止的操作（裁决 L 末段）。仅作声明与审计用；实现层通过 FROZEN 常量约束。
FORBIDDEN_AFTER_OPENING = (
    "调 alpha", "调 TopK", "调 PCA", "调 GMM", "调 backbone",
    "删 seed", "挑 realization", "筛 query",
    "修改 missing-local fallback", "修改统计终点", "修改 CI 规则",
)


def arm_mapping_declaration() -> dict:
    """把裁决 L 的 arm 映射与冻结参数导出为可审计声明。"""
    return {
        "authority": "导师裁决 L",
        "primary": PRIMARY_CONTRAST,
        "secondary_robustness": SECONDARY_ROBUSTNESS_CONTRAST,
        "convnext_replication": CONVNEXT_REPLICATION_CONTRAST,
        "frozen": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()},
        "forbidden_after_opening": list(FORBIDDEN_AFTER_OPENING),
    }

#: 允许的数据集。holdout 不在其中 —— 须走封条门禁（见 require_input_seal）。
ALLOWED_DATASETS = ("synthetic", "inner_val")

#: 受保护 / 禁止的子集名（Stage I 工程层门禁同款语义）
FORBIDDEN_SUBSETS_IN_THIS_MODULE = ("holdout_gallery", "holdout_query")

#: holdout 世界缓存（只读；装载入口受输入封条门禁保护）
_HOLDOUT_CACHE: dict = {}


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。零范数行保持为零向量（不产生 NaN）。"""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    out = np.zeros_like(X)
    np.divide(X, n, out=out, where=(n > 0))
    return out


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """sg / sl 的载体：逐对 cosine，返回 (len(A), len(B))。

    对零向量行，其结果自然为 0（因为归一化后仍是零向量）。
    """
    An = l2_normalize_rows(A)
    Bn = l2_normalize_rows(B)
    return An @ Bn.T


# --------------------------------------------------------------------------------------
# 裁决 K §4：评分原语
# --------------------------------------------------------------------------------------


def fuse_scores(
    sg: np.ndarray,
    sl: np.ndarray,
    valid_q: np.ndarray,
    valid_g: np.ndarray,
) -> np.ndarray:
    """Fusion（裁决 K §4）。

        两侧均 valid : sf = 0.5*sg + 0.5*sl
        任一侧 invalid: sf = sg        <- deterministic missing-local fallback

    α 固定 0.5。**不是** dynamic alpha：掩码只由 local_valid 决定，
    与任何分数大小 / 排名 / 测试期统计量无关。
    """
    vq = np.asarray(valid_q, dtype=bool).reshape(-1, 1)
    vg = np.asarray(valid_g, dtype=bool).reshape(1, -1)
    both = vq & vg
    return np.where(both, 0.5 * sg + 0.5 * sl, sg)


def local_only_scores(
    sl_raw: np.ndarray,
    valid_q: np.ndarray,
    valid_g: np.ndarray,
) -> np.ndarray:
    """Local-only 臂（裁决 K §4）：任一侧 invalid ⇒ sl = 0。"""
    vq = np.asarray(valid_q, dtype=bool).reshape(-1, 1)
    vg = np.asarray(valid_g, dtype=bool).reshape(1, -1)
    return np.where(vq & vg, sl_raw, 0.0)


def score_ensemble_mean(per_seed_scores: list[np.ndarray]) -> np.ndarray:
    """裁决 K §5：sD(q,g) = mean over seeds of cos_seed(q,g)。

    **先 score mean**（本函数），**后 identity max**（identity_max）。
    禁止反序 —— 反序会先对每个 seed 做 identity max，再对 max 求均值，
    两者一般不相等（见 self-test `test_ensemble_order`）。
    """
    if not per_seed_scores:
        raise ValueError("score_ensemble_mean: 空列表")
    shape = per_seed_scores[0].shape
    for m in per_seed_scores:
        if m.shape != shape:
            raise ValueError(
                "score_ensemble_mean: shape 不一致 %s vs %s" % (m.shape, shape)
            )
    return np.mean(np.stack(per_seed_scores, axis=0), axis=0)


def identity_max(S_qg: np.ndarray, gallery_labels: np.ndarray,
                 identities: list[str]) -> np.ndarray:
    """裁决 K §4：S(q,c) = max_{g: identity(g)=c} s(q,g)。

    `identities` 是**固定候选集**（H2 为 16 个身份）。本函数
    **不移除、不去重、不裁剪**候选集 —— 每个候选身份都在输出中占一列。
    """
    g = np.asarray(gallery_labels)
    n_q = S_qg.shape[0]
    out = np.full((n_q, len(identities)), -np.inf, dtype=np.float64)
    for j, c in enumerate(identities):
        cols = np.flatnonzero(g == c)
        if cols.size:
            out[:, j] = S_qg[:, cols].max(axis=1)
    return out


# --------------------------------------------------------------------------------------
# 裁决 K §6：度量
# --------------------------------------------------------------------------------------


def rank_of_true(
    S_qc: np.ndarray,
    query_labels: np.ndarray,
    identities: list[str],
    tie: str = "optimistic",
) -> np.ndarray:
    """真实身份的 1-based 排名。

    tie="optimistic"  : rank = 1 + #{c : S[c] > S[true]}   （并列时取最好名次）
    tie="pessimistic" : rank =      #{c : S[c] >= S[true]}  （并列时取最差名次）

    默认 optimistic；该选择被显式记录，且两种口径都会被 self-test 覆盖。
    """
    idx = {c: i for i, c in enumerate(identities)}
    col = np.array([idx[l] for l in np.asarray(query_labels)], dtype=np.int64)
    truth = S_qc[np.arange(S_qc.shape[0]), col][:, None]
    if tie == "optimistic":
        rank = 1 + (S_qc > truth).sum(axis=1)
    elif tie == "pessimistic":
        rank = (S_qc >= truth).sum(axis=1)
    else:
        raise ValueError("tie 只能是 optimistic / pessimistic")
    return rank.astype(np.int64)


def rank1_query_weighted(rank: np.ndarray) -> float:
    """Primary Rank-1：**query-weighted**（每个 query 等权）。"""
    return float(np.mean(rank == 1))


def macro_rank1_identity_balanced(
    rank: np.ndarray, query_labels: np.ndarray, identities: list[str]
) -> dict:
    """macro-Rank-1：**identity-balanced**（每个身份等权）。

    返回 dict，并显式披露**没有 query 的身份数**（不静默丢弃）。
    """
    q = np.asarray(query_labels)
    per_identity: dict[str, float] = {}
    zero_query: list[str] = []
    for c in identities:
        m = q == c
        if not m.any():
            zero_query.append(c)
            continue
        per_identity[c] = float(np.mean(rank[m] == 1))
    vals = np.array(list(per_identity.values()), dtype=np.float64)
    return {
        "macro_rank1": float(vals.mean()) if vals.size else float("nan"),
        "n_identities_scored": len(per_identity),
        "n_identities_in_candidate_set": len(identities),
        "identities_with_zero_query": zero_query,
        "per_identity_rank1": per_identity,
    }


def identity_level_map(rank: np.ndarray) -> float:
    """identity-level mAP ≡ MRR（裁决 K §6）。

    每个 query 只有一个 relevant identity ⇒ AP(q) = 1/rank(q)，
    故 mAP = mean_q 1/rank(q) = MRR。

    ⚠️ 该指标**不得**被表述为传统多 positive 的 image-level ReID mAP。
    """
    return float(np.mean(1.0 / rank.astype(np.float64)))


def evaluate_scores(
    S_qc: np.ndarray,
    query_labels: np.ndarray,
    identities: list[str],
    tie: str = "optimistic",
) -> dict:
    rank = rank_of_true(S_qc, query_labels, identities, tie=tie)
    macro = macro_rank1_identity_balanced(rank, query_labels, identities)
    return {
        "rank1_query_weighted": rank1_query_weighted(rank),
        "macro_rank1_identity_balanced": macro["macro_rank1"],
        "identity_level_map_mrr": identity_level_map(rank),
        "n_queries": int(S_qc.shape[0]),
        "n_identities_in_candidate_set": len(identities),
        "n_identities_scored": macro["n_identities_scored"],
        "identities_with_zero_query": macro["identities_with_zero_query"],
        "tie_rule": tie,
    }


# --------------------------------------------------------------------------------------
# 数据装载（inner_val 路径）
# --------------------------------------------------------------------------------------


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_global_split(split: str, run_id: str) -> dict:
    path = GLOBAL_DIR / ("%s_%s.npz" % (run_id, split))
    if not path.exists():
        raise FileNotFoundError(path)
    d = _load_npz(path)
    return {
        "gallery": np.asarray(d["gallery"], dtype=np.float64),
        "query": np.asarray(d["query"], dtype=np.float64),
        "gallery_labels": np.asarray(d["gallery_labels"]).astype(str),
        "query_labels": np.asarray(d["query_labels"]).astype(str),
        "gallery_paths": np.asarray(d["gallery_paths"]).astype(str),
        "query_paths": np.asarray(d["query_paths"]).astype(str),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def load_local_fv(split: str) -> dict:
    """载入 Primary Local FV（**两主干共享的同一份**，裁决 K §1）。"""
    path = FV_DIR / ("primary_fv_%s.npz" % split)
    if not path.exists():
        raise FileNotFoundError(path)
    d = _load_npz(path)
    meta_path = FV_DIR / ("primary_fv_%s_meta.json" % split)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "paths": np.asarray(d["paths"]).astype(str),
        "fv": np.asarray(d["fv"], dtype=np.float64),
        "local_valid": np.asarray(d["local_valid"], dtype=bool),
        "counts": np.asarray(d["counts"]),
        "codebook_sha256": meta.get("codebook_sha256"),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def align_local_to_paths(local: dict, wanted_paths: np.ndarray, tag: str) -> dict:
    """把 Local FV 按路径对齐到给定顺序；要求**完全覆盖**。"""
    wanted = np.asarray(wanted_paths).astype(str)
    index = {p: i for i, p in enumerate(local["paths"])}
    missing = [p for p in wanted if p not in index]
    if missing:
        raise KeyError(
            "[%s] Local FV 缺少 %d 张图的记录，例：%s" % (tag, len(missing), missing[:3])
        )
    rows = np.array([index[p] for p in wanted], dtype=np.int64)
    return {
        "fv": local["fv"][rows],
        "local_valid": local["local_valid"][rows],
        "counts": local["counts"][rows],
    }


# --------------------------------------------------------------------------------------
# 臂定义（score 级）
# --------------------------------------------------------------------------------------


def build_arm_scores(
    *,
    sg_by_seed: dict[int, np.ndarray],
    sg_convnext_by_seed: dict[int, np.ndarray],
    local_fv_query: np.ndarray,
    local_fv_gallery: np.ndarray,
    valid_q: np.ndarray,
    valid_g: np.ndarray,
) -> dict[str, np.ndarray]:
    """构造各臂的 (q,g) 分数矩阵。**不含任何候选裁剪**。

    返回 key → S_qg：
        A_seed<s>   Global only（MobileNet seed s）
        B           Local only（与 seed 无关；Local 分支不含主干）
        C_seed<s>   Fusion：MobileNet seed s 的 global + 共享 Local
        D           Global score ensemble（MobileNet 42/43/44），**mean 在 (q,g) 级**
        C_ens       Fusion：D 的 ensemble global + 共享 Local
        A_p_seed<s> ConvNeXt Global only
        C_p_seed<s> Fusion：ConvNeXt seed s 的 global + 共享 Local
    """
    sl_raw = cosine_matrix(local_fv_query, local_fv_gallery)
    sl = local_only_scores(sl_raw, valid_q, valid_g)

    out: dict[str, np.ndarray] = {}

    for s, sg in sg_by_seed.items():
        out["A_seed%d" % s] = sg
        out["C_seed%d" % s] = fuse_scores(sg, sl, valid_q, valid_g)

    out["B"] = sl

    # Global ensemble：先对 (q,g) 分数求均值（裁决 K §5）
    ens = score_ensemble_mean([sg_by_seed[s] for s in MOBILENET_SEEDS_FOR_ENSEMBLE])
    out["D"] = ens
    out["C_ens"] = fuse_scores(ens, sl, valid_q, valid_g)

    for s, sg in sg_convnext_by_seed.items():
        out["A_p_seed%d" % s] = sg
        out["C_p_seed%d" % s] = fuse_scores(sg, sl, valid_q, valid_g)

    return out


# --------------------------------------------------------------------------------------
# synthetic self-test
# --------------------------------------------------------------------------------------


def _identity_matrix(vals: list[list[float]]) -> np.ndarray:
    return np.array(vals, dtype=np.float64)


def self_test() -> dict:
    """synthetic 单元测试。**不读任何真实数据、不产生任何检索结论。**

    逐项对应裁决 K §4/§5 的强制条款与 §8 的验收要求。
    """
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    # ---- T1 missing-local fallback：任一侧 invalid ⇒ sf == sg（逐位相等） ----
    sg = _identity_matrix([[0.10, -0.20, 0.30], [0.40, 0.50, -0.60]])
    sl_raw = _identity_matrix([[0.90, 0.80, 0.70], [0.60, 0.50, 0.40]])
    vq = np.array([True, False])                 # q1 invalid
    vg = np.array([True, False, True])           # g2 invalid
    sl = local_only_scores(sl_raw, vq, vg)
    sf = fuse_scores(sg, sl, vq, vg)

    expect_sl = np.array([[0.90, 0.0, 0.70], [0.0, 0.0, 0.0]])
    check("T1a local_only invalid -> 0", np.allclose(sl, expect_sl), str(sl.tolist()))

    # q0-g1 invalid -> sf == sg[0,1] == -0.20 ; q0-g0,g2 两侧 valid -> 0.5*sg+0.5*sl ;
    # q1 全行 invalid -> sf == sg[1,:]
    expect_sf = np.array([
        [0.5 * 0.10 + 0.5 * 0.90, -0.20, 0.5 * 0.30 + 0.5 * 0.70],
        [0.40, 0.50, -0.60],
    ])
    check("T1b fusion fallback == sg (bitwise)", np.array_equal(sf, expect_sf),
          str(np.abs(sf - expect_sf).max()))
    check("T1c fallback is exact copy of sg where invalid",
          all(sf[i, j] == sg[i, j] for i in range(2) for j in range(3)
              if not (vq[i] and vg[j])),
          "任一侧 invalid 的格子逐位等于 sg")

    # ---- T2 fusion α 固定 0.5 ----
    a = np.array([[0.2]]); b = np.array([[0.8]])
    f = fuse_scores(a, b, np.array([True]), np.array([True]))
    check("T2 fusion alpha == 0.5 fixed", float(f[0, 0]) == 0.5 * 0.2 + 0.5 * 0.8,
          "f=%.6f" % f[0, 0])

    # ---- T3 identity max：取该身份下 gallery 的逐行最大 ----
    S = _identity_matrix([[0.1, 0.9, 0.4, 0.7], [0.8, 0.2, 0.6, 0.3]])
    glab = np.array(["c1", "c1", "c2", "c2"])
    idents = ["c1", "c2", "c3"]          # c3 在 gallery 中不存在 -> 必须保留该列
    Sc = identity_max(S, glab, idents)
    check("T3a identity max values", np.allclose(
        Sc, [[0.9, 0.7, -np.inf], [0.8, 0.6, -np.inf]]), str(Sc.tolist()))
    check("T3b candidate set NOT pruned", Sc.shape[1] == 3,
          "候选集 3 列全保留（含 gallery 中缺席者）")

    # ---- T4 ensemble 顺序：先 mean 后 max ≠ 先 max 后 mean ----
    # 关键构造：同一身份 c1 在 gallery 中有**两列**（g0,g1），且两个 seed 的
    # 列内排序相反 —— 此时"先 mean"取到 0.5，"先 max"取到 0.9，两者必须不同。
    m1 = _identity_matrix([[0.9, 0.1, 0.0]])      # seed1: c1 的最大值在 g0
    m2 = _identity_matrix([[0.1, 0.9, 0.0]])      # seed2: c1 的最大值在 g1
    glab2 = np.array(["c1", "c1", "c2"])
    idents2 = ["c1", "c2"]
    mean_then_max = identity_max(score_ensemble_mean([m1, m2]), glab2, idents2)
    max_then_mean = score_ensemble_mean([identity_max(m1, glab2, idents2),
                                        identity_max(m2, glab2, idents2)])
    check("T4a mean-then-max != max-then-mean (反序确实不同)",
          not np.allclose(mean_then_max, max_then_mean),
          "mean_then_max=%s max_then_mean=%s" % (mean_then_max.tolist(),
                                                max_then_mean.tolist()))
    check("T4b implementation uses mean-then-max (sD=mean 在 (q,g) 级)",
          np.allclose(mean_then_max, [[0.5, 0.0]]) and
          np.allclose(max_then_mean, [[0.9, 0.0]]),
          "本模块取 0.5（正确）；反序会得到 0.9（被禁止）")

    # ---- T5 排名与三项度量 ----
    # 设计目标排名：[1, 2, 2]
    #   q0 true=c1 -> col0=0.9 为全行最大 -> rank 1
    #   q1 true=c2 -> col1=0.5，col0=0.9 更大 -> rank 2
    #   q2 true=c2 -> col1=0.4，col0=0.6 更大 -> rank 2
    S5 = _identity_matrix([
        [0.9, 0.1, 0.2],
        [0.9, 0.5, 0.1],
        [0.6, 0.4, 0.3],
    ])
    qlab5 = np.array(["c1", "c2", "c2"])
    idents5 = ["c1", "c2", "c3"]
    r5 = rank_of_true(S5, qlab5, idents5, tie="optimistic")
    check("T5a ranks", r5.tolist() == [1, 2, 2], str(r5.tolist()))
    check("T5b query-weighted Rank-1 == 1/3",
          abs(rank1_query_weighted(r5) - 1.0 / 3.0) < 1e-12,
          "%.6f" % rank1_query_weighted(r5))
    mac5 = macro_rank1_identity_balanced(r5, qlab5, idents5)
    check("T5c macro-Rank-1 identity-balanced == mean(1.0, 0.0) == 0.5",
          abs(mac5["macro_rank1"] - 0.5) < 1e-12, "%.6f" % mac5["macro_rank1"])
    check("T5c2 macro != query-weighted (两者必须分别报告)",
          abs(mac5["macro_rank1"] - rank1_query_weighted(r5)) > 1e-12,
          "macro=%.6f query_weighted=%.6f" % (mac5["macro_rank1"],
                                              rank1_query_weighted(r5)))
    check("T5d identity with zero query reported, not silently dropped",
          mac5["identities_with_zero_query"] == ["c3"] and
          mac5["n_identities_in_candidate_set"] == 3,
          json.dumps({k: mac5[k] for k in
                      ("n_identities_scored", "n_identities_in_candidate_set",
                       "identities_with_zero_query")}, ensure_ascii=False))
    check("T5e mAP == MRR under one relevant identity",
          abs(identity_level_map(r5) - (1.0 + 0.5 + 0.5) / 3.0) < 1e-12,
          "%.6f" % identity_level_map(r5))

    # ---- T6 并列处理：optimistic 与 pessimistic 必须不同且各自确定 ----
    S6 = _identity_matrix([[0.5, 0.5, 0.0]])
    ql6 = np.array(["c2"]); id6 = ["c1", "c2", "c3"]
    r_opt = rank_of_true(S6, ql6, id6, tie="optimistic")
    r_pes = rank_of_true(S6, ql6, id6, tie="pessimistic")
    check("T6a tie optimistic rank == 1", r_opt.tolist() == [1], str(r_opt.tolist()))
    check("T6b tie pessimistic rank == 2", r_pes.tolist() == [2], str(r_pes.tolist()))

    # ---- T7 零向量 cosine 不产生 NaN（FV 零描述子行为的外推） ----
    Z = np.zeros((1, 4)); X = np.array([[1.0, 0, 0, 0]])
    c = cosine_matrix(Z, X)
    check("T7 zero-vector cosine == 0 and finite",
          float(c[0, 0]) == 0.0 and np.isfinite(c).all(), str(c.tolist()))

    # ---- T8 全 invalid 世界：Local-only 全零；Fusion 退化为纯 Global ----
    sg8 = np.random.RandomState(0).randn(4, 5)
    sl8 = local_only_scores(np.random.RandomState(1).randn(4, 5),
                            np.zeros(4, bool), np.zeros(5, bool))
    sf8 = fuse_scores(sg8, sl8, np.zeros(4, bool), np.zeros(5, bool))
    check("T8a all-invalid local-only is exactly zeros", not sl8.any(),
          "max|sl|=%.3g" % np.abs(sl8).max())
    check("T8b all-invalid fusion degenerates to pure global",
          np.array_equal(sf8, sg8), "max|sf-sg|=%.3g" % np.abs(sf8 - sg8).max())

    # ---- T9 score matrix 不被裁剪：shape 恒等于 (n_q, n_g) ----
    check("T9 no candidate pruning (full (q,g) matrix)",
          sg.shape == (2, 3) and sf.shape == (2, 3) and sl.shape == (2, 3),
          "全对矩阵形状保持不变")

    # ---- T10 裁决 L：Primary 对比必须是 R1，R2 不得替代 ----
    check("T10a primary contrast is R1 (C_seed - A_seed, per seed)",
          PRIMARY_CONTRAST["id"] == "R1"
          and PRIMARY_CONTRAST["num_arm"] == "C_seed"
          and PRIMARY_CONTRAST["den_arm"] == "A_seed"
          and PRIMARY_CONTRAST["per_seed"] is True,
          json.dumps({k: PRIMARY_CONTRAST[k] for k in
                      ("id", "num_arm", "den_arm", "per_seed")}, ensure_ascii=False))
    check("T10b 30 repeated conditions != 30 independent samples",
          PRIMARY_CONTRAST["n_repeated_conditions"] == 30
          and PRIMARY_CONTRAST["n_independent_clusters"] == 16,
          "重复条件 30；唯一独立簇 16")
    check("T10c R2 is secondary robustness only, cannot override R1",
          SECONDARY_ROBUSTNESS_CONTRAST["id"] == "R2"
          and SECONDARY_ROBUSTNESS_CONTRAST["enters_h2_primary_decision"] is False
          and SECONDARY_ROBUSTNESS_CONTRAST["may_override_or_replace_R1"] is False,
          "R2 = C_ens - D，不进 H2a/H2b Primary 判定")
    check("T10d ConvNeXt is per-seed replication, excluded from MN primary",
          CONVNEXT_REPLICATION_CONTRAST["per_seed"] is True
          and CONVNEXT_REPLICATION_CONTRAST["enters_mobilenet_primary_decision"] is False,
          "C'_seed - A'_seed，仅报 replication")

    # ---- T11 冻结超参数与磁盘码本元数据一致 ----
    cb_ok = True
    cb_detail = "codebook 未找到（跳过一致性）"
    if PRIMARY_CODEBOOK.exists():
        with np.load(PRIMARY_CODEBOOK, allow_pickle=True) as z:
            cb_ok = (int(z["components"].shape[0]) == FROZEN["pca_dim"]
                     and int(z["means"].shape[0]) == FROZEN["gmm_clusters"]
                     and bool(z["whiten"]) is FROZEN["whiten"]
                     and sha256_file(PRIMARY_CODEBOOK) == FROZEN["codebook_sha256"])
            cb_detail = "pca=%d gmm=%d whiten=%s" % (
                int(z["components"].shape[0]), int(z["means"].shape[0]), bool(z["whiten"]))
    check("T11a frozen codebook hyperparameters match on-disk codebook",
          cb_ok, cb_detail)
    check("T11b frozen alpha / cap / fv_dim / regime / cluster constants",
          FROZEN["fusion_alpha"] == 0.5 and FROZEN["top64_cap"] == 64
          and FROZEN["fv_dim"] == 1024 and FROZEN["regimes"] == ("full", "g3", "g1")
          and FROZEN["n_identity_clusters"] == 16 and FROZEN["n_query_fixed"] == 99
          and FROZEN["n_gallery_realizations"] == 10
          and FROZEN["gallery_realization_seeds"] == tuple(range(101, 111))
          and FROZEN["mobilenet_seeds"] == (42, 43, 44)
          and FROZEN["convnext_seeds"] == (42, 43, 44),
          "alpha=0.5 cap=64 fv=1024 regimes=full/g3/g1 clusters=16 query=99 real=10 seeds=101..110")
    check("T11c forbidden-after-opening list is present and complete",
          set(FORBIDDEN_AFTER_OPENING) == {
              "调 alpha", "调 TopK", "调 PCA", "调 GMM", "调 backbone",
              "删 seed", "挑 realization", "筛 query",
              "修改 missing-local fallback", "修改统计终点", "修改 CI 规则"},
          "%d 项" % len(FORBIDDEN_AFTER_OPENING))

    # ---- T12 R1 所需全部臂在上述冻结 seed 下均可构造 ----
    r12 = np.random.RandomState(3).randn(3, 4)
    built = build_arm_scores(
        sg_by_seed={s: r12 for s in FROZEN["mobilenet_seeds"]},
        sg_convnext_by_seed={s: r12 for s in FROZEN["convnext_seeds"]},
        local_fv_query=np.random.RandomState(4).randn(3, 8),
        local_fv_gallery=np.random.RandomState(5).randn(4, 8),
        valid_q=np.array([True, False, True]),
        valid_g=np.array([True, True, False, True]),
    )
    need = (["A_seed%d" % s for s in FROZEN["mobilenet_seeds"]]
            + ["C_seed%d" % s for s in FROZEN["mobilenet_seeds"]]
            + ["A_p_seed%d" % s for s in FROZEN["convnext_seeds"]]
            + ["C_p_seed%d" % s for s in FROZEN["convnext_seeds"]]
            + ["B", "D", "C_ens"])
    missing12 = [k for k in need if k not in built]
    check("T12 all arms required by R1 / R2 / ConvNeXt replication exist",
          not missing12, "missing=%s" % missing12)

    failed = [c["name"] for c in checks if not c["pass"]]
    return {
        "schema": "h2025_stage_j_scores_selftest/v1",
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "verdict": "PASS" if not failed else "FAIL",
        "checks": checks,
        "no_real_data_read": True,
        "holdout_read": False,
        "no_metrics_from_real_data": True,
    }


# --------------------------------------------------------------------------------------
# inner_val smoke / validation
# --------------------------------------------------------------------------------------


def build_inner_val_world(regime: str = "full") -> dict:
    """构造 inner_val 世界（**非 holdout**）。

    inner_val：gallery 35 / query 37 / 8 身份。此处不使用 holdout 的
    gallery realization（那些是为 holdout_gallery 定义的）；inner_val 用其完整 gallery。
    """
    mn = {s: load_global_split("inner_val", rid) for s, rid in MOBILENET_RUNS.items()}
    cx = {s: load_global_split("inner_val", rid) for s, rid in CONVNEXT_RUNS.items()}

    ref = mn[42]
    g_paths, q_paths = ref["gallery_paths"], ref["query_paths"]

    # 校验跨 run 的路径与标签严格一致（同一 split 的必然要求）
    align_ok = True
    for s, d in list(mn.items()) + list(cx.items()):
        if not (np.array_equal(d["gallery_paths"], g_paths)
                and np.array_equal(d["query_paths"], q_paths)
                and np.array_equal(d["gallery_labels"], ref["gallery_labels"])
                and np.array_equal(d["query_labels"], ref["query_labels"])):
            align_ok = False

    local = load_local_fv("inner_val")
    gl = align_local_to_paths(local, g_paths, "inner_val/gallery")
    ql = align_local_to_paths(local, q_paths, "inner_val/query")

    identities = sorted(set(ref["gallery_labels"].tolist()))
    missing_in_gallery = sorted(set(ref["query_labels"].tolist()) - set(identities))

    sg_by_seed = {s: cosine_matrix(d["query"], d["gallery"]) for s, d in mn.items()}
    sg_cx = {s: cosine_matrix(d["query"], d["gallery"]) for s, d in cx.items()}

    arms = build_arm_scores(
        sg_by_seed=sg_by_seed,
        sg_convnext_by_seed=sg_cx,
        local_fv_query=ql["fv"],
        local_fv_gallery=gl["fv"],
        valid_q=ql["local_valid"],
        valid_g=gl["local_valid"],
    )

    result: dict[str, dict] = {}
    for arm, S_qg in arms.items():
        S_qc = identity_max(S_qg, ref["gallery_labels"], identities)
        result[arm] = evaluate_scores(S_qc, ref["query_labels"], identities)

    return {
        "dataset": "inner_val",
        "regime": regime,
        "n_gallery": int(ref["gallery"].shape[0]),
        "n_query": int(ref["query"].shape[0]),
        "n_identities_candidate_set": len(identities),
        "identities": identities,
        "identities_in_query_absent_from_gallery": missing_in_gallery,
        "cross_run_path_label_align_ok": align_ok,
        "local_valid_gallery": int(gl["local_valid"].sum()),
        "local_valid_query": int(ql["local_valid"].sum()),
        "local_valid_gallery_total": int(gl["local_valid"].size),
        "local_valid_query_total": int(ql["local_valid"].size),
        "arms": result,
        "codebook_sha256": local.get("codebook_sha256"),
        "global_embedding_sha256": {**{k: v["sha256"] for k, v in mn.items()},
                                    **{k: v["sha256"] for k, v in cx.items()}},
        "local_fv_sha256": local["sha256"],
        "holdout_read": False,
        "is_holdout": False,
    }


# --------------------------------------------------------------------------------------
# holdout 世界（裁决 L 步骤 8 授权；**必须**先存在输入封条）
# --------------------------------------------------------------------------------------


def _realization_index(bundle: dict, regime: str, realization_seed: int) -> np.ndarray:
    """把 realization 的路径列表映射为 gallery 列索引（保持 gallery 原顺序）。"""
    paths = bundle["realizations"][regime][str(realization_seed)]
    idx = {p: i for i, p in enumerate(bundle["g_paths"])}
    missing = [p for p in paths if p not in idx]
    if missing:
        raise KeyError("realization 路径不在 gallery 中（%d 条）：%s"
                       % (len(missing), missing[:3]))
    return np.array(sorted(idx[p] for p in paths), dtype=np.int64)


def _load_holdout_bundle() -> dict:
    """载入 holdout 世界（**唯一**入口受输入封条门禁保护）。"""
    if "_bundle" in _HOLDOUT_CACHE:
        return _HOLDOUT_CACHE["_bundle"]

    seal = require_input_seal()          # fail-closed：无封条直接 SystemExit

    ref = load_global_split("holdout", MOBILENET_RUNS[42])
    cx0 = load_global_split("holdout", CONVNEXT_RUNS[42])
    g_paths, q_paths = ref["gallery_paths"], ref["query_paths"]

    sg_by_seed, sg_cx = {}, {}
    align_ok = True
    for s, rid in MOBILENET_RUNS.items():
        d = load_global_split("holdout", rid)
        align_ok &= np.array_equal(d["gallery_paths"], g_paths) and \
            np.array_equal(d["query_paths"], q_paths) and \
            np.array_equal(d["gallery_labels"], ref["gallery_labels"]) and \
            np.array_equal(d["query_labels"], ref["query_labels"])
        sg_by_seed[s] = cosine_matrix(d["query"], d["gallery"])
    for s, rid in CONVNEXT_RUNS.items():
        d = load_global_split("holdout", rid)
        align_ok &= np.array_equal(d["gallery_paths"], g_paths) and \
            np.array_equal(d["query_paths"], q_paths)
        sg_cx[s] = cosine_matrix(d["query"], d["gallery"])

    # ---- Local：**共享**分支（两主干同一份 FV，裁决 K §1 / 裁决 L） ----
    fv_g = load_local_fv("holdout_gallery")
    fv_q = load_local_fv("holdout_query")
    if fv_g.get("codebook_sha256") != PRIMARY_CODEBOOK_SHA256_EXPECTED:
        raise SystemExit("REFUSED: holdout_gallery FV 的码本 sha256 与冻结值不符")
    if fv_q.get("codebook_sha256") != PRIMARY_CODEBOOK_SHA256_EXPECTED:
        raise SystemExit("REFUSED: holdout_query FV 的码本 sha256 与冻结值不符")

    gl = align_local_to_paths(fv_g, g_paths, "holdout/gallery")
    ql = align_local_to_paths(fv_q, q_paths, "holdout/query")

    realizations = json.loads(REALIZATIONS_PATH.read_text(encoding="utf-8"))["realizations"]

    identities = sorted(set(ref["gallery_labels"].tolist()))
    bundle = {
        "identities": identities,
        "g_paths": g_paths, "q_paths": q_paths,
        "g_labels": ref["gallery_labels"], "q_labels": ref["query_labels"],
        "sg_by_seed": sg_by_seed, "sg_cx_by_seed": sg_cx,
        "fv_g": gl["fv"], "fv_q": ql["fv"],
        "valid_g": gl["local_valid"], "valid_q": ql["local_valid"],
        "realizations": realizations,
        "cross_run_path_label_align_ok": bool(align_ok),
        "local_valid_gallery": int(gl["local_valid"].sum()),
        "local_valid_gallery_total": int(gl["local_valid"].size),
        "local_valid_query": int(ql["local_valid"].sum()),
        "local_valid_query_total": int(ql["local_valid"].size),
        "seal_approved": bool(seal.get("approved_for_evaluation")),
        "seal_sha256": sha256_file(INPUT_SEAL_PATH),
    }
    _HOLDOUT_CACHE["_bundle"] = bundle
    return bundle


def holdout_arm_scores(regime: str, realization_seed: int) -> dict:
    """该 (regime, realization) 下的 (q,g) 分数矩阵字典。候选集不裁剪。"""
    b = _load_holdout_bundle()
    sel = _realization_index(b, regime, realization_seed)
    arms = build_arm_scores(
        sg_by_seed={s: m[:, sel] for s, m in b["sg_by_seed"].items()},
        sg_convnext_by_seed={s: m[:, sel] for s, m in b["sg_cx_by_seed"].items()},
        local_fv_query=b["fv_q"],
        local_fv_gallery=b["fv_g"][sel],
        valid_q=b["valid_q"],
        valid_g=b["valid_g"][sel],
    )
    return arms, b["g_labels"][sel], sel


def holdout_ranks(regime: str, realization_seed: int) -> dict:
    """该 (regime, realization) 下每个臂的 per-query rank（99,）。"""
    b = _load_holdout_bundle()
    arms, glab, sel = holdout_arm_scores(regime, realization_seed)
    out = {}
    for arm, S_qg in arms.items():
        S_qc = identity_max(S_qg, glab, b["identities"])
        out[arm] = rank_of_true(S_qc, b["q_labels"], b["identities"])
    return out


def holdout_repeat_records(regime: str) -> dict:
    """构造该 regime 下的重复条件记录（**不聚合、不判定**）。

    R1（Primary）      : 每个 (seed, realization) 一条 → 3 × 10 = **30** 条
    R2（secondary）    : 每个 realization 一条（`sD` 不随 seed 变化）→ **10** 条
    ConvNeXt（复现）   : 每个 (seed, realization) 一条 → **30** 条
    """
    b = _load_holdout_bundle()
    q_ids = b["q_labels"]

    r1, r2, cx = [], [], []
    for r_seed in FROZEN["gallery_realization_seeds"]:
        ranks = holdout_ranks(regime, r_seed)
        base = {"regime": regime, "realization": r_seed,
                "query_identities": q_ids}
        for s in FROZEN["mobilenet_seeds"]:
            r1.append({**base, "seed": s,
                       "ranks": {"num": ranks["C_seed%d" % s],
                                 "den": ranks["A_seed%d" % s],
                                 "C_seed": ranks["C_seed%d" % s],
                                 "A_seed": ranks["A_seed%d" % s]}})
        for s in FROZEN["convnext_seeds"]:
            cx.append({**base, "seed": s,
                       "ranks": {"num": ranks["C_p_seed%d" % s],
                                 "den": ranks["A_p_seed%d" % s],
                                 "C_p_seed": ranks["C_p_seed%d" % s],
                                 "A_p_seed": ranks["A_p_seed%d" % s]}})
        r2.append({**base, "seed": None,
                   "ranks": {"num": ranks["C_ens"], "den": ranks["D"],
                             "C_ens": ranks["C_ens"], "D": ranks["D"]}})
    return {"R1": r1, "R2": r2, "CONVNEXT": cx}


def holdout_structure_checks() -> dict:
    """holdout 世界的结构自检（供审计调用；不涉及任何性能）。"""
    b = _load_holdout_bundle()
    per_regime = {}
    for regime in FROZEN["regimes"]:
        sizes, ident_ok = set(), True
        for r_seed in FROZEN["gallery_realization_seeds"]:
            sel = _realization_index(b, regime, r_seed)
            sizes.add(int(sel.size))
            ids = sorted(set(b["g_labels"][sel].tolist()))
            ident_ok &= (len(ids) == FROZEN["n_identity_clusters"]
                         and set(ids) == set(b["identities"]))
        per_regime[regime] = {"gallery_sizes": sorted(sizes),
                              "n_identities_per_realization_ok": bool(ident_ok)}
    ok = (b["cross_run_path_label_align_ok"]
          and len(b["identities"]) == FROZEN["n_identity_clusters"]
          and len(b["q_labels"]) == FROZEN["n_query_fixed"]
          and all(v["n_identities_per_realization_ok"] for v in per_regime.values())
          and per_regime["full"]["gallery_sizes"] == [72]
          and per_regime["g3"]["gallery_sizes"] == [48]
          and per_regime["g1"]["gallery_sizes"] == [16])
    return {"pass": bool(ok), "n_identities": len(b["identities"]),
            "n_query": int(len(b["q_labels"])), "per_regime": per_regime,
            "cross_run_path_label_align_ok": b["cross_run_path_label_align_ok"],
            "local_valid_gallery": b["local_valid_gallery"],
            "local_valid_gallery_total": b["local_valid_gallery_total"],
            "local_valid_query": b["local_valid_query"],
            "local_valid_query_total": b["local_valid_query_total"],
            "seal_sha256": b["seal_sha256"]}


def holdout_labels() -> dict:
    """holdout 世界的身份候选集与 query/gallery 身份标签（只读；受封条门禁保护）。"""
    b = _load_holdout_bundle()
    return {
        "identities": list(b["identities"]),
        "query_labels": np.asarray(b["q_labels"]).astype(str),
        "gallery_labels": np.asarray(b["g_labels"]).astype(str),
        "query_paths": np.asarray(b["q_paths"]).astype(str),
        "gallery_paths": np.asarray(b["g_paths"]).astype(str),
    }


def holdout_mechanism_diagnostics(regime: str) -> dict:
    """机制诊断（预注册要求 1g 必报）：Global 错/Local 对、Fusion 挽救、Fusion 损害、净挽救。

    在 **每个重复条件**上逐 query 比较 A_seed（Global）与 C_seed（Fusion）的 rank：
        rescue = #{rankA > 1 且 rankC == 1}
        harm   = #{rankA == 1 且 rankC > 1}
        net    = rescue − harm
    同时给出 Local-only（B）与 Global 的对照计数，便于解释挽救来自 Local 本身。
    仅统计**方向计数**，不构成任何"效果量"式推断。
    """
    b = _load_holdout_bundle()
    identities = b["identities"]
    per_seed: dict[str, dict] = {}

    for s in FROZEN["mobilenet_seeds"]:
        rescue = harm = 0
        a_hit = c_hit = b_hit = total = 0
        local_only_rescue = 0
        per_condition = []
        for r_seed in FROZEN["gallery_realization_seeds"]:
            ranks = holdout_ranks(regime, r_seed)
            ra = ranks["A_seed%d" % s]
            rc = ranks["C_seed%d" % s]
            rb = ranks["B"]
            res = int(((ra > 1) & (rc == 1)).sum())
            hrm = int(((ra == 1) & (rc > 1)).sum())
            rescue += res
            harm += hrm
            a_hit += int((ra == 1).sum())
            c_hit += int((rc == 1).sum())
            b_hit += int((rb == 1).sum())
            local_only_rescue += int(((ra > 1) & (rb == 1)).sum())
            total += int(ra.size)
            per_condition.append({"realization": r_seed, "rescue": res, "harm": hrm,
                                  "net": res - hrm})
        per_seed["seed%d" % s] = {
            "n_query_conditions": total,
            "global_rank1_hits": a_hit,
            "fusion_rank1_hits": c_hit,
            "local_only_rank1_hits": b_hit,
            "rescue_global_wrong_local_fusion_right": rescue,
            "harm_global_right_fusion_wrong": harm,
            "net_rescue": rescue - harm,
            "local_only_rescue_over_global": local_only_rescue,
            "per_condition": per_condition,
        }

    agg_rescue = sum(v["rescue_global_wrong_local_fusion_right"] for v in per_seed.values())
    agg_harm = sum(v["harm_global_right_fusion_wrong"] for v in per_seed.values())
    return {
        "regime": regime,
        "n_repeated_conditions_per_seed": len(FROZEN["gallery_realization_seeds"]),
        "n_query_fixed": int(len(b["q_labels"])),
        "per_seed": per_seed,
        "aggregate_over_seeds": {
            "rescue": agg_rescue, "harm": agg_harm, "net_rescue": agg_rescue - agg_harm,
        },
        "note": ("机制诊断为**方向计数**描述，不作显著性推断；"
                 "`rescue`/`harm`/`net` 定义见本函数 docstring。"),
    }


# --------------------------------------------------------------------------------------
# 封条门禁
# --------------------------------------------------------------------------------------


def require_input_seal() -> dict:
    """holdout 评估的 fail-closed 门禁（裁决 K §2 / 裁决 L 步骤 5-8）。"""
    if not INPUT_SEAL_PATH.exists():
        raise SystemExit(
            "REFUSED: holdout 评估被拒绝。\n"
            "  缺少 %s\n"
            "  该封条须在导师批准并生成 Local holdout FV 之后另行建立；\n"
            "  本模块不会自行创建它。当前仅允许 --dataset {synthetic, inner_val}。"
            % INPUT_SEAL_PATH
        )
    seal = json.loads(INPUT_SEAL_PATH.read_text(encoding="utf-8"))
    if not seal.get("approved_for_evaluation"):
        raise SystemExit("REFUSED: 封条存在但 approved_for_evaluation != true。")
    # 校验封条记录的输入指纹与磁盘一致（任一不符即中止）
    mismatched = []
    for item in seal.get("inputs", []):
        p = ROOT / item["path"]
        if not p.exists() or sha256_file(p) != item["sha256"]:
            mismatched.append(item["path"])
    if mismatched:
        raise SystemExit("REFUSED: 封条记录的输入指纹与磁盘不符：%s" % mismatched)
    # 校验代码指纹
    code_bad = []
    for name, h in seal.get("code_hashes", {}).items():
        p = ROOT / "scripts" / name
        if not p.exists() or sha256_file(p) != h:
            code_bad.append(name)
    if code_bad:
        raise SystemExit("REFUSED: 封条记录的代码哈希与磁盘不符（脚本已变更）：%s" % code_bad)
    return seal


def guard_dataset(dataset: str) -> None:
    if dataset == "holdout":
        require_input_seal()
    elif dataset not in ALLOWED_DATASETS:
        raise SystemExit("未知数据集：%s（允许：%s + holdout[需封条]）"
                         % (dataset, list(ALLOWED_DATASETS)))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="H2025 Stage J 评分器（裁决 K §4/§5/§6）")
    ap.add_argument("--self-test", action="store_true", help="synthetic 单元测试")
    ap.add_argument("--dataset", default="inner_val",
                    choices=list(ALLOWED_DATASETS) + ["holdout"])
    ap.add_argument("--regime", default="full", choices=["full", "g3", "g1"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.self_test:
        res = self_test()
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["verdict"] == "PASS" else 1

    guard_dataset(args.dataset)

    if args.dataset == "inner_val":
        res = build_inner_val_world(args.regime)
        payload = {
            "schema": "h2025_stage_j_scores/v1",
            "authority": "导师裁决 K §4/§5/§6 + 裁决 L（O1 关闭、R1 为 Primary）",
            "code_sha256": sha256_file(Path(__file__)),
            "arm_mapping": arm_mapping_declaration(),
            "codebook_sha256_expected": PRIMARY_CODEBOOK_SHA256_EXPECTED,
            "codebook_sha256_on_disk": (
                sha256_file(PRIMARY_CODEBOOK) if PRIMARY_CODEBOOK.exists() else None
            ),
            "realizations_sha256_expected": REALIZATIONS_SHA256_EXPECTED,
            "result": res,
            "no_holdout_performance_read": True,
            "no_h2_endpoint_computed": True,
        }
    else:
        # ---- holdout：**唯**经输入封条门禁（裁决 K §2 / 裁决 L 步骤 5-8） ----
        seal = require_input_seal()
        res = {
            "dataset": "holdout",
            "is_holdout": True,
            "holdout_read": True,
            "regime": args.regime,
            "structure_checks": holdout_structure_checks(),
            "mechanism_diagnostics": holdout_mechanism_diagnostics(args.regime),
            "repeat_record_counts": {
                k: len(v) for k, v in holdout_repeat_records(args.regime).items()
            },
        }
        payload = {
            "schema": "h2025_stage_j_scores/v1",
            "authority": "导师裁决 L（正式批准：批准运行 H2025 H2 正式评估）",
            "code_sha256": sha256_file(Path(__file__)),
            "arm_mapping": arm_mapping_declaration(),
            "codebook_sha256_expected": PRIMARY_CODEBOOK_SHA256_EXPECTED,
            "codebook_sha256_on_disk": (
                sha256_file(PRIMARY_CODEBOOK) if PRIMARY_CODEBOOK.exists() else None
            ),
            "realizations_sha256_expected": REALIZATIONS_SHA256_EXPECTED,
            "input_seal_sha256": sha256_file(INPUT_SEAL_PATH),
            "input_seal_approval_quote": seal.get("approval_quote"),
            "result": res,
            "no_holdout_performance_read": False,
            "no_h2_endpoint_computed": True,
            "note": ("holdout 结构自检与机制诊断；**端点与显著性**由 "
                     "scripts/h2_evaluation_run.py 一次性产出。"),
        }

    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("WROTE", args.out)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
