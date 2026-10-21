#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H2025 —— H2 统计推断（裁决 K §7 落实）

**唯一独立统计簇 = 16 个身份（holdout query identities）。**
`3 seed × 10 gallery realization = 30` 个重复条件，**不是** 30 个独立样本。

Primary bootstrap（裁决 D8 / §7）：
    每次有放回抽 **16 个身份**（身份簇 bootstrap） →
    在该抽样上算该重复条件的指标 → 对 30 个重复条件求平均 →
    **每次迭代只产出 1 个 Δ** → 重复 B 次（默认 10 000）。

措辞纪律（裁决 K §7，**强制**）：
    identity-cluster bootstrap preserves within-identity dependence
    for uncertainty estimation.

    ❌ **禁止**声称它"消除了 query 数量不均衡"。
    ✅ Primary Rank-1 仍为 **query-weighted**。
    ✅ macro-Rank-1 为 **identity-balanced** 指标。
    ✅ 两者**均须报告**。

推断纪律（裁决 C/E，勘误 T25–T34）：
    - 判据 = **CI 下界规则**：估计 > 0 **且** bootstrap 95% CI 下界 > 0。
    - **不出 formal p-value**；**不用** Holm 作为门槛；**不用** Wilcoxon。
    - 10 个 realization 只允许报 **positive / negative realization count** 与方向一致性。

执行边界（fail-closed）：holdout 需 `locks/H2_EVALUATION_INPUT_SEAL.json`。

用法：
    python scripts/h2_statistical_inference.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\H2025_clean_restart")
INPUT_SEAL_PATH = ROOT / "locks" / "H2_EVALUATION_INPUT_SEAL.json"

#: 唯一独立簇数（holdout 身份数）
N_IDENTITY_CLUSTERS = 16
N_GALLERY_REALIZATIONS = 10
SEEDS = (42, 43, 44)
N_REPEATS = len(SEEDS) * N_GALLERY_REALIZATIONS  # 30 个重复条件（非独立样本）

# ======================================================================================
# 裁决 L：O1 正式关闭 —— R1 为 Primary，R2 为 key secondary robustness
# ======================================================================================

#: **Primary H2 对比 = R1**：
#:     Delta(s,g) = M(C_seed(s,g)) − M(A_seed(s,g))
#:     3 training seeds × 10 gallery realizations = **30 个重复条件**
#:     H2a: Delta_1g > 0 且 bootstrap 95% CI lower bound > 0
#:     H2b: T = Delta_1g − Delta_full > 0 且 bootstrap 95% CI lower bound > 0
PRIMARY_CONTRAST = {
    "id": "R1",
    "authority": "导师裁决 L（O1 正式关闭）",
    "num_arm": "C_seed",
    "den_arm": "A_seed",
    "per_seed": True,
    "n_repeated_conditions": 30,
    "n_independent_clusters": 16,
    "h2a": "Delta_1g > 0 AND bootstrap 95% CI lower bound > 0",
    "h2b": "T = Delta_1g - Delta_full > 0 AND bootstrap 95% CI lower bound > 0",
}

#: **R2 = C_ens − D**：key secondary robustness analysis。
#: 判断 Local 是否还能在 3-seed Global ensemble 之上提供增益。
#: **不进入** H2a/H2b Primary 判定；**不得覆盖或替代 R1**。
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

#: ConvNeXt：逐 seed `C'_seed − A'_seed`，仅报告 H2a/H2b replication。
CONVNEXT_REPLICATION_CONTRAST = {
    "num_arm": "C_p_seed",
    "den_arm": "A_p_seed",
    "per_seed": True,
    "scope": "H2a / H2b replication only",
    "enters_mobilenet_primary_decision": False,
}

#: 冻结项（裁决 L 末段）：正式评估开始后一律禁止变更。
FROZEN = {
    "n_identity_clusters": 16,
    "n_gallery_realizations": 10,
    "seeds": (42, 43, 44),
    "realization_seeds": tuple(range(101, 111)),
    "regimes": ("full", "g3", "g1"),
    "primary_regime": "g1",
    "reference_regime": "full",
    "n_query_fixed": 99,
    "bootstrap_reps": 10_000,
    "ci_method": "percentile (2.5 / 97.5)",
    "ci_rule": "estimate > 0 AND bootstrap 95% CI lower bound > 0",
    "p_value_reported": False,
}

FORBIDDEN_AFTER_OPENING = (
    "调 alpha", "调 TopK", "调 PCA", "调 GMM", "调 backbone",
    "删 seed", "挑 realization", "筛 query",
    "修改 missing-local fallback", "修改统计终点", "修改 CI 规则",
)


def arm_mapping_declaration() -> dict:
    return {
        "authority": "导师裁决 L",
        "primary": PRIMARY_CONTRAST,
        "secondary_robustness": SECONDARY_ROBUSTNESS_CONTRAST,
        "convnext_replication": CONVNEXT_REPLICATION_CONTRAST,
        "frozen": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()},
        "forbidden_after_opening": list(FORBIDDEN_AFTER_OPENING),
    }

METRICS = ("rank1_query_weighted", "macro_rank1_identity_balanced",
           "identity_level_map_mrr")

DEFAULT_N_BOOT = 10_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# 重复条件的打包（把可变长度的身份簇压成规则矩阵，便于向量化 bootstrap）
# --------------------------------------------------------------------------------------


def pack_repeat(
    query_identities: np.ndarray,
    rank: np.ndarray,
    candidate_identities: list[str],
) -> dict:
    """把一个重复条件压成 (n_candidates, max_q_per_identity) 的规则矩阵。

    `idx[s, j] = -1` 表示该身份第 j 个 query 不存在（padding）。
    """
    q = np.asarray(query_identities).astype(str)
    r = np.asarray(rank).astype(np.int64)
    if q.shape != r.shape:
        raise ValueError("query_identities 与 rank 形状不一致")

    n_c = len(candidate_identities)
    counts = np.array([int((q == c).sum()) for c in candidate_identities], dtype=np.int64)
    maxq = int(counts.max()) if n_c else 0

    idx = np.full((n_c, maxq), -1, dtype=np.int64)
    for s, c in enumerate(candidate_identities):
        m = np.flatnonzero(q == c)
        if m.size:
            idx[s, :m.size] = m

    mask = idx >= 0
    safe = np.where(mask, idx, 0)
    return {
        "idx": idx,
        "mask": mask,
        "rank": np.where(mask, r[safe], 0),
        "counts": counts,
        "n_queries": int(r.size),
        "n_identities_with_query": int((counts > 0).sum()),
        "n_candidates": n_c,
    }


def _metric_from_slots(r: np.ndarray, m: np.ndarray, metric: str) -> float:
    """在已按身份槽位打包的 (n_slots, maxq) 上算指标（含重复槽位的多重性）。"""
    if not m.any():
        return float("nan")
    hit = (r == 1) & m
    if metric == "rank1_query_weighted":
        # query-weighted：每个 query 等权（重复身份槽位中的 query 也按多重性计入）
        return float(hit.sum() / m.sum())
    if metric == "identity_level_map_mrr":
        rr = np.where(m, 1.0 / np.maximum(r, 1), 0.0)
        return float(rr.sum() / m.sum())
    if metric == "macro_rank1_identity_balanced":
        # identity-balanced：每个**身份槽位**等权，再对槽位求均值
        per_slot_num = hit.sum(axis=1).astype(np.float64)
        per_slot_den = m.sum(axis=1).astype(np.float64)
        valid = per_slot_den > 0
        if not valid.any():
            return float("nan")
        return float(np.mean(per_slot_num[valid] / per_slot_den[valid]))
    raise ValueError("未知 metric: %s" % metric)


def metric_on_sample(packed: dict, sample_slots: np.ndarray, metric: str) -> float:
    """在给定的身份槽位抽样上算该重复条件的指标。"""
    return _metric_from_slots(packed["rank"][sample_slots],
                              packed["mask"][sample_slots], metric)


# --------------------------------------------------------------------------------------
# 点估计
# --------------------------------------------------------------------------------------


def delta_point_estimate(repeats: list[dict], metric: str,
                         arm_num: str, arm_den: str) -> dict:
    """未抽样的 Δ 点估计，并给出各重复条件的方向计数（纯描述性）。"""
    vals = []
    for rep in repeats:
        a = metric_on_sample(rep["packed"][arm_num],
                             np.arange(rep["packed"][arm_num]["n_candidates"]), metric)
        b = metric_on_sample(rep["packed"][arm_den],
                             np.arange(rep["packed"][arm_den]["n_candidates"]), metric)
        vals.append(a - b)
    v = np.array(vals, dtype=np.float64)
    return {
        "delta_point_estimate_full_sample": float(np.mean(v)),
        "n_repeats": int(v.size),
        "per_repeat_delta": v.tolist(),
        "directional_consistency": {
            "n_positive": int((v > 0).sum()),
            "n_negative": int((v < 0).sum()),
            "n_zero": int((v == 0).sum()),
            "note": ("仅作 robustness descriptor / 方向一致性描述；"
                     "realization 是固定重复条件，不是独立样本，"
                     "**不得**据此产生 p 值或显著性结论"),
        },
    }


# --------------------------------------------------------------------------------------
# identity-cluster bootstrap
# --------------------------------------------------------------------------------------


def identity_cluster_bootstrap(
    repeats: list[dict],
    metric: str,
    arm_num: str,
    arm_den: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20260917,
    n_clusters: int = N_IDENTITY_CLUSTERS,
) -> dict:
    """Primary bootstrap（裁决 D8 / K §7）。

    每次迭代：有放回抽 `n_clusters` 个**身份槽位** → 在该抽样上算每个重复条件的
    指标 → 对全部重复条件求平均 → **只产出 1 个 Δ***。
    """
    rng = np.random.RandomState(seed)
    n_c = repeats[0]["packed"][arm_num]["n_candidates"]

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s = rng.randint(0, n_c, size=n_clusters)
        acc = np.empty(len(repeats), dtype=np.float64)
        for i, rep in enumerate(repeats):
            a = metric_on_sample(rep["packed"][arm_num], s, metric)
            d = metric_on_sample(rep["packed"][arm_den], s, metric)
            acc[i] = a - d
        deltas[b] = float(np.nanmean(acc))

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": metric,
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "n_identity_clusters_resampled": int(n_clusters),
        "n_repeats_averaged": len(repeats),
        "ci_method": "percentile (2.5 / 97.5)",
        "bootstrap_origin": "identity clusters, with replacement, "
                            "preserving within-identity query dependence",
        "delta_bootstrap_mean": float(deltas.mean()),
        "delta_bootstrap_median": float(np.median(deltas)),
        "ci95_lower": float(lo),
        "ci95_upper": float(hi),
    }


def decide(estimate: float, ci_lower: float) -> dict:
    """判据（裁决 C / T25）：estimate > 0 AND CI 下界 > 0。**无 p 值。**"""
    return {
        "rule": "estimate > 0 AND bootstrap 95% CI lower bound > 0",
        "estimate": float(estimate),
        "ci95_lower": float(ci_lower),
        "supported": bool(estimate > 0 and ci_lower > 0),
        "p_value_reported": False,
        "note": "p 值不是本轮判据；Holm / Wilcoxon 均已退出推断（见勘误 T25–T34）。",
    }


def h2_endpoints(
    repeats_by_regime: dict[str, list[dict]],
    metric: str,
    arm_num: str,
    arm_den: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20260917,
    primary_regime: str = "g1",
    reference_regime: str = "full",
) -> dict:
    """H2a = Δ(primary_regime) > 0；H2b = T = Δ(primary) − Δ(reference) > 0。

    两个端点使用**同一次身份抽样**（配对），故优先在同一迭代内计算 T。
    """
    rep_p = repeats_by_regime[primary_regime]
    rep_r = repeats_by_regime[reference_regime]

    point_p = delta_point_estimate(rep_p, metric, arm_num, arm_den)
    point_r = delta_point_estimate(rep_r, metric, arm_num, arm_den)

    rng = np.random.RandomState(seed)
    n_c = rep_p[0]["packed"][arm_num]["n_candidates"]
    d_p = np.empty(n_boot); d_r = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.randint(0, n_c, size=N_IDENTITY_CLUSTERS)
        ap = np.array([metric_on_sample(x["packed"][arm_num], s, metric)
                       - metric_on_sample(x["packed"][arm_den], s, metric) for x in rep_p])
        ar = np.array([metric_on_sample(x["packed"][arm_num], s, metric)
                       - metric_on_sample(x["packed"][arm_den], s, metric) for x in rep_r])
        d_p[b] = float(np.nanmean(ap))
        d_r[b] = float(np.nanmean(ar))

    t = d_p - d_r
    lo_p, hi_p = np.percentile(d_p, [2.5, 97.5])
    lo_t, hi_t = np.percentile(t, [2.5, 97.5])

    return {
        "metric": metric,
        "contrast": "%s - %s" % (arm_num, arm_den),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "H2a": {
            "regime": primary_regime,
            "definition": "Delta_%s > 0" % primary_regime,
            "point_estimate": point_p["delta_point_estimate_full_sample"],
            "ci95": [float(lo_p), float(hi_p)],
            "decision": decide(point_p["delta_point_estimate_full_sample"], lo_p),
            "per_repeat": point_p,
        },
        "H2b": {
            "definition": "T = Delta_%s - Delta_%s > 0" % (primary_regime, reference_regime),
            "T_point_estimate": (point_p["delta_point_estimate_full_sample"]
                                 - point_r["delta_point_estimate_full_sample"]),
            "ci95": [float(lo_t), float(hi_t)],
            "decision": decide(point_p["delta_point_estimate_full_sample"]
                               - point_r["delta_point_estimate_full_sample"], lo_t),
            "reference_regime": reference_regime,
            "reference_point_estimate": point_r["delta_point_estimate_full_sample"],
            "reference_per_repeat": point_r,
        },
        "both_supported": bool(
            point_p["delta_point_estimate_full_sample"] > 0 and lo_p > 0
            and (point_p["delta_point_estimate_full_sample"]
                 - point_r["delta_point_estimate_full_sample"]) > 0 and lo_t > 0
        ),
    }


# --------------------------------------------------------------------------------------
# 净挽救（net rescue）的 identity-cluster bootstrap（预注册 §7.7 / 判定表条件 ③）
# --------------------------------------------------------------------------------------


def pack_signed_repeat(
    query_identities: np.ndarray,
    signed: np.ndarray,
    candidate_identities: list[str],
) -> dict:
    """把 per-query 的有符号事件（+1 = rescue，−1 = damage，0 = 其他）打包成规则矩阵。"""
    q = np.asarray(query_identities).astype(str)
    s = np.asarray(signed, dtype=np.int64)
    if q.shape != s.shape:
        raise ValueError("query_identities 与 signed 形状不一致")

    n_c = len(candidate_identities)
    counts = np.array([int((q == c).sum()) for c in candidate_identities], dtype=np.int64)
    maxq = int(counts.max()) if n_c else 0
    idx = np.full((n_c, maxq), -1, dtype=np.int64)
    for i, c in enumerate(candidate_identities):
        m = np.flatnonzero(q == c)
        if m.size:
            idx[i, :m.size] = m
    mask = idx >= 0
    safe = np.where(mask, idx, 0)
    return {"idx": idx, "mask": mask, "signed": np.where(mask, s[safe], 0),
            "counts": counts, "n_candidates": n_c}


def net_rescue_bootstrap(
    signed_repeats: list[dict],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20260917,
    n_clusters: int = N_IDENTITY_CLUSTERS,
) -> dict:
    """net rescue = rescue − damage 的 identity-cluster bootstrap。

    每次迭代：有放回抽 `n_clusters` 个身份槽位 → 在**每个重复条件**上算
    `net_rate = (Σ signed) / (Σ queries)` → 对全部重复条件取均值 → 1 个值。
    query 视角 = 上述率差；宏观视角 = 先按身份槽位算率再对槽位平均。
    """
    rng = np.random.RandomState(seed)
    n_c = signed_repeats[0]["n_candidates"]
    q_vals = np.empty(n_boot)
    m_vals = np.empty(n_boot)

    for b in range(n_boot):
        s = rng.randint(0, n_c, size=n_clusters)
        q_acc = np.empty(len(signed_repeats))
        m_acc = np.empty(len(signed_repeats))
        for i, rep in enumerate(signed_repeats):
            sg = rep["signed"][s]
            mk = rep["mask"][s]
            cnt = mk.sum()
            q_acc[i] = float(sg.sum() / cnt) if cnt else np.nan
            den = mk.sum(axis=1)
            valid = den > 0
            m_acc[i] = (float(np.mean(sg.sum(axis=1)[valid] / den[valid]))
                        if valid.any() else np.nan)
        q_vals[b] = float(np.nanmean(q_acc))
        m_vals[b] = float(np.nanmean(m_acc))

    def _ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi)

    q_lo, q_hi = _ci(q_vals)
    m_lo, m_hi = _ci(m_vals)
    return {
        "metric": "net_rescue_rate_difference",
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "n_identity_clusters_resampled": int(n_clusters),
        "n_repeats_averaged": len(signed_repeats),
        "ci_method": "percentile (2.5 / 97.5)",
        "query_weighted": {"estimate": float(np.mean(q_vals)),
                           "ci95": [q_lo, q_hi],
                           "decision": decide(float(np.mean(q_vals)), q_lo)},
        "macro_identity_balanced": {"estimate": float(np.mean(m_vals)),
                                    "ci95": [m_lo, m_hi],
                                    "decision": decide(float(np.mean(m_vals)), m_lo)},
    }


def net_rescue_signed_arrays(
    ranks_num: np.ndarray,
    ranks_den: np.ndarray,
) -> dict:
    """由 ranking 生成有符号事件（**定义见预注册 §7.7**）。

    rescue : 分母臂（Global）错 且 分子臂（Fusion）对  -> +1
    damage : 分母臂对 且 分子臂错                     -> −1
    其余 -> 0。仅返回计数，不作推断。
    """
    a = np.asarray(ranks_den, dtype=np.int64)
    c = np.asarray(ranks_num, dtype=np.int64)
    rescue = (a > 1) & (c == 1)
    damage = (a == 1) & (c > 1)
    signed = np.zeros(a.shape, dtype=np.int64)
    signed[rescue] = 1
    signed[damage] = -1
    return {"signed": signed,
            "n_rescue": int(rescue.sum()), "n_damage": int(damage.sum()),
            "n_net": int(rescue.sum() - damage.sum())}


# --------------------------------------------------------------------------------------
# synthetic self-test
# --------------------------------------------------------------------------------------


def _synthetic_world(
    rng: np.random.RandomState,
    n_clusters: int = 16,
    q_per_cluster: tuple[int, int] = (2, 7),
    p_correct_gain: float = 0.0,
    base_rank: int = 2,
    gain_rank: int = 1,
    independent_null: bool = False,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """构造合成世界：返回 (候选身份, 每 query 的身份, {arm: rank})。

    p_correct_gain : arm "C" 相对 "D" 被"纠正"为 rank 1 的 query 比例（配对增益）。
    independent_null: 若 True，C 与 D 各自**独立**随机取 rank1/base_rank
                      （即真值为零效应但**带噪声**，用于检验 CI 是否覆盖 0）。
    """
    ids: list[str] = []
    qi: list[int] = []
    for s in range(n_clusters):
        cid = "id%03d" % s
        n = int(rng.randint(q_per_cluster[0], q_per_cluster[1] + 1))
        ids.append(cid)
        qi.extend([s] * n)
    slot = np.array(qi, dtype=np.int64)
    n_q = slot.size

    ranks = {}
    r_d = np.full(n_q, base_rank, dtype=np.int64)
    if independent_null:
        r_d = np.where(rng.rand(n_q) < 0.5, gain_rank, base_rank).astype(np.int64)
        r_c = np.where(rng.rand(n_q) < 0.5, gain_rank, base_rank).astype(np.int64)
    else:
        r_c = r_d.copy()
        r_c[rng.rand(n_q) < p_correct_gain] = gain_rank
    ranks["D"] = r_d
    ranks["C"] = r_c
    ranks["A"] = r_d.copy()                     # 无增益参照臂
    return ids, np.array([ids[i] for i in slot]), ranks


def _pack_all(ids: list[str], q_ids: np.ndarray, ranks: dict[str, np.ndarray],
              n_repeats: int) -> list[dict]:
    out = []
    for _ in range(n_repeats):
        out.append({"packed": {a: pack_repeat(q_ids, r, ids) for a, r in ranks.items()}})
    return out


def self_test(n_boot: int = 2000) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)[:220]})

    # ---- S1 打包正确性 ----
    ids = ["a", "b", "c"]
    q = np.array(["a", "b", "b", "c"])
    r = np.array([1, 2, 3, 1])
    p = pack_repeat(q, r, ids)
    check("S1a packing counts", p["counts"].tolist() == [1, 2, 1], str(p["counts"].tolist()))
    check("S1b packing preserves ranks",
          p["rank"][0, 0] == 1 and p["rank"][1, 0] == 2 and p["rank"][1, 1] == 3
          and p["rank"][2, 0] == 1, "%s" % p["rank"].tolist())

    # ---- S2 三种指标的定义差异 ----
    # 身份 a: 3 个 query 全对 ; 身份 b: 1 个 query 错
    q2 = np.array(["a", "a", "a", "b"])
    r2 = np.array([1, 1, 1, 2])
    p2 = pack_repeat(q2, r2, ["a", "b"])
    s_all = np.arange(2)
    r1 = metric_on_sample(p2, s_all, "rank1_query_weighted")
    mc = metric_on_sample(p2, s_all, "macro_rank1_identity_balanced")
    check("S2a query-weighted == 3/4", abs(r1 - 0.75) < 1e-12, "%.6f" % r1)
    check("S2b identity-balanced == 1/2", abs(mc - 0.5) < 1e-12, "%.6f" % mc)
    check("S2c 两者必须不同（不均衡 query 数下）", abs(r1 - mc) > 1e-12,
          "qw=%.4f macro=%.4f" % (r1, mc))

    # ---- S3 槽位多重性：重复抽样同一身份应计入两次 ----
    s_dup = np.array([1, 1])          # 身份 b 两次 -> 全错 -> macro 与 qw 均为 0
    check("S3a duplicated slot counted with multiplicity",
          metric_on_sample(p2, s_dup, "rank1_query_weighted") == 0.0
          and metric_on_sample(p2, s_dup, "macro_rank1_identity_balanced") == 0.0,
          "重复槽位按多重性计入")

    # ---- S4 已知正效应：H2a 必须 supported 且 CI 下界 > 0 ----
    rng = np.random.RandomState(7)
    ids4, q4, ranks4 = _synthetic_world(rng, p_correct_gain=0.35)
    reps4 = _pack_all(ids4, q4, ranks4, n_repeats=N_REPEATS)
    res4 = h2_endpoints({"g1": reps4, "full": reps4}, "rank1_query_weighted",
                        "C", "D", n_boot=n_boot, seed=11)
    check("S4a positive-effect world -> H2a supported",
          res4["H2a"]["decision"]["supported"] is True,
          "Delta=%.4f CI=[%.4f, %.4f]" % (res4["H2a"]["point_estimate"],
                                          *res4["H2a"]["ci95"]))
    check("S4b positive-effect world -> CI lower > 0",
          res4["H2a"]["ci95"][0] > 0, "lo=%.4f" % res4["H2a"]["ci95"][0])

    # ---- S5 零效应世界（带噪声）：H2a 必须 NOT supported，CI 必须覆盖 0 ----
    ids5, q5, ranks5 = _synthetic_world(rng, independent_null=True)
    reps5 = _pack_all(ids5, q5, ranks5, n_repeats=N_REPEATS)
    res5 = h2_endpoints({"g1": reps5, "full": reps5}, "rank1_query_weighted",
                        "C", "D", n_boot=n_boot, seed=11)
    check("S5a null world (noisy, independent arms) -> H2a NOT supported",
          res5["H2a"]["decision"]["supported"] is False,
          "Delta=%.4f CI=[%.4f, %.4f]" % (res5["H2a"]["point_estimate"],
                                          *res5["H2a"]["ci95"]))
    check("S5b null world -> CI covers 0",
          res5["H2a"]["ci95"][0] <= 0 <= res5["H2a"]["ci95"][1],
          "lo=%.4f hi=%.4f" % tuple(res5["H2a"]["ci95"]))
    check("S5c null world CI is non-degenerate (width > 0)",
          res5["H2a"]["ci95"][1] - res5["H2a"]["ci95"][0] > 0,
          "width=%.4f" % (res5["H2a"]["ci95"][1] - res5["H2a"]["ci95"][0]))

    # ---- S6 可复现性：同 seed 同结果 ----
    a = identity_cluster_bootstrap(reps4, "rank1_query_weighted", "C", "D",
                                   n_boot=300, seed=99)
    b = identity_cluster_bootstrap(reps4, "rank1_query_weighted", "C", "D",
                                   n_boot=300, seed=99)
    check("S6 same seed -> identical bootstrap",
          a["ci95_lower"] == b["ci95_lower"] and a["ci95_upper"] == b["ci95_upper"],
          "%.10f == %.10f" % (a["ci95_lower"], b["ci95_lower"]))

    # ---- S7 无 p 值 ----
    flat = json.dumps(res4, ensure_ascii=False)
    check("S7 no p-value anywhere in output",
          '"p_value_reported": false' in flat and '"p_value":' not in flat,
          "仅记录 p_value_reported=false")

    # ---- S8 方向一致性只报计数 ----
    dc = res4["H2a"]["per_repeat"]["directional_consistency"]
    check("S8 directional consistency reported as counts only",
          set(("n_positive", "n_negative", "n_zero")) <= set(dc.keys())
          and "p_value" not in dc,
          json.dumps({k: dc[k] for k in ("n_positive", "n_negative", "n_zero")}))

    # ---- S10 H2b：受限登记增益 > full-gallery 增益 -> T 的 CI 下界 > 0 ----
    rng10 = np.random.RandomState(21)
    ids_p, q_p, rk_p = _synthetic_world(rng10, p_correct_gain=0.40)   # g1 增益大
    ids_r, q_r, rk_r = _synthetic_world(rng10, p_correct_gain=0.05)   # full 增益小
    res10 = h2_endpoints(
        {"g1": _pack_all(ids_p, q_p, rk_p, N_REPEATS),
         "full": _pack_all(ids_r, q_r, rk_r, N_REPEATS)},
        "rank1_query_weighted", "C", "D", n_boot=n_boot, seed=5)
    check("S10a H2b positive interaction -> supported",
          res10["H2b"]["decision"]["supported"] is True,
          "T=%.4f CI=[%.4f, %.4f]" % (res10["H2b"]["T_point_estimate"],
                                      *res10["H2b"]["ci95"]))
    check("S10b paired sampling: H2a and H2b share the same identity draw",
          res10["H2a"]["decision"]["supported"] is True
          and res10["both_supported"] is True,
          "both_supported=%s" % res10["both_supported"])

    # ---- S9 重复条件数 = 30（3 seed × 10 realization），但独立簇 = 16 ----
    check("S9 repeats=30 and independent clusters=16 are distinct",
          N_REPEATS == 30 and N_IDENTITY_CLUSTERS == 16,
          "30 重复条件 ≠ 30 独立样本；唯一独立簇 = 16 身份")

    # ---- S11 裁决 L：R1 为 Primary，R2 为次级且不得替代 ----
    check("S11a primary contrast is R1 (per-seed C_seed - A_seed)",
          PRIMARY_CONTRAST["id"] == "R1" and PRIMARY_CONTRAST["per_seed"] is True
          and PRIMARY_CONTRAST["num_arm"] == "C_seed"
          and PRIMARY_CONTRAST["den_arm"] == "A_seed",
          json.dumps({k: PRIMARY_CONTRAST[k] for k in ("id", "num_arm", "den_arm")},
                     ensure_ascii=False))
    check("S11b R2 cannot override or replace R1",
          SECONDARY_ROBUSTNESS_CONTRAST["enters_h2_primary_decision"] is False
          and SECONDARY_ROBUSTNESS_CONTRAST["may_override_or_replace_R1"] is False,
          "R2 仅作 key secondary robustness analysis")
    check("S11c ConvNeXt excluded from MobileNet primary decision",
          CONVNEXT_REPLICATION_CONTRAST["per_seed"] is True
          and CONVNEXT_REPLICATION_CONTRAST["enters_mobilenet_primary_decision"] is False,
          "逐 seed C'_seed - A'_seed，仅 replication")
    check("S11d frozen statistical constants",
          FROZEN["n_identity_clusters"] == 16 and FROZEN["n_gallery_realizations"] == 10
          and FROZEN["seeds"] == (42, 43, 44)
          and FROZEN["realization_seeds"] == tuple(range(101, 111))
          and FROZEN["primary_regime"] == "g1" and FROZEN["reference_regime"] == "full"
          and FROZEN["bootstrap_reps"] == 10_000
          and FROZEN["p_value_reported"] is False
          and N_REPEATS == 30,
          "clusters=16 real=10 seeds=42/43/44 real_seeds=101..110 g1 vs full B=10000 无 p 值")
    check("S11e forbidden-after-opening list complete",
          set(FORBIDDEN_AFTER_OPENING) == {
              "调 alpha", "调 TopK", "调 PCA", "调 GMM", "调 backbone",
              "删 seed", "挑 realization", "筛 query",
              "修改 missing-local fallback", "修改统计终点", "修改 CI 规则"},
          "%d 项" % len(FORBIDDEN_AFTER_OPENING))

    # ---- S12 净挽救（判定表条件 ③）的打包、符号与 CI ----
    # 手工样例（ranks_den = Global，ranks_num = Fusion）：
    #   q0 Global 对/Fusion 错 -> damage；q1 Global 错/Fusion 对 -> rescue；
    #   q2 双方都对 -> 0；q3 Global 对/Fusion 错 -> damage
    #   => rescue=1, damage=2, net=-1, signed=[-1, +1, 0, -1]
    sa = net_rescue_signed_arrays(np.array([2, 1, 1, 3]), np.array([1, 2, 1, 1]))
    check("S12a rescue/damage/signed definition correct",
          sa["n_rescue"] == 1 and sa["n_damage"] == 2 and sa["n_net"] == -1
          and sa["signed"].tolist() == [-1, 1, 0, -1],
          "rescue=%d damage=%d net=%d signed=%s"
          % (sa["n_rescue"], sa["n_damage"], sa["n_net"], sa["signed"].tolist()))
    check("S12a2 net = rescue - damage identity",
          sa["n_net"] == sa["n_rescue"] - sa["n_damage"],
          "%d == %d - %d" % (sa["n_net"], sa["n_rescue"], sa["n_damage"]))

    ids_n = ["id%02d" % i for i in range(16)]
    q_n = np.repeat(ids_n, 3)
    all_rescue = [pack_signed_repeat(q_n, np.ones(q_n.size, dtype=np.int64), ids_n)
                  for _ in range(N_REPEATS)]
    nr_full = net_rescue_bootstrap(all_rescue, n_boot=300, seed=7)
    check("S12b all-rescue -> net rescue CI lower > 0",
          nr_full["query_weighted"]["estimate"] == 1.0
          and nr_full["query_weighted"]["decision"]["supported"] is True,
          "est=%.4f CI=[%.4f, %.4f]" % (nr_full["query_weighted"]["estimate"],
                                        *nr_full["query_weighted"]["ci95"]))

    rng12 = np.random.RandomState(31)
    mixed = [pack_signed_repeat(
        q_n, rng12.choice([-1, 1], size=q_n.size).astype(np.int64), ids_n)
        for _ in range(N_REPEATS)]
    nr_null = net_rescue_bootstrap(mixed, n_boot=300, seed=7)
    check("S12c balanced rescue/damage -> CI contains 0 (not 'supported')",
          nr_null["query_weighted"]["ci95"][0] < 0 < nr_null["query_weighted"]["ci95"][1]
          and nr_null["query_weighted"]["decision"]["supported"] is False,
          "est=%.4f CI=[%.4f, %.4f]" % (nr_null["query_weighted"]["estimate"],
                                        *nr_null["query_weighted"]["ci95"]))

    failed = [c["name"] for c in checks if not c["pass"]]
    return {
        "schema": "h2025_h2_statistical_selftest/v1",
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "verdict": "PASS" if not failed else "FAIL",
        "checks": checks,
        "no_real_data_read": True,
        "holdout_read": False,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def require_input_seal() -> dict:
    """holdout 统计的 fail-closed 门禁（裁决 K §2 / 裁决 L 步骤 5-8）。

    除存在性与 `approved_for_evaluation` 外，**逐条**校验封条记录的
    输入指纹与代码指纹；任一不符立即中止（防止"封条后偷改代码/输入"）。
    """
    if not INPUT_SEAL_PATH.exists():
        raise SystemExit(
            "REFUSED: holdout 统计被拒绝。缺少 %s\n"
            "  该封条须在导师批准并生成 Local holdout FV 之后另行建立。" % INPUT_SEAL_PATH
        )
    seal = json.loads(INPUT_SEAL_PATH.read_text(encoding="utf-8"))
    if not seal.get("approved_for_evaluation"):
        raise SystemExit("REFUSED: 封条存在但 approved_for_evaluation != true。")
    mismatched = []
    for item in seal.get("inputs", []):
        p = ROOT / item["path"]
        if not p.exists() or sha256_file(p) != item["sha256"]:
            mismatched.append(item["path"])
    if mismatched:
        raise SystemExit("REFUSED: 封条记录的输入指纹与磁盘不符：%s" % mismatched)
    code_bad = []
    for name, h in seal.get("code_hashes", {}).items():
        p = ROOT / "scripts" / name
        if not p.exists() or sha256_file(p) != h:
            code_bad.append(name)
    if code_bad:
        raise SystemExit("REFUSED: 封条记录的代码哈希与磁盘不符（脚本已变更）：%s" % code_bad)
    return seal


def repeats_from_records(
    records: list[dict],
    candidate_identities: list[str],
    num_arm: str,
    den_arm: str,
) -> list[dict]:
    """把 Stage J 的重复条件记录转成 bootstrap 可消费的 `packed` 形式。

    每个重复条件保留其 `meta`（regime / seed / realization），便于审计与
    方向一致性描述；**不改变**任何 rank、**不裁剪**任何候选身份。
    """
    out: list[dict] = []
    for rep in records:
        q_ids = np.asarray(rep["query_identities"]).astype(str)
        rk = rep["ranks"]
        out.append({
            "packed": {
                num_arm: pack_repeat(q_ids, np.asarray(rk["num"], dtype=np.int64),
                                     candidate_identities),
                den_arm: pack_repeat(q_ids, np.asarray(rk["den"], dtype=np.int64),
                                     candidate_identities),
            },
            "meta": {"regime": rep.get("regime"), "seed": rep.get("seed"),
                     "realization": rep.get("realization")},
        })
    return out


def per_regime_point_estimates(
    repeats_by_regime: dict[str, list[dict]],
    metric: str,
    arm_num: str,
    arm_den: str,
) -> dict:
    """逐 regime 的 Δ 点估计（H2a/H2b 之外**必须并列报告** Δ_1g/Δ_3g/Δ_full）。"""
    return {g: delta_point_estimate(rep, metric, arm_num, arm_den)
            for g, rep in repeats_by_regime.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="H2025 H2 统计推断（裁决 K §7 / 裁决 L）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--run-holdout", action="store_true",
                    help="仅用于核验门禁；正式端点须走 scripts/h2_evaluation_run.py")
    args = ap.parse_args()

    if args.self_test or not args.run_holdout:
        res = self_test(n_boot=min(args.n_boot, 2000))
        res["code_sha256"] = sha256_file(Path(__file__))
        res["arm_mapping"] = arm_mapping_declaration()
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["verdict"] == "PASS" else 1

    # 门禁实测：无封条 -> REFUSED；有封条 -> 仍 REFUSED（正式统计只允许唯一入口）
    require_input_seal()
    raise SystemExit(
        "REFUSED: 正式 holdout 统计**不得**由本模块单独运行。\n"
        "  唯一正式入口：python scripts/h2_evaluation_run.py（裁决 L 步骤 8）；\n"
        "  该入口会先调用 require_input_seal() 校验封条，再一次性产出 H2 端点。"
    )


if __name__ == "__main__":
    sys.exit(main())
