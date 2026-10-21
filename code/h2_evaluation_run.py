#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
H2025 —— H2 正式评估的**唯一入口**（导师裁决 L 步骤 8-9）

本脚本是裁决 L 授权的 9 步 fail-closed 序列中的第 8-9 步执行体：

    先调用 stage_j_scores.require_input_seal()（存在性 + approved_for_evaluation
    + 输入指纹 + 代码指纹四项全验），全部通过后才计算 H2 端点，**一次性**输出。

端点与判定（预注册 v3 §7-§8 / 裁决 L）：
    R1（Primary）   Delta(s,g) = M(C_seed) − M(A_seed)，3 seed × 10 realization = 30 重复条件
    H2a             Delta_1g > 0 且 bootstrap 95% CI 下界 > 0
    H2b             T = Delta_1g − Delta_full > 0 且 CI 下界 > 0
    R2（次级）      C_ens − D，**不进入** H2a/H2b Primary，不得覆盖或替代 R1
    ConvNeXt        逐 seed C'_seed − A'_seed，仅作 H2a/H2b replication 报告
    净挽救          条件 ③：1g 的 net rescue > 0 且 CI 不含 0（query + 宏平均两视角）
    不劣于集成      条件 ④：C_ens(1g) − D(1g) 的 CI 下界 ≥ −1.0 pp

不确定性**只**来自 16 个身份簇的 identity-cluster bootstrap；30 是重复条件数，
**不是**样本量。**不报告 p 值**。

用法：
    python scripts/h2_evaluation_run.py                 # 正式运行（一次性）
    python scripts/h2_evaluation_run.py --n-boot 2000   # 仅调试用（会标记非正式）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\H2025_clean_restart")
REPORTS = ROOT / "reports"
INPUT_SEAL = ROOT / "locks" / "H2_EVALUATION_INPUT_SEAL.json"
LOCKED_HOLDOUT = "locked holdout with accidental global-metric leakage"

OUT_JSON = REPORTS / "H2_EVALUATION_RESULT.json"
OUT_MD = REPORTS / "H2_EVALUATION_RESULT.md"

REGIMES = ("full", "g3", "g1")
PRIMARY_REGIME = "g1"
REFERENCE_REGIME = "full"
PRIMARY_METRIC = "rank1_query_weighted"
METRICS = ("rank1_query_weighted", "macro_rank1_identity_balanced",
           "identity_level_map_mrr")

#: 对比 → (分子臂, 分母臂)
CONTRASTS = {
    "R1": ("C_seed", "A_seed"),
    "R2": ("C_ens", "D"),
    "CONVNEXT": ("C_p_seed", "A_p_seed"),
}

DIAGNOSTIC_REFERENCE = 0.01   # −1.0 个百分点
FULL_NO_GAIN_CEILING = 0.02   # +2.0 个百分点
BOOTSTRAP_SEED = 20260917


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SJ = _load_module("h2025_stage_j_scores", "scripts/stage_j_scores.py")
HS = _load_module("h2025_h2_stats", "scripts/h2_statistical_inference.py")


def sha256_file(p: Path) -> str:
    return SJ.sha256_file(p)


# --------------------------------------------------------------------------------------
# 打包
# --------------------------------------------------------------------------------------


def build_repeats(identities: list[str]) -> tuple[dict, dict]:
    """返回 (per_contrast[contrast][regime] = packed repeats, raw counts)。"""
    raw = {g: SJ.holdout_repeat_records(g) for g in REGIMES}
    per_contrast: dict[str, dict[str, list[dict]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for cid, (num_arm, den_arm) in CONTRASTS.items():
        per_contrast[cid] = {}
        counts[cid] = {}
        for g in REGIMES:
            per_contrast[cid][g] = HS.repeats_from_records(
                raw[g][cid], identities, num_arm, den_arm)
            counts[cid][g] = len(per_contrast[cid][g])
    return per_contrast, counts


def contrast_metrics(per_contrast: dict, cid: str, n_boot: int) -> dict:
    num_arm, den_arm = CONTRASTS[cid]
    out: dict[str, dict] = {}
    for metric in METRICS:
        ep = HS.h2_endpoints(per_contrast[cid], metric, num_arm, den_arm,
                             n_boot=n_boot, seed=BOOTSTRAP_SEED,
                             primary_regime=PRIMARY_REGIME,
                             reference_regime=REFERENCE_REGIME)
        pts = HS.per_regime_point_estimates(per_contrast[cid], metric,
                                            num_arm, den_arm)
        out[metric] = {
            "contrast": "%s - %s" % (num_arm, den_arm),
            "H2a": ep["H2a"]["decision"],
            "H2a_ci95": ep["H2a"]["ci95"],
            "H2a_point_estimate": ep["H2a"]["point_estimate"],
            "H2b": ep["H2b"]["decision"],
            "H2b_ci95": ep["H2b"]["ci95"],
            "H2b_T_point_estimate": ep["H2b"]["T_point_estimate"],
            "both_supported": ep["both_supported"],
            "per_regime_delta": {
                g: {
                    "delta_mean_over_repeats": v["delta_point_estimate_full_sample"],
                    "n_repeats": v["n_repeats"],
                    "n_positive": v["directional_consistency"]["n_positive"],
                    "n_negative": v["directional_consistency"]["n_negative"],
                    "n_zero": v["directional_consistency"]["n_zero"],
                } for g, v in pts.items()
            },
        }
    return out


def net_rescue(regime: str, identities: list[str], q_ids: np.ndarray,
               n_boot: int) -> dict:
    """净挽救（预注册 §7.7）：逐 (seed, realization) 条件构造有符号事件后 bootstrap。"""
    signed_reps = []
    per_seed_counts: dict[str, dict] = {}
    for s in (42, 43, 44):
        agg = {"n_rescue": 0, "n_damage": 0, "n_net": 0}
        for r_seed in range(101, 111):
            ranks = SJ.holdout_ranks(regime, r_seed)
            ev = HS.net_rescue_signed_arrays(ranks["C_seed%d" % s],
                                             ranks["A_seed%d" % s])
            signed_reps.append(HS.pack_signed_repeat(q_ids, ev["signed"], identities))
            for k in agg:
                agg[k] += ev[k]
        per_seed_counts["seed%d" % s] = agg
    boot = HS.net_rescue_bootstrap(signed_reps, n_boot=n_boot, seed=BOOTSTRAP_SEED)
    return {"regime": regime, "per_seed_counts": per_seed_counts,
            "bootstrap": boot,
            "definitions": {
                "rescue": "臂 A(Global) 的 Rank-1 ≠ 真值 且 臂 C(Fusion) 的 Rank-1 = 真值",
                "damage": "臂 A 对 且 臂 C 错",
                "net rescue": "rescue − damage",
            }}


def mechanism_table(regime: str) -> dict:
    return SJ.holdout_mechanism_diagnostics(regime)


# --------------------------------------------------------------------------------------
# 判定（预注册 v3 §8.1，事前写死）
# --------------------------------------------------------------------------------------


def verdict(r1_primary: dict, nr_g1: dict, r2_g1: dict) -> dict:
    a = bool(r1_primary["H2a"]["supported"])
    b = bool(r1_primary["H2b"]["supported"])
    c = bool(nr_g1["bootstrap"]["query_weighted"]["decision"]["supported"])
    r2_lower = float(r2_g1["H2a_ci95"][0])
    d = bool(r2_lower >= -DIAGNOSTIC_REFERENCE)

    if a and b and c and d:
        label = "LIMITED-ENROLLMENT COMPLEMENTARITY SUPPORTED"
    elif a and not b:
        label = "LOCAL GAIN PRESENT, NO ENROLLMENT INTERACTION"
    elif (not a) and (not c) and d:
        label = "H2 NOT SUPPORTED"
    else:
        label = "INDETERMINATE / ADDITIONAL ANALYSIS REQUIRED"
    return {
        "verdict": label,
        "criteria": {
            "① H2a (Delta_1g > 0 and CI lower > 0)": a,
            "② H2b (T = Delta_1g - Delta_full > 0 and CI lower > 0)": b,
            "③ net rescue(1g) > 0 with CI excluding 0": c,
            "④ C_ens(1g) not worse than D(1g) (CI lower >= -1.0 pp)": d,
        },
        "r2_ci95_lower_at_g1": r2_lower,
        "rule_source": "预注册 v3 §8.1（事前写死，不得事后修改）",
        "p_value_reported": False,
        "note": ("单调性不是硬门槛；full-gallery 几乎无增益本身**绝不**构成"
                 "「局部路线失败」（§8.2）。"),
    }


def full_gallery_no_gain(r1_primary: dict) -> dict:
    pt = r1_primary["per_regime_delta"]["full"]
    lo, hi = None, None
    return {
        "definition": "mean Delta_full(Rank-1) <= +2.0 pp AND 95% CI contains 0",
        "delta_full_mean": pt["delta_mean_over_repeats"],
        "ceiling": FULL_NO_GAIN_CEILING,
        "note": "该描述本身不构成任何否定性结论（§8.2 第 3 条）。",
    }


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="H2025 H2 正式评估唯一入口（裁决 L）")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--overwrite", action="store_true",
                    help="允许覆盖既有结果（默认一次性输出，已存在即拒绝）")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if (out_json.exists() or out_md.exists()) and not args.overwrite:
        raise SystemExit("REFUSED: 结果已存在（一次性输出）。加 --overwrite 才可重跑。")

    # ---- 步骤 8 前置：唯一入口的封条门禁（四项全验） ----
    seal = SJ.require_input_seal()
    HS.require_input_seal()
    seal_sha = sha256_file(INPUT_SEAL)

    formal = args.n_boot == 10_000
    labels = SJ.holdout_labels()
    identities = labels["identities"]
    q_ids = labels["query_labels"]

    if len(identities) != 16 or len(q_ids) != 99:
        raise SystemExit("REFUSED: 候选身份/query 数与冻结值不符（%d / %d）"
                         % (len(identities), len(q_ids)))

    structure = SJ.holdout_structure_checks()
    if not structure["pass"]:
        raise SystemExit("REFUSED: holdout 结构自检未通过：%s"
                         % json.dumps(structure, ensure_ascii=False)[:400])

    per_contrast, counts = build_repeats(identities)

    results: dict[str, dict] = {}
    for cid in CONTRASTS:
        results[cid] = contrast_metrics(per_contrast, cid, args.n_boot)

    nr = {g: net_rescue(g, identities, q_ids, args.n_boot) for g in REGIMES}
    mech = {g: mechanism_table(g) for g in REGIMES}

    r1_primary = results["R1"][PRIMARY_METRIC]
    r2_g1 = results["R2"][PRIMARY_METRIC]
    v = verdict(r1_primary, nr[PRIMARY_REGIME], r2_g1)

    payload = {
        "schema": "h2025_h2_evaluation_result/v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "authority": "导师裁决 K（§8 前置审计）+ 导师裁决 L（正式批准：批准运行 H2025 H2 正式评估）",
        "approval_quote": seal.get("approval_quote"),
        "formal_run": bool(formal),
        "n_boot": int(args.n_boot),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "arm_mapping": SJ.arm_mapping_declaration(),
        "data_scope": {
            "holdout_name": LOCKED_HOLDOUT,
            "not_used_for_model_selection": True,
            "d3_disclosure": ("该 holdout 曾因 D3 事件被全局指标意外污染（打印进训练日志/"
                              "写入 result.json）；该泄漏**未被用于任何模型选择**，"
                              "但 holdout 因此不再是未被全局指标触碰过的集合。"),
            "protocol_wording": "filename-derived date-separated protocol",
            "protocol_disclosure": ("capture dates are inferred from filenames and are not "
                                    "independently verified by upstream metadata."),
        },
        "provenance": {
            "code_sha256": {
                "stage_j_scores.py": sha256_file(ROOT / "scripts" / "stage_j_scores.py"),
                "h2_statistical_inference.py": sha256_file(
                    ROOT / "scripts" / "h2_statistical_inference.py"),
                "h2_evaluation_run.py": sha256_file(Path(__file__)),
            },
            "code_hashes_in_seal": seal.get("code_hashes"),
            "input_seal_path": "locks/H2_EVALUATION_INPUT_SEAL.json",
            "input_seal_sha256": seal_sha,
            "codebook_sha256": sha256_file(SJ.PRIMARY_CODEBOOK),
            "realizations_sha256": sha256_file(SJ.REALIZATIONS_PATH),
            "structure_checks": structure,
            "repeat_record_counts": counts,
        },
        "primary_contrast_R1": {
            "definition": "Delta(s,g) = M(C_seed(s,g)) - M(A_seed(s,g))",
            "n_repeated_conditions": 30,
            "n_independent_clusters": 16,
            "per_metric": results["R1"],
        },
        "secondary_robustness_R2": {
            "definition": "M(C_ens) - M(D)",
            "role": "key secondary robustness analysis",
            "enters_h2_primary_decision": False,
            "may_override_or_replace_R1": False,
            "n_repeated_conditions": 10,
            "per_metric": results["R2"],
        },
        "convnext_replication": {
            "definition": "per-seed M(C'_seed) - M(A'_seed)",
            "scope": "H2a / H2b replication only",
            "enters_mobilenet_primary_decision": False,
            "note": ("仅回答「Local 增益是否也存在于更强 Global backbone 之上」；"
                     "不作主干比较，不形成 dual-backbone 结论，不报告 ConvNeXt ensemble。"),
            "n_repeated_conditions": 30,
            "per_metric": results["CONVNEXT"],
        },
        "net_rescue": nr,
        "mechanism_diagnostics": mech,
        "full_gallery_no_gain_descriptor": full_gallery_no_gain(r1_primary),
        "decision": v,
        "no_holdout_performance_read": False,
        "holdout_read": True,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    write_markdown(payload, out_md)
    print("WROTE", out_json)
    print("WROTE", out_md)
    print(json.dumps({"verdict": v["verdict"], "formal_run": payload["formal_run"],
                      "criteria": v["criteria"]}, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------------------------------
# 人类可读报告
# --------------------------------------------------------------------------------------


def write_markdown(p: dict, path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# H2025 H2 正式评估结果（唯一入口一次性输出）\n")
    A("> 授权：导师裁决 K §8 前置审计 + 导师裁决 L「**批准运行 H2025 H2 正式评估**」。  ")
    A("> 生成时间：%s  " % p["created_at"])
    A("> 正式运行：**%s**（bootstrap 重复数 = %d，seed = %d）\n"
      % (p["formal_run"], p["n_boot"], p["bootstrap_seed"]))

    A("## 0. 判定\n")
    A("**结论：`%s`**\n" % p["decision"]["verdict"])
    A("| 条件 | 成立 |")
    A("|---|---|")
    for k, vv in p["decision"]["criteria"].items():
        A("| %s | %s |" % (k, "✅" if vv else "❌"))
    A("")
    A("判据来源：%s。**不报告 p 值**。\n" % p["decision"]["rule_source"])

    A("## 1. Primary（R1）：Delta(s,g) = M(C_seed) − M(A_seed)\n")
    A("30 个重复条件 = 3 training seeds × 10 gallery realizations；"
      "**30 是重复条件数，不是独立样本**；唯一独立簇 = 16 身份。\n")
    for metric in METRICS:
        m = p["primary_contrast_R1"]["per_metric"][metric]
        A("### 1.%s  %s\n" % (METRICS.index(metric) + 1, metric))
        A("| 量 | 点估计 | 95%% CI | 判据 |")
        A("|---|---|---|---|")
        A("| Δ_1g (H2a) | %+.4f | [%+.4f, %+.4f] | %s |"
          % (m["H2a_point_estimate"], *m["H2a_ci95"],
             "supported" if m["H2a"]["supported"] else "not supported"))
        A("| T = Δ_1g − Δ_full (H2b) | %+.4f | [%+.4f, %+.4f] | %s |"
          % (m["H2b_T_point_estimate"], *m["H2b_ci95"],
             "supported" if m["H2b"]["supported"] else "not supported"))
        A("")
        A("| regime | Δ 均值（跨重复条件） | n_repeats | 正 / 负 / 零 |")
        A("|---|---|---|---|")
        for g in REGIMES:
            d = m["per_regime_delta"][g]
            A("| %s | %+.4f | %d | %d / %d / %d |"
              % (g, d["delta_mean_over_repeats"], d["n_repeats"],
                 d["n_positive"], d["n_negative"], d["n_zero"]))
        A("")

    A("## 2. 次级稳健性（R2）：M(C_ens) − M(D)\n")
    A("**不进入** H2a/H2b Primary 判定，**不得**覆盖或替代 R1；重复条件数 = **10**"
      "（`s_ens` 不随 seed 变化，只沿 gallery 实现维度）。\n")
    m = p["secondary_robustness_R2"]["per_metric"][PRIMARY_METRIC]
    A("| 量 | 点估计 | 95%% CI |")
    A("|---|---|---|")
    A("| Δ_1g | %+.4f | [%+.4f, %+.4f] |" % (m["H2a_point_estimate"], *m["H2a_ci95"]))
    A("| Δ_full | %+.4f | — |"
      % m["per_regime_delta"]["full"]["delta_mean_over_repeats"])
    A("")

    A("## 3. ConvNeXt robustness replication：per-seed M(C′_seed) − M(A′_seed)\n")
    A("仅作 H2a/H2b replication 报告；**不作主干比较**，**不形成 dual-backbone 结论**。\n")
    m = p["convnext_replication"]["per_metric"][PRIMARY_METRIC]
    A("| 量 | 点估计 | 95%% CI | supported |")
    A("|---|---|---|---|")
    A("| Δ_1g (H2a) | %+.4f | [%+.4f, %+.4f] | %s |"
      % (m["H2a_point_estimate"], *m["H2a_ci95"], m["H2a"]["supported"]))
    A("| T (H2b) | %+.4f | [%+.4f, %+.4f] | %s |"
      % (m["H2b_T_point_estimate"], *m["H2b_ci95"], m["H2b"]["supported"]))
    A("")

    A("## 4. 机制诊断（1g 必报，并与 full 并列）\n")
    A("| regime | rescue | damage | net rescue |")
    A("|---|---|---|---|")
    for g in REGIMES:
        a = p["net_rescue"][g]["bootstrap"]["query_weighted"]["estimate"]
        agg = p["mechanism_diagnostics"][g]["aggregate_over_seeds"]
        A("| %s | %d | %d | %d（率差 %+.4f，CI [%+.4f, %+.4f]）|"
          % (g, agg["rescue"], agg["harm"], agg["net_rescue"], a,
             *p["net_rescue"][g]["bootstrap"]["query_weighted"]["ci95"]))
    A("")
    A("> net rescue 的 bootstrap 判据（条件 ③）以 **1g** 的 query 视角为准；"
      "同时给出身份宏平均视角（见 JSON）。\n")

    A("## 5. 数据与措辞披露\n")
    A("- holdout 命名：`%s`；**not used for model selection**。"
      % p["data_scope"]["holdout_name"])
    A("- D3 披露：%s" % p["data_scope"]["d3_disclosure"])
    A("- 协议措辞：`%s`" % p["data_scope"]["protocol_wording"])
    A("- 声明：%s" % p["data_scope"]["protocol_disclosure"])
    A("- 术语与判据：见 `H2025_TERMINOLOGY_ERRATUM_v2.md` 与预注册 v3 §8。")
    A("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
