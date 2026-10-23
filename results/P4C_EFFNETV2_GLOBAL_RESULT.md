# P4-C — Modern Global 一次性正式评价

**时间**：`2026-10-23T20:40:20`

> 一次性打开 frozen CFF outer-eval endpoint。**两个预冻结 contrast，无第三个 confirmatory contrast。**

## 1. 端点与纪律

| 项 | 值 |
|---|---|
| endpoint | **256 IDs / 768 gallery / 1104 query / 1872 images** |
| retrieval | **fold-local**；先在每 fold 完整候选集上冻结 outcomes，再 pooled |
| aggregation | identity score = **gallery max**；tie = **optimistic** |
| fusion | **0.5·s_modern + 0.5·s_local**（α 固定，无 dynamic weighting）|
| bootstrap | identity cluster，B=**10000**，seed=**20260922**
| MobileNet | **直读 P3-C 已封存 per-query outcomes，未重跑** |
| Local | **直读 P3-B frozen FV，未重提 / 未重拟合** |

## 2. 三个 arm 的完整结果（三个 seed 全部报告）

| arm | seed | Rank-1 | Macro-Rank-1 | MRR |
|---|---|---|---|---|
| DINOv3 Global | 42 | **0.9719** | 0.9792 | 0.9829 |
| DINOv3 + Local | 42 | **0.9746** | 0.9788 | 0.9851 |
| DINOv3 Global | 43 | **0.9737** | 0.9767 | 0.9844 |
| DINOv3 + Local | 43 | **0.9783** | 0.9800 | 0.9873 |
| DINOv3 Global | 44 | **0.9701** | 0.9690 | 0.9798 |
| DINOv3 + Local | 44 | **0.9710** | 0.9670 | 0.9808 |
| MobileNet Global（**P3-C 封存**）| 42 | 0.9067 | 0.9068 | 0.9375 |
| MobileNet Global（**P3-C 封存**）| 43 | 0.9130 | 0.9244 | 0.9425 |
| MobileNet Global（**P3-C 封存**）| 44 | 0.8958 | 0.9262 | 0.9332 |

## 3. Contrast S — representation strength

| seed | R1_DINOv3 | R1_MobileNet | Δ_s |
|---|---|---|---|
| 42 | 0.9719 | 0.9067 | **+0.0652** |
| 43 | 0.9737 | 0.9130 | **+0.0607** |
| 44 | 0.9701 | 0.8958 | **+0.0743** |

**Δ_Global = mean_seed = +0.066727** · 95% CI **[+0.048273, +0.086378]** ⇒ **SUPPORTED**

## 4. Contrast C — Local complementarity under the modern Global

| seed | R1_DINOv3+Local | R1_DINOv3 | Δ_s |
|---|---|---|---|
| 42 | 0.9746 | 0.9719 | **+0.0027** |
| 43 | 0.9783 | 0.9737 | **+0.0045** |
| 44 | 0.9710 | 0.9701 | **+0.0009** |

**Δ_Local|Modern = mean_seed = +0.002717** · 95% CI **[-0.003543, +0.009180]** ⇒ **NOT supported**

## 5. 联合解释（按 P4-A 冻结规则）

> ### **Case B**
>
> Complementarity remained directionally positive but was not statistically supported under the stronger modern Global.

**Global 命名**：empirically stronger Global under the frozen CFF protocol

> `consistent with` ≠ 因果证明。

## 6. Fold consistency（仅 descriptive，不作 n=5 检验）

| fold | S mean Δ | 方向 | C mean Δ | 方向 |
|---|---|---|---|---|
| fold0 | +0.064472 | + | -0.005487 | - |
| fold1 | +0.071212 | + | +0.003030 | + |
| fold2 | +0.056836 | + | -0.001536 | - |
| fold3 | +0.063863 | + | +0.003115 | + |
| fold4 | +0.077778 | + | +0.015873 | + |

**S 正方向 fold：5/5 · C 正方向 fold：3/5**

## 7. Training-stability disclosure

- P4-B inner-val 训练存在**明显的 seed/run variability** ⇒ **三个 seed 全部单独报告**
- **未删除任何 seed**；该 variability **不构成**删除或重训的理由
- 论文措辞必须限定：**under the frozen fine-tuning recipe**
- 🚫 不得把结果泛化为对 DINOv3 架构 / 基础预训练的普遍性能判断

## 8. 边界

- **未**重训 / 重选 checkpoint / 调 recipe / 换 backbone / 改 α / 删 seed / 删 fold
- **未**新增第三个 confirmatory contrast
- MobileNet 与 Local 侧**只读封存件**，未重跑、未重拟合
- Macro-Rank-1 / MRR 为 **secondary descriptive**，不覆盖 primary 解释
- 🔴 **STOP** —— 不自动进入 recent cattle/animal ReID baseline
