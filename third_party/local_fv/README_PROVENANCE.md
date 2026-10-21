# Local FV —— 参考实现来源说明（vendored reference code）

本目录下的实现**不是**本工程原创，而是从**公开参考仓库**收编（vendor）而来，
用于 Stage H / I / J 的局部描述子路径。收编时**逐文件复制、未做任何修改**。

## 来源

| 项 | 值 |
|---|---|
| 参考仓库根 | `D:\复现论文\Norppa`（Norppa —— Nepovinnykh et al. 2024 图案型动物个体重识别官方代码，只读） |
| 相对路径 | `reidentification/hesaff_pytorch/` |
| 收编目标 | `D:\H2025_clean_restart\code\local_fv\hesaff_pytorch\` |
| 收编方式 | 逐文件 `Copy-Item`，**字节级不改动** |
| 校验 | 每个文件的 sha256 记于 `SOURCE_SHA256.txt`（源文件与副本哈希一致） |

## 收录文件与用途

| 文件 | 用途 |
|---|---|
| `SparseImgRepresenter.py` | `ScaleSpaceAffinePatchExtractor` —— Hessian 尺度空间 + 仿射形状估计（支持传入 AffNet/OriNet） |
| `HandCraftedModules.py` | `HessianResp` / `AffineShapeEstimator` / `OrientationDetector` / `ScalePyramid` / `NMS3dAndComposeA` |
| `architectures.py` | `AffNetFast` / `OriNetFast` 等网络定义（用于加载公开 AffNet 权重） |
| `HardNet.py` | HardNet 描述子网络定义 |
| `LAF.py` | 局部仿射框架（LAF ↔ ellipse 转换、patch 采样） |
| `Utils.py` | 数值工具（高斯核、L2Norm、网格生成等） |
| `HessianAffinePatches.py` | 参考仓库自带的最高层封装（**Stage H 不使用**：它 import `skimage.filters.unsharp_mask`，会引入额外依赖；Stage H 直接使用 `SparseImgRepresenter` 路径） |
| `ReprojectonStuff.py` / `pytorch_sift.py` | 参考仓库内的其他工具（未使用，收编以保持包完整性） |

## 公开预训练权重（只读，不下载）

| 权重 | 路径 | 大小 |
|---|---|---|
| AffNet（仿射形状 + 方向） | `D:\复现论文\reproduction\models\AffNet.pth` | 339,871 |
| HardNet++（描述子） | `D:\复现论文\reproduction\models\HardNet++.pth` | 5,345,606 |

> 这两份是**公开预训练权重**，属于归档清单 `reference_roots` 明确允许的只读来源。
> **禁止**读取旧 `D:\牛脸识别` 工程里的任何 local descriptor cache 或 FV encoding。

## 依赖（Stage H 需要）

`torch`, `torchvision`, `numpy`, `opencv-python`（`cv2`）, `Pillow`。
**不需要** `scikit-image`（已刻意避开 `HessianAffinePatches.py`）。

## 纪律

1. **不改动**收编的文件；若必须改，另建 `patches/` 并记录 diff。
2. 运行前校验 `SOURCE_SHA256.txt` 中每个文件的 sha256，不一致即中止。
3. 收编来源路径与哈希写入最终报告的"实现来源"一节。
