#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage H —— 重算局部描述子（HesAffNet + HardNet++，全冻结）。

纪律
    * 检测器（HesAff/尺度空间）与描述子（HardNet++）**全部冻结**，eval + no_grad。
    * 预训练权重只读 `D:\\复现论文\\reproduction\\models\\`（公开权重）。
    * **禁止**读取旧工程任何 local descriptor cache / FV encoding。
    * 收编的参考代码不改动；运行前校验 `code/local_fv/SOURCE_SHA256.txt`。

设计（写死，不做自适应）
    * 输入：**原图灰度**（不做统一 resize —— 尺度本身是要编码的信息）。
    * 检测器：ScaleSpaceAffinePatchExtractor(mrSize=9.348280997446642,
      init_sigma=1.3213713243956968, nlevels=10, border=5, num_Baum_iters=26)
      —— 逐项对齐作者 `Norppa/config.py` 的 config["hesaff_args"]
    * 每图关键点上限：--num-features（默认 480，即论文协议值，**不按图像自适应**）
    * 描述子：HardNet++，128 维，patch 32×32
    * 输出：每图 (n_i, 128) float32 描述子 + (n_i, 2, 3) LAF

用法
    python stage_h_local_descriptors.py --subset train_pool --limit 8 --selftest
    python stage_h_local_descriptors.py --subset train_pool
    python stage_h_local_descriptors.py --subset inner_val
    python stage_h_local_descriptors.py --subset holdout
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2025_common as C

VENDOR = (C.ROOT / "code" / "local_fv" / "hesaff_pytorch")
if not VENDOR.is_dir():
    VENDOR = C.ROOT / "third_party" / "local_fv" / "hesaff_pytorch"
sys.path.insert(0, str(VENDOR))

WEIGHT_DIR = Path(r"D:\复现论文\reproduction\models")
W_AFFNET = WEIGHT_DIR / "AffNet.pth"
W_HARDNET = WEIGHT_DIR / "HardNet++.pth"
W_ORINET_CANDIDATES = [WEIGHT_DIR / "OriNet.pth", WEIGHT_DIR / "OriNet_weights.pth"]

# --------------------------------------------------------------------------- #
# 检测器参数 —— **必须与作者 Norppa 权威配置一致**
#
# 来源：`D:\复现论文\Norppa\config.py` 的 config["hesaff_args"]
#       + `Norppa/reidentification/hesaff_pytorch/HessianAffinePatches.py`
#         的 extract_hesaff_patches() 函数默认值（配置未覆盖者取默认）。
#
#   config["hesaff_args"] = {'init_sigma': 1.3213713243956968,
#                            'mrSize': 9.348280997446642,
#                            'nlevels': 10,
#                            'num_features': 480,
#                            'unsharp_amount': 6.80631647207343,
#                            'unsharp_radius': None,
#                            'patch_size': 32}
#
# 调用方式：本脚本**直接调用**作者的 extract_hesaff_patches()（与 alfreid.py 相同），
#       因此需要 scikit-image（作者模块顶层 `from skimage.filters import unsharp_mask`）。
#       参数值逐项对齐作者配置，不做任何"顺手调参"。
#
#   * unsharp_radius=None ⇒ 作者代码里 unsharp_mask 分支不执行，
#     因此 unsharp_amount 虽被传入但**不产生数值影响**。
#   * num_Baum_iters 与 border 未被 config 覆盖，取函数默认 26 / 5。
#   * patch 由该函数**从尺度金字塔**采样（不是在全分辨率原图上采），
#     这是必须调用作者函数、而非自行复刻的关键原因。
MRSIZE = 9.348280997446642          # 作者 config.py
INIT_SIGMA = 1.3213713243956968     # 作者 config.py
NLEVELS = 10                        # 作者 config.py
# Top64：H2025 H2 的 primary local representation（导师裁决 2026-09-16）
#   注意：作者 config.py 里的 480 是 **参考工程 ALFRE-ID** 的论文协议值，
#         本轮 H2 的 local 表示固定为 Top64，故此处默认 64。
#         480 口径**不得**在本轮混用（如需须另行预注册）。
NUM_FEATURES_DEFAULT = 64
TOPK_BY_RESPONSE = 64
PREFILTER_FACTOR = 1.5              # SparseImgRepresenter.py:206  int(1.5 * num_features)
TRUNCATION_LOCUS = ("author internal: multiScaleDetector NMS-topk(prefilter) "
                    "+ getAffineShape topk(num_features)")
NATIVE_SEMANTICS = ("HesAffNet-native response-selected Top64: 全体候选 → prefilter "
                    "int(1.5*64)=96 → AffNet/Baum/border/ratio 存活筛选 → 对存活 response "
                    "再 topk 至最多 64 → 不足 64 全部保留（不复制、不补点）")
BORDER = 5                          # extract_hesaff_patches 默认
NUM_BAUM_ITERS = 26                 # extract_hesaff_patches 默认
PATCH_SIZE = 32                     # 作者 config.py
# HardNet++ 前向 batch：与参考实现 alfreid.py::Extractor(batch_size=64) 一致
HARDNET_BATCH = 64
UNSHARP_RADIUS = None               # 作者 config.py（None ⇒ 不做 unsharp）
UNSHARP_AMOUNT = 6.80631647207343   # 作者 config.py（因 radius=None 而不生效）
PARAM_SOURCE = ("Norppa/config.py:config['hesaff_args'] + "
                "HessianAffinePatches.py:extract_hesaff_patches defaults")


# --------------------------------------------------------------------------- #
def verify_vendor():
    src = VENDOR / "SOURCE_SHA256.txt"
    # SOURCE_SHA256.txt 位于上一级（code/local_fv/）
    if not src.is_file():
        src = VENDOR.parent / "SOURCE_SHA256.txt"
    if not src.is_file():
        return {"checked": False, "reason": "SOURCE_SHA256.txt not found"}
    bad = []
    n = 0
    missing = []
    # 注意：SOURCE_SHA256.txt 由 PowerShell 写出，带 UTF-8 BOM。
    # 若用 encoding="utf-8" 读取，首行的 parts[0] 会多出一个 \ufeff（长度 65 而非 64），
    # 于是硬编码的 `len(parts[0]) == 64` 守卫会**静默跳过第一个文件**（实测即 HardNet.py）。
    # 用 utf-8-sig 剥离 BOM，并对首行做二次防御。
    for line in src.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        parts[0] = parts[0].lstrip("\ufeff")
        if len(parts) >= 3 and len(parts[0]) == 64:
            h, _, name = parts[0], parts[1], parts[2]
            fp = VENDOR / name
            if fp.is_file():
                n += 1
                if C.sha256_file(fp) != h:
                    bad.append(name)
            else:
                missing.append(name)
    return {"checked": True, "n_files": n, "mismatch": bad, "missing": missing}


def load_models(device):
    """按作者 `HessianAffinePatches.init_*` 加载三个冻结网络。

    刻意**不复刻**权重加载逻辑：直接调用作者函数，保证 state_dict 键名、
    patch_size、eval()/cuda 行为与参考实现完全一致。
    """
    from HessianAffinePatches import init_affnet, init_orinet, init_hardnet
    use_cuda = str(device).startswith("cuda")
    info = {"loader": "HessianAffinePatches.init_affnet/init_orinet/init_hardnet (author)"}

    aff = init_affnet(str(W_AFFNET), patch_size=PATCH_SIZE, use_cuda=use_cuda)
    info["affnet"] = {"path": str(W_AFFNET), "sha256": C.sha256_file(W_AFFNET)}

    ori = None
    for cand in W_ORINET_CANDIDATES:
        if cand.is_file():
            ori = init_orinet(str(cand), patch_size=PATCH_SIZE, use_cuda=use_cuda)
            info["orinet"] = {"path": str(cand), "sha256": C.sha256_file(cand)}
            break
    if ori is None:
        # 作者配置显式传入 OriNet；缺失会静默改变 LAF 方向 → 直接失败，不降级
        raise RuntimeError("缺 OriNet 权重，作者配置要求 OriNet：%s" % W_ORINET_CANDIDATES)

    hn = init_hardnet(str(W_HARDNET), use_cuda=use_cuda)
    info["hardnet"] = {"path": str(W_HARDNET), "sha256": C.sha256_file(W_HARDNET)}
    info["patch_size"] = PATCH_SIZE
    return aff, ori, hn, info


def extract_one(aff, ori, hardnet, abs_path, device, num_features):
    """作者路径：PIL convert('L') → extract_hesaff_patches → HardNet++。

    与 `alfreid.py::Extractor.__call__` 的 `name == 'hesaff'` 分支逐行对应：

        patches, ell = extract_hesaff_patches(image.convert('L'), num_features=...,
                        nlevels=10, mrSize=..., init_sigma=..., patch_size=32,
                        unsharp_radius=None, unsharp_amount=..., AffNet=, OriNet=)
        t = as_tensor(patches/255)[:, None]
        descriptors = hardnet(t)
        xy = asarray(ell)[:, :2]

    关键点：patch **从尺度金字塔**按 LAF 级别采样（由作者函数内部完成），
    而不是在全分辨率原图上直接采样 —— 后者会忽略尺度，是实质性的数值偏差。
    """
    from PIL import Image
    from HessianAffinePatches import extract_hesaff_patches
    use_cuda = str(device).startswith("cuda")

    with Image.open(abs_path) as im:
        gray = im.convert("L")                    # 作者一致：ITU-R 601-2 luma
    with torch.no_grad():
        patches, ell = extract_hesaff_patches(
            gray, num_features=int(num_features), nlevels=NLEVELS,
            mrSize=MRSIZE, init_sigma=INIT_SIGMA, patch_size=PATCH_SIZE,
            unsharp_radius=UNSHARP_RADIUS, unsharp_amount=UNSHARP_AMOUNT,
            AffNet=aff, OriNet=ori, use_cuda=use_cuda)
        if patches is None or len(patches) == 0:
            return np.zeros((0, 128), "float32"), np.zeros((0, 5), "float32"), 0
        blocks = []
        for i in range(0, len(patches), HARDNET_BATCH):
            chunk = patches[i:i + HARDNET_BATCH] / 255.0
            tt = torch.as_tensor(chunk, dtype=torch.float32, device=device)[:, None]
            blocks.append(hardnet(tt).cpu().numpy())
        desc = np.concatenate(blocks).astype("float32")   # HardNet 内部已 L2Norm
    ell = np.asarray(ell, dtype="float32").reshape(-1, 5)  # (n, [x, y, s1, s2, angle])
    return desc, ell, int(len(desc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    choices=["train_pool", "inner_val_gallery", "inner_val_query",
                             "holdout_gallery", "holdout_query", "inner_val", "holdout",
                             "dev_probe"])
    ap.add_argument("--num-features", type=int, default=NUM_FEATURES_DEFAULT,
                    help="每图关键点上限。H2 primary 口径固定为 64（Top64）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 张（自检用）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--allow-non-top64", action="store_true",
                    help="显式允许 num_features != 64（仅用于非 H2 的独立实验）")
    args = ap.parse_args()

    if args.num_features != TOPK_BY_RESPONSE and not args.allow_non_top64:
        raise SystemExit(
            "拒绝：H2 的 local representation 固定为 Top64（num_features=%d），收到 %d。\n"
            "如需其它口径（例如参考工程的 480），须**另行预注册**并显式加 --allow-non-top64。"
            % (TOPK_BY_RESPONSE, args.num_features))

    C.set_seed(42)
    prec = C.disable_tf32()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    vend = verify_vendor()
    print("[vendor] %s" % json.dumps(vend, ensure_ascii=False), flush=True)
    if vend.get("checked") and vend.get("mismatch"):
        raise RuntimeError("收编代码哈希不匹配：%s" % vend["mismatch"])

    splits = C.json_load(C.ROOT / "splits" / "h2025_clean_splits.json")
    if args.subset == "inner_val":
        rels = splits["inner_val_gallery"] + splits["inner_val_query"]
    elif args.subset == "holdout":
        rels = splits["holdout_gallery"] + splits["holdout_query"]
    elif args.subset == "dev_probe":
        rels = splits["train_pool"][:12]
    else:
        rels = splits[args.subset]
    if args.limit > 0:
        rels = rels[:args.limit]

    aff, ori, hn, winfo = load_models(device)
    print("[model] %s" % json.dumps(winfo, ensure_ascii=False)[:600], flush=True)
    print("[run] subset=%s n=%d num_features=%d device=%s" %
          (args.subset, len(rels), args.num_features, device), flush=True)

    descs, ells, paths, counts = [], [], [], []
    t0 = time.time()
    for i, rel in enumerate(rels):
        d, l, n = extract_one(aff, ori, hn, C.DATA_ROOT / rel, device, args.num_features)
        descs.append(d)
        ells.append(l)
        paths.append(rel)
        counts.append(n)
        if (i + 1) % 50 == 0 or args.selftest:
            el = time.time() - t0
            print("  [%5d/%5d] %4.1fs  last_n=%d  mean_n=%.1f"
                  % (i + 1, len(rels), el, n, float(np.mean(counts))), flush=True)
    el = time.time() - t0

    out = {"paths": np.array(paths),
           "counts": np.array(counts, "int32"),
           "num_features_cap": args.num_features,
           "top_k_by_response": TOPK_BY_RESPONSE,
           "prefilter_factor": PREFILTER_FACTOR,
           "truncation_locus": TRUNCATION_LOCUS,
           "native_semantics": NATIVE_SEMANTICS,
           "mrSize": MRSIZE, "init_sigma": INIT_SIGMA, "nlevels": NLEVELS,
           "border": BORDER, "num_Baum_iters": NUM_BAUM_ITERS,
           "patch_size": PATCH_SIZE,
           "unsharp_radius": UNSHARP_RADIUS, "unsharp_amount": UNSHARP_AMOUNT,
           "hardnet_batch": HARDNET_BATCH,
           "param_source": PARAM_SOURCE,
           "extractor_fn": "HessianAffinePatches.extract_hesaff_patches (author)",
           "weights": json.dumps(winfo),
           "precision": json.dumps(prec),
           "vendor": json.dumps(vend)}
    # 变长 → 用 object array 存，避免 padding 造成的信息损失
    out["descriptors"] = np.array(descs, dtype=object)   # 每项 (n_i, 128) float32
    out["ellipses"] = np.array(ells, dtype=object)       # 每项 (n_i, 5) [x, y, s1, s2, angle]
    # 注：作者 geometric 校验用的是 xy = ellipses[:, :2]

    tag = args.subset if not args.limit else "%s_limit%d" % (args.subset, args.limit)
    dst = Path(args.out) if args.out else (
        C.ROOT / "features" / "local" / ("%s_descriptors.npz" % tag))
    C.ensure_dir(dst.parent)
    np.savez_compressed(dst, **out)

    meta = {
        "subset": args.subset, "n_images": len(paths),
        "elapsed_sec": round(el, 1),
        "total_descriptors": int(np.sum(counts)),
        "mean_descriptors_per_image": float(np.mean(counts)) if counts else 0.0,
        "min_descriptors": int(np.min(counts)) if counts else 0,
        "max_descriptors": int(np.max(counts)) if counts else 0,
        "zero_descriptor_images": int(np.sum(np.array(counts) == 0)),
        "zero_descriptor_paths": [p for p, n in zip(paths, counts) if n == 0],
        "images_below_cap": int(np.sum(np.array(counts) < args.num_features)),
        "images_at_cap": int(np.sum(np.array(counts) == args.num_features)),
        "images_above_cap": int(np.sum(np.array(counts) > args.num_features)),
        "cap_binding": bool(int(np.max(counts)) >= args.num_features) if counts else False,
        "num_features_cap": args.num_features,
        "top_k_by_response": TOPK_BY_RESPONSE,
        "prefilter_factor": PREFILTER_FACTOR,
        "truncation_locus": TRUNCATION_LOCUS,
        "native_semantics": NATIVE_SEMANTICS,
        "detector_params": {
            "mrSize": MRSIZE, "init_sigma": INIT_SIGMA, "nlevels": NLEVELS,
            "border": BORDER, "num_Baum_iters": NUM_BAUM_ITERS,
            "patch_size": PATCH_SIZE, "unsharp_radius": UNSHARP_RADIUS,
            "unsharp_amount": UNSHARP_AMOUNT, "hardnet_batch": HARDNET_BATCH,
            "param_source": PARAM_SOURCE,
            "extractor_fn": "HessianAffinePatches.extract_hesaff_patches (author)",
        },
        "device": device, "weights": winfo, "vendor": vend,
        "output": str(dst), "output_sha256": C.sha256_file(dst),
        "no_metrics_computed": True,
        "created_at": C.now_iso(),
    }
    mp = C.ensure_dir(C.ROOT / "features" / "local") / ("%s_meta.json" % tag)
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[out] %s" % dst, flush=True)
    print("[out] %s" % mp, flush=True)
    print(json.dumps(meta, ensure_ascii=False, indent=1)[:1600], flush=True)
    print("STAGE_H_OK %s" % tag)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
