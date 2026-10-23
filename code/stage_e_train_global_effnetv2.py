#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Stage E copy of stage_g_train_global.py with ONE added backbone branch
# (efficientnet_v2_s) and an --out-root redirect.  Everything else is byte-identical
# to the frozen original (sha256 7573791d9b5bbbcfba520a854cca4bb5dbd027456dc995ee3b7e4fed416c2f91).  The original is NOT modified.
"""Stage G —— 从零训练强全局基线（MobileNetV2 / ConvNeXt-Tiny + ArcFace）。

纪律
    * FP32：tf32_matmul / tf32_cudnn 显式关闭。
    * checkpoint 选择**只用 inner-validation**（train identities 内部身份留出），不碰 legacy dev / official test。
    * 不使用任何旧 checkpoint / 旧 embedding cache。
    * 增强配方写死在 config 里，不做结构搜索。

梯度累积（仅 ConvNeXt-Tiny robustness replication 使用；见 v3 §2.2）
    * `--batch` = **micro_batch**（DataLoader batch_size）；`--grad-accum` = 累积步数；
      二者乘积 = **effective_batch**。学习率按 **effective_batch** 线性缩放。
    * 本脚本使用的措辞固定为：
        **hardware-adapted effective-batch-64 training without BatchNorm
        running-statistics dependence**
      **不得**声称与真实 batch=64 "严格数学 / bitwise 等价"——
      浮点累积顺序与随机算子仍可能导致**非逐位**等价（已在 config 中登记该 caveat）。
    * fail-closed：模型内若存在任何 `BatchNorm*` 模块（且 grad_accum > 1）→ 立即终止。
    * fail-closed：主干 `features` 的 state_dict 键集合与形状必须与 torchvision 一致。
    * fail-closed：显存峰值超过 `--vram-limit-gb` → 立即终止。

产物
    models/global/<run_id>/best.pt, last.pt, config.json, history.json, result.json
    features/global/<run_id>_{inner_val,holdout}.npz   （embedding 缓存）
        ⚠️ holdout **只缓存嵌入**，本阶段**不计算、不落盘、不打印**任何 holdout 指标（D3 修复）。

用法
    python stage_g_train_global.py --backbone mobilenetv2 --seed 42
    python stage_g_train_global.py --backbone convnext_tiny --seed 42 \
        --imgsz 224 --batch 8 --grad-accum 8
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r'D:////H2025_clean_restart////scripts')
import h2025_common as C

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# 预训练权重只允许落在本项目自己的缓存目录（禁止读取旧工程 checkpoint）。
import os as _os
_os.environ.setdefault("TORCH_HOME",
                       r"D:\H2025_clean_restart\models\global\_torch_home")


def build_transform(train, imgsz):
    from torchvision import transforms
    ops = []
    if train:
        ops += [transforms.Resize((imgsz, imgsz)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=(0.5, 1.5))]
    else:
        ops += [transforms.Resize((imgsz, imgsz))]
    ops += [transforms.ToTensor()]
    if train:
        ops += [transforms.RandomErasing(p=0.5, scale=(0.02, 0.25),
                                        ratio=(0.3, 3.3), value=(0.485, 0.456, 0.406))]
    ops += [transforms.Normalize(IMG_MEAN, IMG_STD)]
    return transforms.Compose(ops)


class FaceSet:
    def __init__(self, rel_paths, label_map, train, imgsz):
        self.rel = list(rel_paths)
        self.labels = [label_map[r.split("/")[1]] for r in self.rel]
        self.tf = build_transform(train, imgsz)

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        with Image.open(C.DATA_ROOT / self.rel[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]


def make_model(backbone, n_classes, emb=128):
    import torch                      # Net.__init__ 用 torch.empty、forward 用 torch.flatten
    import torch.nn as nn
    from torchvision import models
    if backbone == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        feat_dim, trunk = 1280, m.features
    elif backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feat_dim, trunk = 768, m.features
    elif backbone == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        feat_dim, trunk = 1280, m.features
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = 2048
        trunk = nn.Sequential(*(list(m.children())[:-2]))
    else:
        raise ValueError(backbone)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.emb = nn.Linear(feat_dim, emb)
            self.weight = nn.Parameter(torch.empty(n_classes, emb))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x):
            import torch.nn.functional as F
            f = self.trunk(x)
            if f.dim() > 2:
                f = self.pool(f)
            f = torch.flatten(f, 1)
            return F.normalize(self.emb(f), dim=1)

        def logits(self, z, y, margin, scale):
            import torch, torch.nn.functional as F
            cos = F.linear(z, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-6, 1 - 1e-6)
            phi = cos * math.cos(margin) - torch.sqrt((1 - cos.square()).clamp_min(1e-6)) * math.sin(margin)
            phi = torch.where(cos > math.cos(math.pi - margin), phi,
                              cos - math.sin(math.pi - margin) * margin)
            mask = F.one_hot(y, num_classes=self.weight.shape[0]).bool()
            return torch.where(mask, phi, cos) * scale

    return Net()


def assert_no_batchnorm(net, enforce=True):
    """fail-closed：梯度累积的前提是模型中**没有** BatchNorm 运行统计量依赖。

    ConvNeXt-Tiny 仅使用 LayerNorm，故累积梯度与 effective-batch 等价。
    若发现任何 `BatchNorm*` 模块 → 在 enforce=True 时**立即终止**。
    """
    import torch.nn as nn
    found = []
    for name, m in net.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            found.append("%s (%s)" % (name or "<root>", type(m).__name__))
    if found and enforce:
        raise RuntimeError(
            "fail-closed：检测到 BatchNorm 模块 %s；梯度累积机制对其不成立。" % found)
    return {"has_batchnorm": bool(found), "layers": found[:20], "enforced": bool(enforce)}


def assert_backbone_structure(net, backbone, emb=128):
    """fail-closed：主干 `features` 的 state_dict 键集合与形状必须与 torchvision 一致
    （即**只**替换了分类头 / 只加了嵌入层，未改动主干结构）。"""
    import torch.nn as nn
    from torchvision import models
    if backbone == "mobilenetv2":
        ref = models.mobilenet_v2(weights=None)
    elif backbone == "convnext_tiny":
        ref = models.convnext_tiny(weights=None)
    elif backbone == "efficientnet_v2_s":
        ref = models.efficientnet_v2_s(weights=None)
    elif backbone == "resnet50":
        ref = models.resnet50(weights=None)
        ref = nn.Sequential(*(list(ref.children())[:-2]))
        want = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
        got = {k: tuple(v.shape) for k, v in net.trunk.state_dict().items()}
        ok = want == got
        if not ok:
            raise RuntimeError("fail-closed：主干 state_dict 与 torchvision 不一致")
        return {"checked": True, "n_keys": len(got), "match": True}
    want = {"trunk." + k[len("features."):]: tuple(v.shape)
            for k, v in ref.state_dict().items() if k.startswith("features.")}
    got = {"trunk." + k: tuple(v.shape) for k, v in net.trunk.state_dict().items()}
    if want != got:
        diff = sorted(set(want) ^ set(got))[:10]
        raise RuntimeError(
            "fail-closed：主干 state_dict 与 torchvision %s 不一致（键/形状差异样例 %s）"
            % (backbone, diff))
    if int(net.emb.out_features) != int(emb):
        raise RuntimeError("fail-closed：嵌入维度不是 %d" % emb)
    return {"checked": True, "n_keys": len(got), "match": True,
            "emb_dim": int(net.emb.out_features),
            "classifier_replaced": True}


@__import__("torch").no_grad()
def embed(net, rel_paths, label_map, imgsz, device, bs=32):
    import torch
    from torch.utils.data import DataLoader
    net.eval()
    ds = FaceSet(rel_paths, label_map, False, imgsz)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    out = []
    for x, _ in dl:
        out.append(net(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def retrieval_metrics(q_emb, q_lab, g_emb, g_lab):
    """身份级聚合（每身份取最大 gallery 相似度）后的 Rank-1/5、mAP、macro Rank-1。"""
    import numpy as np
    sim = q_emb @ g_emb.T
    gids = sorted(set(g_lab))
    id2cols = {k: [i for i, l in enumerate(g_lab) if l == k] for k in gids}
    S = np.stack([sim[:, cols].max(1) for cols in id2cols.values()], 1)
    order = np.argsort(-S, axis=1, kind="stable")
    ranked = np.array(gids)[order]
    hits = ranked == np.asarray(q_lab)[:, None]
    r1 = hits[:, 0]
    r5 = hits[:, :5].any(1)
    denom = np.arange(1, len(gids) + 1)
    ap = (hits * np.cumsum(hits, 1) / denom).sum(1) / np.maximum(hits.sum(1), 1)
    macro = float(np.mean([r1[np.asarray(q_lab) == k].mean() for k in set(q_lab)]))
    return {"rank1": float(r1.mean()), "rank5": float(r5.mean()),
            "mAP": float(ap.mean()), "macro_rank1": macro,
            "n_queries": int(len(q_lab)), "n_gallery_ids": len(gids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilenetv2",
                    choices=["mobilenetv2", "convnext_tiny", "resnet50",
                             "efficientnet_v2_s"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=112)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--arc_margin", type=float, default=0.5)
    ap.add_argument("--arc_scale", type=float, default=30.0)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="梯度累积步数（micro_batch=batch）。effective_batch = batch × grad_accum")
    ap.add_argument("--vram-limit-gb", type=float, default=4.0,
                    help="显存峰值硬上限；超过即 fail-closed 终止（本机 4.29GB）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-batchnorm-accum", action="store_true",
                    help="stage E: allow gradient accumulation with BatchNorm backbones; the "
                         "deviation (BN statistics over the micro-batch) is recorded in config.json")
    ap.add_argument("--out-root", default=None,
                    help="stage E: write run artefacts here instead of models/global")
    args = ap.parse_args()
    if args.grad_accum < 1:
        raise SystemExit("--grad-accum 必须 >= 1")

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    C.set_seed(args.seed)
    prec = C.disable_tf32()
    splits = C.json_load(C.ROOT / "splits" / "h2025_clean_splits.json")
    micro_batch = args.batch
    grad_accum = args.grad_accum
    eff_batch = micro_batch * grad_accum
    run_id = ("%s_s%d_img%d_emb%d" % (args.backbone, args.seed, args.imgsz, args.emb)
              + ("_mb%dx%d" % (micro_batch, grad_accum) if grad_accum > 1 else ""))
    if args.out:
        run_id = args.out          # 仅用于离线 smoke / 自检，**正式 run 必须用 canonical run_id**
    out_root = Path(args.out_root) if args.out_root else (C.ROOT / "models" / "global")
    run_dir = C.ensure_dir(out_root / run_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    train_rel = splits["train_pool"]
    label_map = {i: n for n, i in enumerate(sorted({p.split("/")[1] for p in train_rel}))}
    n_classes = len(label_map)
    if micro_batch > len(train_rel):
        micro_batch = max(4, len(train_rel))

    # ------------------------------------------------------------------ #
    # fail-closed 结构检查（GLOBAL_TRAINING_PREREGISTRATION_v3 §2.1/§2.2/§6）
    net = make_model(args.backbone, n_classes, args.emb).to(device)
    struct = assert_backbone_structure(net, args.backbone)
    bn = assert_no_batchnorm(net, enforce=(grad_accum > 1 and not args.allow_batchnorm_accum))
    if grad_accum > 1 and args.allow_batchnorm_accum and bn.get("has_batchnorm"):
        print("[STAGE-E DEVIATION] BatchNorm running statistics are accumulated over the "
              "micro-batch (%d), not over the effective batch (%d).  This is an explicit, "
              "recorded deviation from the effective-batch-64 protocol used by MobileNetV2; "
              "it is NOT equivalent to a single batch-64 forward/backward." %
              (micro_batch, eff_batch), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = {
        "run_id": run_id, "backbone": args.backbone, "seed": args.seed,
        "epochs": args.epochs, "batch": micro_batch, "imgsz": args.imgsz,
        "micro_batch": micro_batch, "gradient_accumulation_steps": grad_accum,
        "effective_batch": eff_batch,
        "lr": args.lr, "weight_decay": args.wd, "embedding": args.emb,
        "arc_margin": args.arc_margin, "arc_scale": args.arc_scale,
        "optimizer": "AdamW", "scheduler": "none",
        "augmentation": "HFlip + ColorJitter(brightness .5-1.5) + RandomErasing(p=.5,area .02-.25)",
        "normalize": {"mean": IMG_MEAN, "std": IMG_STD},
        "loss": "ArcFace(cosface-style additive angular margin)",
        "checkpoint_selection": "inner-validation mAP (身份留出; 不使用 legacy dev / official test)",
        "n_classes": n_classes, "n_train_images": len(train_rel),
        "split_sha256": C.json_load(C.ROOT / "locks" / "split_lock.json")["split_sha256"],
        "precision": prec, "device": device,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backbone_structure_check": struct,
        "batchnorm_check": bn,
        "stage_e_deviation": ({
            "type": "batchnorm_running_statistics_group_size",
            "micro_batch": micro_batch, "effective_batch": eff_batch,
            "statement": "BatchNorm running statistics are updated per micro-batch (%d) rather "
                         "than per effective batch (%d); the training is therefore NOT equivalent "
                         "to the effective-batch-64 protocol used for the MobileNetV2 baseline."
                         % (micro_batch, eff_batch),
            "reason": "4.29 GB VRAM cannot hold EfficientNetV2-S at 224 px with batch 64",
        } if (grad_accum > 1 and args.allow_batchnorm_accum and bn.get("has_batchnorm")) else None),
        "stage_e_script": {
            "based_on": "D:/H2025_clean_restart/scripts/stage_g_train_global.py",
            "original_sha256": "7573791d9b5bbbcf" ,
            "changes": ["added backbone branch efficientnet_v2_s (torchvision IMAGENET1K_V1)",
                        "added --out-root", "added --allow-batchnorm-accum (recorded deviation)"],
        },
        "accumulation_semantics": (
            "hardware-adapted effective-batch-%d training without BatchNorm "
            "running-statistics dependence" % eff_batch) if grad_accum > 1 else "n/a (no accumulation)",
        "accumulation_equivalence_caveat": (
            "float accumulation order and non-deterministic kernels may still make this "
            "NOT bitwise identical to a single batch-%d forward/backward" % eff_batch)
        if grad_accum > 1 else None,
        "code_sha256": C.sha256_file(Path(__file__)),
        "created_at": C.now_iso(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("[cfg] %s  classes=%d train=%d micro=%d accum=%d eff=%d device=%s"
          % (run_id, n_classes, len(train_rel), micro_batch, grad_accum, eff_batch, device),
          flush=True)
    print("[check] structure=%s batchnorm=%s"
          % (json.dumps(struct, ensure_ascii=False), json.dumps(bn, ensure_ascii=False)),
          flush=True)

    ds = FaceSet(train_rel, label_map, True, args.imgsz)
    dl = DataLoader(ds, batch_size=micro_batch, shuffle=True, num_workers=0, drop_last=False)
    lr = args.lr * (eff_batch / 64.0)   # 按 **effective batch** 线性缩放
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.wd)

    vg, vq = splits["inner_val_gallery"], splits["inner_val_query"]
    lab_map_v = {i: i for i in splits["inner_val_identities"]}

    def inner_val(net_):
        ge = embed(net_, vg, lab_map_v, args.imgsz, device)
        qe = embed(net_, vq, lab_map_v, args.imgsz, device)
        return retrieval_metrics(qe, [p.split("/")[1] for p in vq],
                                 ge, [p.split("/")[1] for p in vg])

    hist = []
    best = None
    # 梯度累积周期划分（**确定性**）：满周期用 grad_accum，末周期按实际 micro-batch 数归一化
    n_micro = len(dl)
    cycle_sizes = []
    _s = n_micro
    while _s > 0:
        _t = min(grad_accum, _s)
        cycle_sizes.append(_t)
        _s -= _t
    peak_gb = 0.0
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        tot, nb, opt_steps = 0.0, 0, 0
        it = iter(dl)
        for csize in cycle_sizes:
            opt.zero_grad(set_to_none=True)
            for _ in range(csize):
                x, y = next(it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = nn.functional.cross_entropy(
                    net.logits(net(x), y, args.arc_margin, args.arc_scale), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss at epoch %d micro-batch %d" % (ep, nb))
                (loss / float(csize)).backward()
                tot += loss.item()
                nb += 1
            opt.step()
            opt_steps += 1
            if torch.cuda.is_available():
                pg = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                peak_gb = max(peak_gb, pg)
                if pg > args.vram_limit_gb:
                    raise RuntimeError(
                        "fail-closed：显存峰值 %.2f GB 超过上限 %.2f GB"
                        % (pg, args.vram_limit_gb))
        m = inner_val(net)
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "lr": lr,
               "optimizer_steps": opt_steps, "micro_batches": nb,
               "peak_vram_gb": round(peak_gb, 3),
               "seconds": round(time.time() - t0, 1), **m}
        hist.append(rec)
        (run_dir / "history.json").write_text(
            json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[%s ep%02d] loss=%.4f inner_val R1=%.4f mAP=%.4f (%.0fs)"
              % (run_id, ep, rec["train_loss"], m["rank1"], m["mAP"], rec["seconds"]), flush=True)
        if best is None or m["mAP"] > best["mAP"]:
            best = m
            torch.save({"state_dict": net.state_dict(), "class_to_idx": label_map,
                        "config": config, "epoch": ep, "inner_val": m},
                       run_dir / "best.pt")
    torch.save({"state_dict": net.state_dict(), "class_to_idx": label_map, "config": config},
               run_dir / "last.pt")

    # 冻结的 embedding 缓存（inner_val + holdout），供融合/统计使用
    #
    # 纪律（GLOBAL_TRAINING_PREREGISTRATION §10 / H2025_DEVIATION_LOG D3）：
    #   **holdout 只缓存嵌入，绝不在此阶段计算、落盘或打印任何 holdout 指标。**
    #   holdout 指标统一由评估阶段从已锁定的 features/global/*_holdout.npz 重算，
    #   以避免"训练日志泄漏正式评估数字"导致结论可被指为受其结果影响。
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    bundle = {"run_id": run_id, "config": config, "best_epoch": ck["epoch"],
              "inner_val_best": best, "history": hist,
              "peak_vram_gb": round(peak_gb, 3),
              "backbone_structure_check": struct,
              "batchnorm_check": bn}
    for tag, gal, qry in (("inner_val", vg, vq), ("holdout", splits["holdout_gallery"],
                                                 splits["holdout_query"])):
        lmap = {i: i for i in sorted({p.split("/")[1] for p in gal + qry})}
        ge = embed(net, gal, lmap, args.imgsz, device)
        qe = embed(net, qry, lmap, args.imgsz, device)
        np.savez_compressed(C.ensure_dir(C.ROOT / "features" / "global") / ("%s_%s.npz" % (run_id, tag)),
                            gallery=ge, query=qe,
                            gallery_labels=np.array([p.split("/")[1] for p in gal]),
                            query_labels=np.array([p.split("/")[1] for p in qry]),
                            gallery_paths=np.array(gal), query_paths=np.array(qry))
        if tag == "inner_val":
            bundle[tag] = retrieval_metrics(qe, [p.split("/")[1] for p in qry],
                                            ge, [p.split("/")[1] for p in gal])
    (run_dir / "result.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("\n[RESULT] inner_val_best = %s"
          % json.dumps(bundle["inner_val_best"], ensure_ascii=False), flush=True)
    print("STAGE_G_OK %s" % run_id)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
