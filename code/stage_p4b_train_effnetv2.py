# -*- coding: utf-8 -*-
"""P4-B — train the DINOv3 ConvNeXt-Tiny Global: 5 folds x 3 seeds = 15 runs.

Recipe is byte-frozen from P4-A. The only change from H2025 is the pretrained backbone
weights. Checkpoint selection uses inner-validation identity-level mAP only (ties -> the
earlier epoch); outer_eval never touches training, selection or any decision.

RESUMABLE AND VERIFIABLE: after every epoch the run writes `resume.pt` holding model,
optimizer, epoch index and the RNG states of python/numpy/torch. A restart resumes from
there, so an interrupted run continues under the identical recipe instead of silently
changing it. `--verify-resume` prints whether a resume point exists and what it contains.

This script computes NO retrieval metric on outer_eval. The inner-val mAP it reports is
the frozen checkpoint-selection signal, not a result.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"D:\H2025_clean_restart"
P3A = os.path.join(ROOT, "analysis", "stage_p3a")
P3B = os.path.join(ROOT, "analysis", "stage_p3b")
P4A = os.path.join(ROOT, "analysis", "stage_p4a")
P4B = os.path.join(ROOT, "analysis", "stage_p4b_effnetv2")
RUNS = os.path.join(P4B, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL_ID = "torchvision:efficientnet_v2_s:IMAGENET1K_V1"
F_PROV = os.path.join(P4A, "P4_EFFNETV2_WEIGHT_PROVENANCE.json")
F_NEST = os.path.join(P3A, "CFF_P3A2_NESTED_SPLIT.json")
F_CACHE = os.path.join(P3B, "CFF_P3B_IMAGE_CACHE.json")
F_ROI = os.path.join(ROOT, "analysis", "stage_p2r_det1",
                     "CFF_PRIMARY_ROI_COMPATIBILITY_AUDIT.json")
F_RECORD = os.path.join(P4B, "P4B_EFFNETV2_TRAINING_RECORD.json")
F_SEL = os.path.join(P4B, "P4B_EFFNETV2_CHECKPOINT_SELECTION_RECORD.json")
PROGRESS = os.path.join(P4B, "P4B_EFFNETV2_PROGRESS.log")

SEEDS = [42, 43, 44]
EPOCHS, MICRO, ACCUM, EFF = 20, 16, 4, 64   # BatchNorm statistics over the micro-batch (16), not over the effective batch (64): recorded deviation
IMGSZ, EMB, MARGIN, SCALE = 224, 128, 0.5, 30.0
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
LR, WD = 1e-4, 1e-4


def logline(msg):
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), msg))


def write_atomic(p, text):
    t = p + ".tmp"
    with open(t, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


PROV = json.load(open(F_PROV, encoding="utf-8"))
EXPECT_WEIGHT_SHA = PROV["pretrained_weight_sha256"]
WEIGHT_PATHS = [f["path"] for f in PROV["pretrained_weight_files"]]



def verify_weights():
    """FAIL-CLOSED: the exact frozen DINOv3 weights must be present and unchanged."""
    for p in WEIGHT_PATHS:
        if not os.path.isfile(p):
            raise SystemExit("STOP: frozen pretrained weight missing: %s" % p)
    got = fsha(WEIGHT_PATHS[0])
    if got != EXPECT_WEIGHT_SHA:
        raise SystemExit("STOP: pretrained weight sha256 mismatch\n  got %s\n  want %s"
                         % (got, EXPECT_WEIGHT_SHA))
    return got


def build_transforms():
    from torchvision import transforms
    tr = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 1.5)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3),
                                 value=(0.485, 0.456, 0.406)),
        transforms.Normalize(MEAN, STD)])
    ev = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return tr, ev


class CachedFaceSet(Dataset):
    """Reads the FROZEN ROI at native resolution and applies the 224 transform.

    The 112 cache built for P3-B cannot be reused here because P4 trains at 224; the source
    is the same frozen ROI file, so nothing about the region of interest changes.
    """

    def __init__(self, recs, label_map, train, tf):
        self.recs = list(recs)
        self.labels = [label_map[r["canonical"]] for r in self.recs]
        self.tf = tf

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        p = os.path.join(ROOT, r["source"])
        with Image.open(p) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]


class Net(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.trunk = backbone
        self.emb = nn.Linear(backbone.num_features, EMB)
        self.weight = nn.Parameter(torch.empty(n_classes, EMB))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.normalize(self.emb(self.trunk(x)), dim=1)

    def logits(self, z, y, margin, scale):
        cos = F.linear(z, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-6, 1 - 1e-6)
        phi = cos * math.cos(margin) - torch.sqrt((1 - cos.square()).clamp_min(1e-6)) * math.sin(margin)
        phi = torch.where(cos > math.cos(math.pi - margin), phi,
                          cos - math.sin(math.pi - margin) * margin)
        mask = F.one_hot(y, num_classes=self.weight.shape[0]).bool()
        return torch.where(mask, phi, cos) * scale


def make_net(n_classes, device):
    from torchvision import models
    bb = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    bb.num_features = bb.classifier[-1].in_features   # torchvision exposes it on the classifier
    bb.classifier = nn.Identity()                      # keep the pooled 1280-D features, not the logits
    return Net(bb, n_classes).to(device)


@torch.no_grad()
def embed(net, recs, lm, tf, device, bs=32):
    net.eval()
    dl = DataLoader(CachedFaceSet(recs, lm, False, tf), batch_size=bs, shuffle=False, num_workers=0)
    out = []
    for x, _ in dl:
        out.append(net(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def retrieval_metrics(q_emb, q_lab, g_emb, g_lab):
    sim = q_emb @ g_emb.T
    gids = sorted(set(g_lab))
    id2cols = {k: [i for i, l in enumerate(g_lab) if l == k] for k in gids}
    S = np.stack([sim[:, cols].max(1) for cols in id2cols.values()], 1)
    order = np.argsort(-S, axis=1, kind="stable")
    ranked = np.array(gids)[order]
    hits = ranked == np.asarray(q_lab)[:, None]
    denom = np.arange(1, len(gids) + 1)
    ap = (hits * np.cumsum(hits, 1) / denom).sum(1) / np.maximum(hits.sum(1), 1)
    return {"rank1": float(hits[:, 0].mean()), "mAP": float(ap.mean()),
            "macro_rank1": float(np.mean([hits[:, 0][np.asarray(q_lab) == k].mean()
                                          for k in set(q_lab)])),
            "n_queries": int(len(q_lab))}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    """torch insists on a CPU uint8 tensor; a checkpoint round-trip can hand back something
    else, so coerce rather than crash."""
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    t = s["torch"]
    torch.set_rng_state(t.detach().to("cpu", torch.uint8).clone()
                        if torch.is_tensor(t) else torch.ByteTensor(list(t)))
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [x.detach().to("cpu", torch.uint8).clone() if torch.is_tensor(x)
             else torch.ByteTensor(list(x)) for x in s["cuda"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-runs", type=int, default=15)
    ap.add_argument("--max-seconds", type=int, default=10 ** 9)
    ap.add_argument("--only", default="", help="comma list of run ids, e.g. fold0_s42,fold1_s43")
    ap.add_argument("--verify-resume", action="store_true")
    a = ap.parse_args()

    wsha = verify_weights()
    logline("weight sha256 verified: %s" % wsha)
    if a.verify_resume:
        for k in range(5):
            for s in SEEDS:
                rp = os.path.join(RUNS, "fold%d_s%d" % (k, s), "resume.pt")
                if os.path.isfile(rp):
                    ck = torch.load(rp, map_location="cpu", weights_only=False)
                    print("fold%d_s%d resume at epoch %d" % (k, s, ck["epoch"]))
    nest = json.load(open(F_NEST, encoding="utf-8"))
    cache = json.load(open(F_CACHE, encoding="utf-8"))
    roi = json.load(open(F_ROI, encoding="utf-8"))
    role_of = {(r["canonical"], r["rel"]): r["role"] for r in roi["per_image"]}
    src_of = {(r["canonical"], r["rel"]): r["source"] for r in cache["records"]}
    by_id = {}
    for r in cache["records"]:
        by_id.setdefault(r["canonical"], []).append(dict(r, source=r["source"]))
    for c, rs in by_id.items():
        for r in rs:
            r["role"] = role_of.get((c, r["rel"]), "query")
    dev = a.device
    tr_tf, ev_tf = build_transforms()
    recipe_sha = hashlib.sha256(json.dumps(
        {"model": MODEL_ID, "weight_sha256": wsha,
         "batchnorm_statistic_group": MICRO, "deviation": "BatchNorm running statistics over the micro-batch (%d) rather than the effective batch (%d)" % (MICRO, EFF), "epochs": EPOCHS, "micro": MICRO,
         "accum": ACCUM, "eff": EFF, "imgsz": IMGSZ, "emb": EMB, "margin": MARGIN,
         "scale": SCALE, "lr": LR, "wd": WD, "opt": "AdamW", "sched": "none",
         "prec": "FP32", "aug": "HFlip+ColorJitter(b .5-1.5)+RandomErasing(p.5,.02-.25)",
         "interp": "bicubic", "crop_pct": 1.0, "norm": [MEAN, STD],
         "selection": "inner-val identity-level mAP, ties->earlier"},
        sort_keys=True).encode()).hexdigest()
    code_sha = fsha(os.path.abspath(__file__))

    record = json.load(open(F_RECORD, encoding="utf-8")) if os.path.isfile(F_RECORD) else \
        {"schema": "h2025_p4b_training_record/v1", "stage": "P4-B",
         "recipe_sha256": recipe_sha, "code_sha256": code_sha,
         "weight_sha256": wsha, "runs": []}
    sel = json.load(open(F_SEL, encoding="utf-8")) if os.path.isfile(F_SEL) else \
        {"schema": "h2025_p4b_checkpoint_selection/v1", "runs": []}
    done = {r["run_id"] for r in record["runs"]}

    t_start = time.time()
    n_done = 0
    for k in range(5):
        e = nest["per_outer_fold"]["fold%d" % k]
        fit_ids, val_ids, out_ids = e["inner_fit"], e["inner_val"], e["outer_eval"]
        fit_recs = [r for c in fit_ids for r in by_id[c]]
        vg = [r for r in cache["records"] if r["canonical"] in set(val_ids) and r["role"] == "gallery"]
        vq = [r for r in cache["records"] if r["canonical"] in set(val_ids) and r["role"] == "query"]
        for r in vg + vq:
            r["source"] = src_of[(r["canonical"], r["rel"])]
        for r in fit_recs:
            r["source"] = src_of[(r["canonical"], r["rel"])]
        lm = {c: i for i, c in enumerate(sorted(fit_ids))}
        lv = {c: i for i, c in enumerate(sorted(val_ids))}
        for s in SEEDS:
            rid = "fold%d_s%d" % (k, s)
            if rid in done:
                continue
            if a.only and rid not in set(a.only.split(",")):
                continue
            if n_done >= a.max_runs or (time.time() - t_start) > a.max_seconds:
                logline("time/run budget reached; stopping cleanly")
                json.dump(record, open(F_RECORD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                json.dump(sel, open(F_SEL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                return
            rd = os.path.join(RUNS, rid)
            os.makedirs(rd, exist_ok=True)
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision("highest")
            net = make_net(len(lm), dev)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            hist, best, start_ep = [], None, 1
            rp = os.path.join(rd, "resume.pt")
            if os.path.isfile(rp):
                ck = torch.load(rp, map_location=dev, weights_only=False)
                net.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                start_ep = ck["epoch"] + 1
                hist = ck["hist"]
                best = ck["best"]
                set_rng_state(ck["rng"])
                logline("%s resumed at epoch %d" % (rid, start_ep))
            dl = DataLoader(CachedFaceSet(fit_recs, lm, True, tr_tf), batch_size=MICRO,
                            shuffle=True, num_workers=0, drop_last=False)
            t0 = time.time()
            for ep in range(start_ep, EPOCHS + 1):
                net.train()
                tot, nb = 0.0, 0
                opt.zero_grad(set_to_none=True)
                steps = 0
                for x, y in dl:
                    x, y = x.to(dev), y.to(dev)
                    loss = F.cross_entropy(net.logits(net(x), y, MARGIN, SCALE), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss %s ep%d" % (rid, ep))
                    (loss / ACCUM).backward()
                    tot += loss.item()
                    nb += 1
                    steps += 1
                    if steps % ACCUM == 0:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                if steps % ACCUM:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                ge = embed(net, vg, lv, ev_tf, dev)
                qe = embed(net, vq, lv, ev_tf, dev)
                m = retrieval_metrics(qe, [lv[r["canonical"]] for r in vq],
                                      ge, [lv[r["canonical"]] for r in vg])
                hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
                if best is None or m["mAP"] > best["mAP"]:
                    best = {"epoch": ep, **m}
                    torch.save({"state_dict": net.state_dict(), "class_to_idx": lm, "epoch": ep},
                               os.path.join(rd, "best.pt"))
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                            "hist": hist, "best": best, "rng": rng_state(),
                            "recipe_sha256": recipe_sha}, rp)
                logline("%s ep%02d loss=%.4f inner_val_mAP=%.4f" % (rid, ep, tot / max(nb, 1), m["mAP"]))
            json.dump(hist, open(os.path.join(rd, "history.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            bp = os.path.join(rd, "best.pt")
            entry = {"run_id": rid, "fold": k, "seed": s,
                     "n_train_ids": len(fit_ids), "n_inner_val_ids": len(val_ids),
                     "n_outer_ids": len(out_ids), "n_train_images": len(fit_recs),
                     "n_inner_val_images": len(vg) + len(vq),
                     "n_outer_images": sum(1 for r in roi["per_image"]
                                           if r["canonical"] in set(out_ids)),
                     "model": MODEL_ID, "pretrained_weight_sha256": wsha,
                     "code_sha256": code_sha, "recipe_sha256": recipe_sha,
                     "optimizer": "AdamW", "lr": LR, "weight_decay": WD,
                     "microbatch": MICRO, "accum": ACCUM, "effective_batch": EFF,
                     "precision": "FP32", "wall_sec": round(time.time() - t0, 1),
                     "best_pt_sha256": fsha(bp)}
            record["runs"] = [r for r in record["runs"] if r["run_id"] != rid] + [entry]
            record["runs"].sort(key=lambda r: (r["fold"], r["seed"]))
            record["n_runs"] = len(record["runs"])
            sel["runs"] = [r for r in sel["runs"] if r["run_id"] != rid] + [{
                "run_id": rid, "fold": k, "seed": s, "history": hist,
                "selected_epoch": best["epoch"], "selected_inner_val_mAP": best["mAP"],
                "selection_rule": "inner-val identity-level mAP maximal; ties -> earlier epoch",
                "outer_eval_used_for_selection": False}]
            sel["runs"].sort(key=lambda r: (r["fold"], r["seed"]))
            json.dump(record, open(F_RECORD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            json.dump(sel, open(F_SEL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            logline("%s DONE best_epoch=%d mAP=%.4f (%.0fs)"
                    % (rid, best["epoch"], best["mAP"], time.time() - t0))
            n_done += 1
    print("runs completed this invocation:", n_done, "| total:", record.get("n_runs"))
    print("WROTE", F_RECORD)


if __name__ == "__main__":
    main()
