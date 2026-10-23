# -*- coding: utf-8 -*-
"""P5-B — train the frozen Candidate A baseline (DCFuseViT + EffiLibCatReID) on the frozen
CFF nested split: 5 outer folds x 3 seeds = 15 runs.

Scientific core is imported unchanged from the released source (mobile_vit_x_small). Native
recipe is preserved verbatim: SGD(0.003, momentum 0.9, wd 1e-4), CosineAnnealingLR(T_max=30),
CrossEntropyLoss, 100 epochs, batch 4, native RandomResizedCrop + HorizontalFlip.
NO warmup (R5: the released config carries warmup_epochs=5 as an inactive field and the
execution path never calls it).

A minimal seed wrapper is added purely for RNG control; it changes no part of the method.
Checkpoint selection uses inner-val identity-level mAP only (ties -> earlier epoch).
outer_eval is never touched for any decision, and no retrieval metric is computed.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"D:\H2025_clean_restart"
P3A = os.path.join(ROOT, "analysis", "stage_p3a")
P3B = os.path.join(ROOT, "analysis", "stage_p3b")
P5A = os.path.join(ROOT, "analysis", "stage_p5a")
P5B = os.path.join(ROOT, "analysis", "stage_p5b")
SRC = r"D:\tmp\p5a\rarout\code\DCfuseVit"
RUNS = os.path.join(P5B, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, SRC)

F_NEST = os.path.join(P3A, "CFF_P3A2_NESTED_SPLIT.json")
F_ROI = os.path.join(ROOT, "analysis", "stage_p2r_det1",
                     "CFF_PRIMARY_ROI_COMPATIBILITY_AUDIT.json")
F_CACHE = os.path.join(P3B, "CFF_P3B_IMAGE_CACHE.json")
F_FREEZE = os.path.join(P5A, "P5_SELECTED_BASELINE_FREEZE.json")
F_REC = os.path.join(P5B, "P5B_BASELINE_TRAINING_RECORD.json")
F_SEL = os.path.join(P5B, "P5B_CHECKPOINT_SELECTION_RECORD.json")
F_PROG = os.path.join(P5B, "P5B_PROGRESS.log")

SEEDS = [42, 43, 44]
EPOCHS, BATCH, LR, MOM, WD, IMG = 100, 4, 0.003, 0.9, 1e-4, 224
T_MAX, ETA_MIN = 30, 1e-6
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NO_WARMUP_NOTE = ("warmup_epochs=5 exists as an inactive configuration field in the released "
                  "code; no warmup is executed in this reproduction")


def logline(m):
    with open(F_PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))


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


def seed_everything(seed, generator=None):
    """RNG-control wrapper. Reproducibility instrumentation, not a method change."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return generator


def capture_rng(generator):
    """Everything a resumed run needs so it neither restarts the scheduler nor reshuffles
    from an untracked RNG state (P5-B addendum, item C)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator": generator.get_state() if generator is not None else None,
    }


def restore_rng(st, generator):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    t = st["torch"]
    torch.set_rng_state(t.detach().to("cpu", torch.uint8).clone() if torch.is_tensor(t)
                        else torch.ByteTensor(list(t)))
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [x.detach().to("cpu", torch.uint8).clone() if torch.is_tensor(x)
             else torch.ByteTensor(list(x)) for x in st["cuda"]])
    if generator is not None and st.get("generator") is not None:
        generator.set_state(st["generator"])


SEED_WRAPPER_SHA = hashlib.sha256(b"p5b_seed_wrapper_v1").hexdigest()
from model import mobile_vit_x_small  # noqa: E402


def build_transforms():
    from torchvision import transforms
    tr = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        transforms.Resize((IMG, IMG)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    ev = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    return tr, ev


class RoiSet(Dataset):
    def __init__(self, recs, label_map, tf):
        self.recs = list(recs)
        self.labels = [label_map[r["canonical"]] for r in self.recs]
        self.tf = tf

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        with Image.open(os.path.join(ROOT, self.recs[i]["source"])) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]


@torch.no_grad()
def extract(net, recs, lm, tf, device, bs=32):
    net.eval()
    dl = DataLoader(RoiSet(recs, lm, tf), batch_size=bs, shuffle=False, num_workers=0)
    out = []
    for x, _ in dl:
        out.append(net.extract_features(x.to(device)).float().cpu())
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
    return {"rank1": float(hits[:, 0].mean()), "mAP": float(ap.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="")
    ap.add_argument("--max-runs", type=int, default=15)
    a = ap.parse_args()

    freeze = json.load(open(F_FREEZE, encoding="utf-8"))
    nest = json.load(open(F_NEST, encoding="utf-8"))
    roi = json.load(open(F_ROI, encoding="utf-8"))
    cache = json.load(open(F_CACHE, encoding="utf-8"))
    src_of = {(r["canonical"], r["rel"]): r["source"] for r in cache["records"]}
    role_of = {(r["canonical"], r["rel"]): r["role"] for r in roi["per_image"]}
    by_id = {}
    for r in roi["per_image"]:
        by_id.setdefault(r["canonical"], []).append(r)
    dev = a.device
    tr_tf, ev_tf = build_transforms()
    src_tree_sha = freeze["repo"]["unpacked_tree_sha256"]
    recipe = {"model": "mobile_vit_x_small (released)", "source_tree_sha256": src_tree_sha,
              "optimizer": "SGD", "lr": LR, "momentum": MOM, "weight_decay": WD,
              "scheduler": "CosineAnnealingLR(T_max=%d, eta_min=%g)" % (T_MAX, ETA_MIN),
              "loss": "CrossEntropyLoss", "epochs": EPOCHS, "batch": BATCH, "img": IMG,
              "augmentation": "RandomResizedCrop(224,0.5-1.0,1.0) + Resize + HFlip",
              "warmup": "NONE", "warmup_note": NO_WARMUP_NOTE,
              "seeds": SEEDS, "selection": "inner-val identity-level mAP, ties -> earlier"}
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    code_sha = fsha(os.path.abspath(__file__))

    rec = json.load(open(F_REC, encoding="utf-8")) if os.path.isfile(F_REC) else \
        {"schema": "h2025_p5b_baseline_training_record/v1", "stage": "P5-B",
         "recipe": recipe, "recipe_sha256": recipe_sha, "code_sha256": code_sha,
         "source_tree_sha256": src_tree_sha, "seed_wrapper_sha256": SEED_WRAPPER_SHA,
         "pretrained_components_loaded": "explicitly NONE (from-scratch, documented init)",
         "runs": []}
    sel = json.load(open(F_SEL, encoding="utf-8")) if os.path.isfile(F_SEL) else \
        {"schema": "h2025_p5b_checkpoint_selection/v1", "runs": []}
    done = {r["run_id"] for r in rec["runs"]}
    n_done = 0

    for k in range(5):
        e = nest["per_outer_fold"]["fold%d" % k]
        fit_ids, val_ids = e["inner_fit"], e["inner_val"]
        fit_recs = []
        for c in fit_ids:
            for r in by_id.get(c, []):
                fit_recs.append({"canonical": c, "rel": r["rel"],
                                 "source": src_of[(c, r["rel"])]})
        vg, vq = [], []
        for c in val_ids:
            for r in by_id.get(c, []):
                rec_ = {"canonical": c, "rel": r["rel"], "role": role_of[(c, r["rel"])],
                        "source": src_of[(c, r["rel"])]}
                (vg if rec_["role"] == "gallery" else vq).append(rec_)
        lm = {c: i for i, c in enumerate(sorted(fit_ids))}
        lv = {c: i for i, c in enumerate(sorted(val_ids))}
        for s in SEEDS:
            rid = "fold%d_s%d" % (k, s)
            if rid in done or (a.only and rid not in set(a.only.split(","))):
                continue
            if n_done >= a.max_runs:
                json.dump(rec, open(F_REC, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                json.dump(sel, open(F_SEL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                return
            rd = os.path.join(RUNS, rid)
            os.makedirs(rd, exist_ok=True)
            g = torch.Generator()
            seed_everything(s, g)
            g.manual_seed(s)                      # DataLoader generator explicitly seeded
            net = mobile_vit_x_small(num_classes=len(lm)).to(dev)
            opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=ETA_MIN)
            lossf = nn.CrossEntropyLoss()
            hist, best, start_ep = [], None, 1
            rp = os.path.join(rd, "resume.pt")
            if os.path.isfile(rp):
                ck = torch.load(rp, map_location=dev, weights_only=False)
                if ck.get("recipe_sha256") != recipe_sha:
                    raise SystemExit("STOP: resume checkpoint belongs to a different recipe")
                if ck.get("seed") != s:
                    raise SystemExit("STOP: resume checkpoint seed mismatch")
                net.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sch.load_state_dict(ck["sch"])       # scheduler is NOT restarted
                start_ep = ck["epoch"] + 1
                hist, best = ck["hist"], ck["best"]
                restore_rng(ck["rng"], g)
                logline("%s RESUMED at epoch %d (sched last_epoch=%d)"
                        % (rid, start_ep, sch.last_epoch))
            t0 = time.time()
            logline("%s START fit=%d imgs / val=%d ids (g%d q%d)"
                    % (rid, len(fit_recs), len(val_ids), len(vg), len(vq)))
            for ep in range(start_ep, EPOCHS + 1):
                net.train()
                dl = DataLoader(RoiSet(fit_recs, lm, tr_tf), batch_size=BATCH, shuffle=True,
                                num_workers=0, generator=g, drop_last=False)
                tot, nb = 0.0, 0
                for x, y in dl:
                    x, y = x.to(dev), y.to(dev)
                    out = net(x)
                    loss = lossf(out, y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss %s ep%d" % (rid, ep))
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    tot += loss.item()
                    nb += 1
                sch.step()
                ge = extract(net, vg, lv, ev_tf, dev)
                qe = extract(net, vq, lv, ev_tf, dev)
                m = retrieval_metrics(qe, [lv[r["canonical"]] for r in vq],
                                      ge, [lv[r["canonical"]] for r in vg])
                hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
                if best is None or m["mAP"] > best["mAP"]:
                    best = {"epoch": ep, **m}
                    torch.save({"state_dict": net.state_dict(), "class_to_idx": lm, "epoch": ep},
                               os.path.join(rd, "best.pt"))
                if ep % 5 == 0 or ep == EPOCHS:
                    logline("%s ep%03d loss=%.4f inner_val_mAP=%.4f (%.0fs)"
                            % (rid, ep, tot / max(nb, 1), m["mAP"], time.time() - t0))
                # resume checkpoint: model + optimizer + scheduler + completed epoch + seed
                # + all four RNG states + the DataLoader generator state (addendum item C).
                # Written every 5 epochs: the state is large (~0.4 GB), so an interruption
                # costs at most 5 epochs, and the semantics are unaffected by the cadence.
                if ep % 5 == 0 or ep == EPOCHS:
                    torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                                "sch": sch.state_dict(), "epoch": ep, "hist": hist,
                                "best": best, "seed": s, "rng": capture_rng(g),
                                "recipe_sha256": recipe_sha},
                               rp)
            json.dump(hist, open(os.path.join(rd, "history.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            bp = os.path.join(rd, "best.pt")
            entry = {"run_id": rid, "fold": k, "seed": s, "n_fit_ids": len(fit_ids),
                     "n_val_ids": len(val_ids), "n_fit_images": len(fit_recs),
                     "n_val_images": len(vg) + len(vq), "recipe_sha256": recipe_sha,
                     "code_sha256": code_sha, "source_tree_sha256": src_tree_sha,
                     "seed_wrapper_sha256": SEED_WRAPPER_SHA,
                     "pretrained_components_loaded": "NONE",
                     "best_pt_sha256": fsha(bp), "wall_sec": round(time.time() - t0, 1)}
            rec["runs"] = [r for r in rec["runs"] if r["run_id"] != rid] + [entry]
            rec["runs"].sort(key=lambda r: (r["fold"], r["seed"]))
            rec["n_runs"] = len(rec["runs"])
            sel["runs"] = [r for r in sel["runs"] if r["run_id"] != rid] + [{
                "run_id": rid, "fold": k, "seed": s, "history": hist,
                "selected_epoch": best["epoch"], "selected_inner_val_mAP": best["mAP"],
                "outer_eval_used_for_selection": False}]
            sel["runs"].sort(key=lambda r: (r["fold"], r["seed"]))
            json.dump(rec, open(F_REC, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            json.dump(sel, open(F_SEL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            logline("%s DONE best_ep=%d mAP=%.4f (%.0fs)"
                    % (rid, best["epoch"], best["mAP"], time.time() - t0))
            n_done += 1
    print("completed this invocation:", n_done, "| total:", rec.get("n_runs"))
    print("WROTE", F_REC)


if __name__ == "__main__":
    main()
