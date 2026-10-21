# -*- coding: utf-8 -*-
"""Compose Figure 5 to journal spec: 2x2 qualitative retrieval panel.

Each cell:  (Query) -> (Ground truth)  +  the exact rank change, read from the frozen
candidate table (no number is retyped or recomputed).

The identity picked by the wrong ranking is absent from every frozen artefact, so the panel
shows the RANK CHANGE rather than a "wrong match image".

Outputs: TIFF (300 dpi, 190 mm) / PNG, RGB.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P = r"D:\H2025_clean_restart"
RAW = os.path.join(P, "external", "cows_frontal_face", "extracted")
AUD = os.path.join(P, "analysis", "stage_p2r_det1",
                   "CFF_PRIMARY_ROI_COMPATIBILITY_AUDIT.json")
CAND = os.path.join(P, "analysis", "paper_master", "FIG4_CANDIDATES_FILTERED.json")
OUT = r"D:\CowReID_Paper\fig5_out"
os.makedirs(OUT, exist_ok=True)

DPI = 300
W_TARGET = int(165 / 25.4 * DPI)

cand = json.load(open(CAND, encoding="utf-8"))
SCAN = os.path.join(P, "analysis", "paper_master", "FIG4_CANDIDATE_SCAN.json")
full = json.load(open(SCAN, encoding="utf-8"))
pool = full["rescue_candidates"] + full["adverse_candidates"]
bykey = {}
for r in pool:
    gid = r.get("GROUND_TRUTH_ID") or r.get("QUERY_ID")
    bykey[(gid, r["FOLD"], r["SEED"])] = r

PICKS = [("RESCUE", "cattle-243", "2", "42"),
         ("RESCUE", "cattle-016", "4", "42"),
         ("ADVERSE", "cattle-059", "0", "42"),
         ("ADVERSE", "cattle-189", "3", "42")]

aud = json.load(open(AUD, encoding="utf-8"))
per = aud["per_image"]


def find(role, canon):
    for e in per:
        if e["canonical"] == canon and e.get("role") == role:
            return e
    return None


fp = None
for c in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
    if os.path.isfile(c):
        fp = c
        break


def F(sz):
    return ImageFont.truetype(fp, sz) if fp else ImageFont.load_default()


FS = int(round(8 / 72 * DPI))
FSs = int(round(7.5 / 72 * DPI))

PAD, GAP = 26, 26
CELL_W = (W_TARGET - 2 * PAD - GAP) // 2
IMG_H = 300
# caption budget: 4 text lines (sub-label / tag / rank / identity) with breathing room
LINE_H = int(FS * 1.45)
CAP = LINE_H * 4 + 10
ROW_H = IMG_H + CAP
H_TOTAL = 2 * ROW_H + GAP + PAD * 2 + int(FS * 2.4)
sheet = Image.new("RGB", (W_TARGET, H_TOTAL), (255, 255, 255))
dr = ImageDraw.Draw(sheet)

SUB_W = (CELL_W - 40) // 2
cells = []
for kind, gt, fold, seed in PICKS:
    rec = bykey.get((gt, fold, seed), {})
    # use the EXACT query image recorded for this (identity, fold, seed)
    qrel = rec.get("QUERY_IMAGE")
    q = {"rel": qrel} if qrel else find("query", gt)
    g = find("gallery", gt)
    cells.append({"kind": kind, "gt": gt, "fold": fold, "seed": seed,
                  "q": q, "g": g, "rec": rec})

for i, c in enumerate(cells):
    r, cc = divmod(i, 2)
    x0 = PAD + cc * (CELL_W + GAP)
    y0 = PAD + int(FS * 2.4) + r * (ROW_H + GAP)

    title = {"RESCUE": "Local rescue", "ADVERSE": "Adverse fusion"}[c["kind"]]
    dr.text((x0, y0), "(%s) %s" % ("abcd"[i], title), fill=(0, 0, 0), font=F(FS))

    yy = y0 + LINE_H
    for j, (e, tag) in enumerate(((c["q"], "Query"), (c["g"], "Ground truth"))):
        sx = x0 + j * (SUB_W + 40)
        if e:
            src = os.path.join(RAW, e["rel"].replace("/", os.sep))
            if os.path.isfile(src):
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((SUB_W, IMG_H), Image.LANCZOS)
                    sheet.paste(im, (sx + (SUB_W - im.width) // 2,
                                     yy + (IMG_H - im.height) // 2))
        dr.rectangle([sx, yy, sx + SUB_W, yy + IMG_H], outline=(140, 140, 140), width=2)
    # tags sit directly under each image, on their own line
    dr.text((x0, yy + IMG_H + 4), "Query", fill=(0, 0, 0), font=F(FSs))
    dr.text((x0 + SUB_W + 40, yy + IMG_H + 4), "Ground truth", fill=(0, 0, 0), font=F(FSs))
    dr.text((x0 + SUB_W + 6, yy + IMG_H // 2), "→", fill=(0, 0, 0), font=F(FS))

    rec = c["rec"]
    gr = rec.get("GLOBAL_GT_RANK", "?")
    fr = rec.get("FUSION_GT_RANK", "?")
    yt = yy + IMG_H + LINE_H + 6
    if c["kind"] == "RESCUE":
        line = "Global rank %s  →  Fusion rank %s  (corrected)" % (gr, fr)
        col = (0, 110, 0)
    else:
        line = "Global rank %s (correct)  →  Fusion rank %s" % (gr, fr)
        col = (170, 0, 0)
    dr.text((x0, yt), line, fill=col, font=F(FS))
    dr.text((x0, yt + LINE_H), "identity %s   |   fold %s, seed %s"
            % (c["gt"], c["fold"], c["seed"]), fill=(90, 90, 90), font=F(FSs))

dr.text((PAD, PAD // 2),
        "Qualitative retrieval: Local rescue (top) and adverse fusion (bottom)",
        fill=(0, 0, 0), font=F(FS))

p_tif = os.path.join(OUT, "figure5_qualitative.tif")
p_png = os.path.join(OUT, "figure5_qualitative.png")
sheet.save(p_tif, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
sheet.save(p_png, dpi=(DPI, DPI))
print("panel:", sheet.size, "= %.1f mm wide @300dpi" % (sheet.width / DPI * 25.4))
print("WROTE", p_tif)
print("WROTE", p_png)
for c in cells:
    print("  %-8s %-12s fold%s/seed%s  query=%-28s gallery=%s" % (
        c["kind"], c["gt"], c["fold"], c["seed"],
        (c["q"]["rel"].split("/")[-1] if c["q"] else "NOT FOUND"),
        (c["g"]["rel"].split("/")[-1] if c["g"] else "NOT FOUND")))
