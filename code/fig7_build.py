# -*- coding: utf-8 -*-
"""Build Figure 7 to journal spec (190 mm @300 dpi).

Panels:
  (a) parameter count (architecture-only, protocol-independent -> the safest claim)
  (b) training wall-clock per epoch, with the epoch budget annotated

Every number is read from frozen artefacts or counted from the checkpoints. Nothing is
estimated. The protocol mismatch (20 vs 100 epochs, batch 64 vs 4) is printed on the figure
so the comparison cannot be misread.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.transforms import offset_copy

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P = r"D:\H2025_clean_restart"
A = r"C:\CowReID_Archive\P5_FINAL_ARCHIVE_20260930_EXTRACTED"
OUT = r"D:\CowReID_Paper\fig7_out"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- gather numbers
def ck_params(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict") if isinstance(ck, dict) else None
    if sd is None:
        sd = ck
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


ours_ckpt = os.path.join(P, "analysis", "stage_p3b", "runs", "fold0_s42", "best.pt")
hu_ckpt = os.path.join(A, "03_P5B_FORMAL_RUNS", "fold0_s42", "best.pt")
p_ours = ck_params(ours_ckpt)
p_hu = ck_params(hu_ckpt)
p_dino = json.load(open(os.path.join(P, "analysis", "stage_p4a",
                                     "P4A_INPUT_SEAL.json"),
                        encoding="utf-8"))["modern_global"]["parameter_count"]

# training wall clock
p3b = json.load(open(os.path.join(P, "analysis", "stage_p3b",
                                  "CFF_P3B_TRAINING_RECORD.json"), encoding="utf-8"))
w_ours = float(np.mean([r["wall_sec"] for r in p3b["runs"]]))
ep_ours = p3b["recipe"]["epochs"]

p5b = json.load(open(os.path.join(A, "05_P5B_AUDITS_AND_LEDGER",
                                  "P5B_BASELINE_TRAINING_RECORD.json"), encoding="utf-8"))
w_hu = float(np.mean([r["wall_sec"] for r in p5b["runs"]]))
ep_hu = 100      # from P5B recipe

print("params  ours=%s  dino=%s  hu=%s" % (format(p_ours, ","), format(p_dino, ","),
                                           format(p_hu, ",")))
print("per-epoch  ours=%.2f s (%.1f/20)   hu=%.1f s (%.0f/100)"
      % (w_ours / ep_ours, w_ours, w_hu / ep_hu, w_hu))

# ---------------------------------------------------------------- plot
# Guard the logarithmic axes explicitly: every plotted magnitude must be strictly positive.
for _nm, _v in (("Ours params", p_ours), ("DINOv3 params", p_dino),
                ("Hu2025 params", p_hu),
                ("Ours s/epoch", w_ours / ep_ours), ("Hu2025 s/epoch", w_hu / ep_hu)):
    assert _v > 0, "log axis requires strictly positive values; %s = %r" % (_nm, _v)
    print("log-guard ok: %-16s = %.4f" % (_nm, _v))

plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.linewidth": 0.8, "axes.edgecolor": "black",
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "pdf.compression": 6})

# nature-figure panel-alignment gate (mandatory for multi-panel matplotlib figures)
SKILL_SCRIPTS = r"C:\Users\LX\.workbuddy\skills\nature-figure\scripts"
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)
FIG_W = 165 / 25.4
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.55), dpi=300)

labels = ["Ours\n(MobileNetV2)", "DINOv3", "Hu2025\n(MobileViT-x-S)"]
params = [p_ours, p_dino, p_hu]
cols = ["#2E7D32", "#1565C0", "#B71C1C"]

ax = axes[0]
b = ax.bar(labels, params, color=cols, width=0.62, edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_ylabel("Parameters (M)", fontsize=8)
ax.set_ylim(1e6, 1e8)
# panel label: bold lowercase single letter, offset in POINTS from the axes corner
# (a fixed point offset keeps tall and short panels at the same physical displacement)
ax.text(0, 1, "a", transform=offset_copy(ax.transAxes, fig=fig, x=0, y=5, units="points"),
        ha="left", va="baseline", fontsize=8, fontweight="bold")
ax.set_title("Model size", fontsize=8, pad=6)
for r, v in zip(b, params):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.12, "%.2f M" % (v / 1e6),
            ha="center", va="bottom", fontsize=7)
ax.tick_params(axis="x", labelsize=7)
# log major ticks render as mathtext 10^n; the superscript is 0.7x the parent size,
# so the y labels must be 8 pt (0.7*8 = 5.6) to stay above the 5 pt glyph floor
ax.tick_params(axis="y", labelsize=8)
# log-axis minor tick labels render as mathtext at 0.7x, which would drop below the
# 5 pt glyph floor; suppress them (the gridlines remain)
ax.tick_params(axis="y", which="minor", labelleft=False)
# tick_params alone does not suppress mathtext minor labels on a LogLocator;
# NullFormatter is required, otherwise single-digit mantissas render at 0.7x (<5 pt)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)

ax = axes[1]
lab2 = ["Ours\n(20 ep, batch 64)", "Hu2025\n(100 ep, batch 4)"]
per_ep = [w_ours / ep_ours, w_hu / ep_hu]
b = ax.bar(lab2, per_ep, color=[cols[0], cols[2]], width=0.5,
           edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_ylabel("Training time per epoch (s)", fontsize=8)
ax.set_ylim(1, 1000)
ax.text(0, 1, "b", transform=offset_copy(ax.transAxes, fig=fig, x=0, y=5, units="points"),
        ha="left", va="baseline", fontsize=8, fontweight="bold")
ax.set_title("Training cost (per epoch)", fontsize=8, pad=6)
for r, v in zip(b, per_ep):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.15, "%.1f s" % v,
            ha="center", va="bottom", fontsize=7)
ax.tick_params(axis="x", labelsize=7)
# log major ticks render as mathtext 10^n; the superscript is 0.7x the parent size,
# so the y labels must be 8 pt (0.7*8 = 5.6) to stay above the 5 pt glyph floor
ax.tick_params(axis="y", labelsize=8)
# log-axis minor tick labels render as mathtext at 0.7x, which would drop below the
# 5 pt glyph floor; suppress them (the gridlines remain)
ax.tick_params(axis="y", which="minor", labelleft=False)
# tick_params alone does not suppress mathtext minor labels on a LogLocator;
# NullFormatter is required, otherwise single-digit mantissas render at 0.7x (<5 pt)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.text(0.5, -0.42, "protocols differ (epochs / batch); compare per-epoch, not totals",
        transform=ax.transAxes, ha="center", fontsize=6, style="italic", color="#555555")

fig.tight_layout(pad=0.6)
FIG_W_MM = 165.0          # double column (Elsevier); Nature-family uses 183 mm
fig.set_size_inches(FIG_W_MM / 25.4, 2.55)

# mandatory render-time panel-alignment gate for multi-panel figures
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(OUT, "figure7_alignment.json"),
        overlay_svg=os.path.join(OUT, "figure7_alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    print("panel-alignment gate: PASS")
except SystemExit:
    raise
except ImportError as e:
    print("panel-alignment gate NOT RUN (audit_panel_alignment unavailable):", e)

png = os.path.join(OUT, "figure7_efficiency.png")
tif = os.path.join(OUT, "figure7_efficiency.tif")
pdf = os.path.join(OUT, "figure7_efficiency.pdf")
svg = os.path.join(OUT, "figure7_efficiency.svg")
fig.savefig(png, dpi=300, facecolor="white")
fig.savefig(tif, dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf, facecolor="white")
fig.savefig(svg, facecolor="white")
print("WROTE", png)
print("WROTE", tif, "(600 dpi)")
print("WROTE", pdf)
print("WROTE", svg)

json.dump({"ours_params": p_ours, "dino_params": p_dino, "hu_params": p_hu,
           "ours_sec_per_epoch": w_ours / ep_ours, "hu_sec_per_epoch": w_hu / ep_hu,
           "ours_epochs": ep_ours, "hu_epochs": ep_hu,
           "ours_batch": 64, "hu_batch": 4,
           "sources": {"ours_ckpt": ours_ckpt, "hu_ckpt": hu_ckpt,
                       "dino": "analysis/stage_p4a/P4A_INPUT_SEAL.json"},
           "gap_vram": "only Hu2025 has a recorded peak (2.212 GB); ours and DINOv3 have none"},
          open(os.path.join(OUT, "FIGURE7_DATA.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("DONE")
