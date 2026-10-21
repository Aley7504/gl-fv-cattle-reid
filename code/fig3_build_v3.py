# -*- coding: utf-8 -*-
"""Build Figure 3 to journal spec: Delta_R1 with 95% CI across the four groups.

All four numbers are read from the frozen FIG3_DELTA_RESULT report; nothing is retyped or
recomputed. Significance (whether the CI excludes zero) comes from the stored CI, not from a
new test.

Output: 190 mm @ 300 dpi (2244 px) TIFF + PNG.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P = r"D:\H2025_clean_restart"
SRC = os.path.join(P, "analysis", "paper_master", "FIG3_DELTA_RESULT_REPORT.json")
OUT = r"D:\CowReID_Paper\fig3_out"
os.makedirs(OUT, exist_ok=True)

rep = json.load(open(SRC, encoding="utf-8"))
rows = rep["rows"]
print("rows loaded:", len(rows))

# order: H2025 pair, then CFF pair
order = [("H2025", "MobileNetV2"), ("H2025", "ConvNeXt"),
         ("CFF", "MobileNetV2"), ("CFF", "DINOv3")]


def find(ds, key):
    for r in rows:
        if r["Dataset"] == ds and key.lower() in r["Global_representation"].lower():
            return r
    raise KeyError("%s / %s" % (ds, key))


sel = [find(a, b) for a, b in order]
for r in sel:
    print("  %-6s %-46s %+8.4f  [%+8.4f, %+8.4f]  excl0=%s"
          % (r["Dataset"], r["Global_representation"][:46], r["Delta_R1_pp"],
             r["CI_low_pp"], r["CI_high_pp"], r["CI_excludes_zero"]))

labels = ["H2025\nMobileNetV2", "H2025\nConvNeXt-Tiny",
          "CFF\nMobileNetV2", "CFF\nDINOv3"]
delta = np.array([r["Delta_R1_pp"] for r in sel])
lo = np.array([r["CI_low_pp"] for r in sel])
hi = np.array([r["CI_high_pp"] for r in sel])
err = np.vstack([delta - lo, hi - delta])
excl = [r["CI_excludes_zero"] == "YES" for r in sel]

# colour: CI excluding zero = solid blue; CI spanning zero = orange (not significant)
COLS = ["#1F4E9C" if e else "#E07B39" for e in excl]

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.8,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "pdf.compression": 6})
FIG_W = 165 / 25.4
fig, ax = plt.subplots(figsize=(FIG_W, 2.5), dpi=300)

x = np.arange(len(sel))
b = ax.bar(x, delta, width=0.58, color=COLS, edgecolor="black", linewidth=0.7,
           zorder=3)
ax.errorbar(x, delta, yerr=err, fmt="none", ecolor="black", elinewidth=1.0,
            capsize=3.5, capthick=1.0, zorder=4)

ax.axhline(0, color="black", lw=0.8, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$\Delta$Rank-1 (pp)", fontsize=8)
ax.set_ylim(-8, 32)
ax.tick_params(axis="y", labelsize=7)

# value + CI printed above/below each bar
for i, r in enumerate(sel):
    d = r["Delta_R1_pp"]
    top = hi[i]
    ax.text(i, top + 1.0, "%+.2f" % d, ha="center", va="bottom", fontsize=7.5,
            fontweight="bold")
    ax.text(i, top + 3.6, "[%+.2f, %+.2f]" % (lo[i], hi[i]), ha="center",
            va="bottom", fontsize=6.2, color="#444444")

# dataset grouping separators + labels
ax.axvline(1.5, color="#999999", lw=0.7, ls="--", zorder=1)
ax.text(0.5, -7.2, "H2025 (held-out, in-domain)", ha="center", fontsize=7,
        color="#333333")
ax.text(2.5, -7.2, "CFF (external, identity-disjoint)", ha="center", fontsize=7,
        color="#333333")

# significance is encoded by colour + an explicit legend (a floating '*' would sit
# behind the error bar and be unreadable)
from matplotlib.patches import Patch
handles = [Patch(facecolor="#1F4E9C", edgecolor="black", linewidth=0.7,
                 label="95% CI excludes 0"),
           Patch(facecolor="#E07B39", edgecolor="black", linewidth=0.7,
                 label="95% CI includes 0 (n.s.)")]
ax.legend(handles=handles, fontsize=6.5, loc="upper left", frameon=False,
          handlelength=1.4, handleheight=0.9, borderaxespad=0.3)

ax.grid(axis="y", ls=":", lw=0.5, alpha=0.55)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.5)
# Final physical size is declared here so the width is explicit in the source.
FIG_W_MM = 165.0          # double column (Elsevier); Nature-family uses 183 mm
fig.set_size_inches(FIG_W_MM / 25.4, 2.5)
png = os.path.join(OUT, "figure3_delta_r1.png")
tif = os.path.join(OUT, "figure3_delta_r1.tif")
pdf = os.path.join(OUT, "figure3_delta_r1.pdf")
svg = os.path.join(OUT, "figure3_delta_r1.svg")
fig.savefig(png, dpi=300, facecolor="white")
fig.savefig(tif, dpi=600, facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(pdf, facecolor="white")
fig.savefig(svg, facecolor="white")
print()
print("WROTE", png)
print("WROTE", tif, "(600 dpi)")
print("WROTE", pdf)
print("WROTE", svg)

json.dump({"figure": "3", "quantity": "Delta_R1 = R1(Global+Local) - R1(Global), pp",
           "source": "analysis/paper_master/FIG3_DELTA_RESULT_REPORT.json",
           "rows": sel,
           "note": "CI excludes zero => '*' ; CI spans zero => 'n.s.' ; "
                   "colour encodes the same distinction, read from the stored CI"},
          open(os.path.join(OUT, "FIGURE3_DATA.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("DONE")
