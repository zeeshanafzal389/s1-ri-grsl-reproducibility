# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Build the structured-gap robustness figure (random vs contiguous-gap thinning).

Recreates figures/figure4_structured_gap_robustness.{png,pdf} from
data/downsampling/structured_gap_dose_response.csv for the primary
VV-ascending case. Publication styling: Times New Roman, no on-figure
title (the descriptive caption belongs in the manuscript).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Publication typography: Times New Roman (Liberation Serif is a
# metric-compatible fallback when TNR is unavailable, e.g. on Linux).
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DATA = PROJECT / "data" / "downsampling" / "structured_gap_dose_response.csv"
FIG_DIR = PROJECT / "figures"

# Reference band reproducing the original annotation: the spread of real
# consecutive-window endpoint changes, with median |delta| ~ 0.20 dB.
REAL_SIGNAL_MEDIAN = 0.20
REAL_SIGNAL_BAND = (0.10, 0.37)

POL = "vv_db_mean"
ORBIT = "ASCENDING"


def main() -> int:
    df = pd.read_csv(DATA)
    sub = df[(df["pol"] == POL) & (df["orbit"] == ORBIT)]
    rand = sub[sub["mode"] == "random"].sort_values("k")
    cont = sub[sub["mode"] == "contiguous"].sort_values("k")

    fig, ax = plt.subplots(figsize=(3.3, 2.65))

    ax.axhspan(*REAL_SIGNAL_BAND, color="#9aa0a6", alpha=0.16, zorder=0)
    ax.axhline(REAL_SIGNAL_MEDIAN, color="#5f6368", linestyle=":", linewidth=1.0, zorder=1)

    ax.plot(
        rand["k"],
        rand["mean_abs_bias"],
        marker="o",
        markersize=5,
        linewidth=1.7,
        color="#1f6fb2",
        label="Random thinning (best case)",
        zorder=3,
    )
    ax.plot(
        cont["k"],
        cont["mean_abs_bias"],
        marker="o",
        markersize=5,
        linewidth=1.7,
        linestyle="--",
        color="#d62728",
        label="Contiguous-gap thinning (structured)",
        zorder=3,
    )

    # Annotation placed in the open lower-left region, clear of both curves.
    ax.annotate(
        "Typical real signal\n(median |Δ| = 0.20 dB)",
        xy=(1.0, REAL_SIGNAL_MEDIAN),
        xytext=(1.15, 0.135),
        fontsize=7.5,
        color="#3c4043",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#9aa0a6", lw=0.7),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels([1, 2, 4, 8, 16])
    ax.set_xlim(0.9, 17.5)
    ax.set_xlabel("Retained scenes per window, k")
    ax.set_ylabel("Mean absolute endpoint-change bias (dB)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.tight_layout(pad=0.4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "figure4_structured_gap_robustness.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
