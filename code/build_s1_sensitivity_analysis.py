# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
S1-RI Sensitivity Analysis

Tests stability of the S1 Reliability Index under:
  (A) 7 alternative weighting schemes
  (B) +/-1-scene counting tolerance on mean acquisition counts

Outputs
-------
  data/ri_outputs/s1_ri_sensitivity_table.csv     -- per-city scores under all schemes
  data/ri_outputs/s1_ri_sensitivity_summary.csv   -- scheme-level stats + correlations
  data/ri_outputs/s1_ri_scene_tolerance_table.csv -- per-city +/-1-scene delta
  data/ri_outputs/s1_ri_sensitivity_qc.json       -- run summary
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# -- paths --------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[1]
ROOT    = PROJECT.parent
RI_CSV  = PROJECT / "data" / "ri_outputs" / "s1_city_reliability_index.csv"
OUT_DIR = PROJECT / "data" / "ri_outputs"

# -- score component columns (in order) ---------------------------------------
COMPONENTS = [
    "score_component_mean_acquisition",
    "score_component_observation_continuity",
    "score_component_gap",
    "score_component_dual_pol",
    "score_component_orbit_diversity",
    "score_component_1b_shock_resilience",
]

COMP_LABELS = [
    "mean_acquisition",
    "obs_continuity",
    "gap",
    "dual_pol",
    "orbit_diversity",
    "1b_shock_resilience",
]

# tier labels (ASCII-only)
TIER_LOW    = "low (<0.50)"
TIER_MEDIUM = "medium (0.50-0.75)"
TIER_HIGH   = "high (>0.75)"

# -- weighting schemes --------------------------------------------------------
# Each entry: (scheme_id, label, weights_list)
# Weights are renormalised to sum=1 before use.
SCHEMES = [
    # W0 -- Baseline (published)
    (
        "W0_baseline",
        "Baseline (published): acquisition+continuity-dominant",
        [0.25, 0.25, 0.20, 0.15, 0.10, 0.05],
    ),
    # W1 -- Equal weight
    (
        "W1_equal",
        "Equal weight (1/6 each)",
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    ),
    # W2 -- Gap-heavy
    (
        "W2_gap_heavy",
        "Gap-heavy (gap=0.35, acquisition+continuity=0.40, others=0.25)",
        [0.20, 0.20, 0.35, 0.10, 0.10, 0.05],
    ),
    # W3 -- Continuity-heavy
    (
        "W3_continuity_heavy",
        "Continuity-heavy (obs_continuity=0.40, acquisition=0.30, others=0.30)",
        [0.30, 0.40, 0.15, 0.05, 0.05, 0.05],
    ),
    # W4 -- Coverage-only (drop polarisation and orbit diversity)
    (
        "W4_coverage_only",
        "Coverage-only (acquisition+continuity+gap only, others=0)",
        [0.35, 0.35, 0.30, 0.00, 0.00, 0.00],
    ),
    # W5 -- Dual-pol-heavy
    (
        "W5_dualpol_heavy",
        "Dual-pol-heavy (dual_pol=0.35, acquisition+continuity=0.40, others=0.25)",
        [0.20, 0.20, 0.10, 0.35, 0.10, 0.05],
    ),
    # W6 -- No 1B-shock resilience component
    (
        "W6_no_1b_shock",
        "No-1B-shock (weight=0, redistributed to acquisition+continuity)",
        [0.275, 0.275, 0.20, 0.15, 0.10, 0.00],
    ),
]


# -- helpers ------------------------------------------------------------------

def normalise_weights(w):
    s = sum(w)
    return [x / s for x in w] if s > 0 else w


def compute_scheme_scores(df, weights):
    w = normalise_weights(weights)
    return sum(df[c] * wt for c, wt in zip(COMPONENTS, w)).clip(0, 1)


def tier(s):
    return pd.cut(
        s,
        bins=[0, 0.5, 0.75, 1.001],
        labels=[TIER_LOW, TIER_MEDIUM, TIER_HIGH],
        right=False,
    )


def pearsonr(a, b):
    return float(np.corrcoef(a.values, b.values)[0, 1])


def spearmanr(a, b):
    return pearsonr(a.rank(), b.rank())


def kendalltau(a, b):
    """Vectorised Kendall tau (no ties correction)."""
    av = a.values.astype(float)
    bv = b.values.astype(float)
    n = len(av)
    da = np.sign(av[:, None] - av[None, :])
    db = np.sign(bv[:, None] - bv[None, :])
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    prod = (da * db)[mask]
    conc = int((prod > 0).sum())
    disc = int((prod < 0).sum())
    denom = n * (n - 1) / 2
    return float((conc - disc) / denom) if denom > 0 else 0.0


def scheme_summary(df, scores, baseline, scheme_id, label, weights):
    w = normalise_weights(weights)
    r_pearson  = pearsonr(baseline, scores)
    r_spearman = spearmanr(baseline, scores)
    r_kendall  = kendalltau(baseline, scores)
    tier_base  = tier(baseline)
    tier_alt   = tier(scores)
    tier_change = int((tier_base != tier_alt).sum())
    mae = float((scores - baseline).abs().mean())
    row = {
        "scheme_id": scheme_id,
        "label": label,
    }
    for lbl, wt in zip(COMP_LABELS, w):
        row["w_" + lbl] = round(wt, 6)
    row.update({
        "mean_score":   round(float(scores.mean()),   6),
        "median_score": round(float(scores.median()), 6),
        "std_score":    round(float(scores.std()),    6),
        "min_score":    round(float(scores.min()),    6),
        "max_score":    round(float(scores.max()),    6),
        "pearson_r_vs_baseline":              round(r_pearson,  6),
        "spearman_r_vs_baseline":             round(r_spearman, 6),
        "kendall_tau_vs_baseline":            round(r_kendall,  6),
        "mae_vs_baseline":                    round(mae,        6),
        "n_cities_tier_changed_vs_baseline":  tier_change,
        "pct_cities_tier_changed_vs_baseline": round(100 * tier_change / len(df), 3),
        "n_low":    int((tier_alt == TIER_LOW).sum()),
        "n_medium": int((tier_alt == TIER_MEDIUM).sum()),
        "n_high":   int((tier_alt == TIER_HIGH).sum()),
    })
    return row


# -- main analysis functions --------------------------------------------------

def run_weighting_sensitivity(df):
    baseline   = df["s1_ri_score"]
    score_cols = {}
    summaries  = []

    for scheme_id, label, weights in SCHEMES:
        scores = compute_scheme_scores(df, weights)
        score_cols[scheme_id] = scores
        summaries.append(scheme_summary(df, scores, baseline, scheme_id, label, weights))

    score_df = df[["city_id", "city_name", "country", "continent", "s1_ri_score"]].copy()
    score_df = score_df.rename(columns={"s1_ri_score": "W0_baseline"})
    for col, series in list(score_cols.items())[1:]:
        score_df[col] = series.values

    summary_df = pd.DataFrame(summaries)
    return score_df, summary_df


def run_scene_tolerance(df):
    """
    Recompute score_component_mean_acquisition and s1_ri_score
    under +1 and -1 scene offsets on mean_monthly_acquisitions.
    All other score components are unchanged.
    """
    base_weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

    def recompute_ri(df_mod):
        w = normalise_weights(base_weights)
        return sum(df_mod[c] * wt for c, wt in zip(COMPONENTS, w)).clip(0, 1)

    results = df[[
        "city_id", "city_name", "country", "continent",
        "mean_monthly_acquisitions_2017_2026",
        "score_component_mean_acquisition",
        "s1_ri_score",
    ]].copy()

    for offset, suffix in [(+1, "plus1"), (-1, "minus1")]:
        mod_acq  = (df["mean_monthly_acquisitions_2017_2026"] + offset).clip(lower=0)
        mod_comp = (mod_acq / 12.0).clip(0, 1)
        df_mod   = df.copy()
        df_mod["score_component_mean_acquisition"] = mod_comp
        mod_ri   = recompute_ri(df_mod)

        results["mean_monthly_acquisitions_" + suffix]      = mod_acq.values
        results["score_component_mean_acquisition_" + suffix] = mod_comp.values
        results["s1_ri_score_" + suffix]                    = mod_ri.values
        results["s1_ri_delta_" + suffix]                    = (mod_ri - df["s1_ri_score"]).values

    return results


# -- entry point --------------------------------------------------------------

def main():
    df = pd.read_csv(RI_CSV)
    print("Loaded {:,} cities from {}".format(len(df), RI_CSV.name))

    # A. Weighting sensitivity
    score_table, summary_table = run_weighting_sensitivity(df)
    score_path   = OUT_DIR / "s1_ri_sensitivity_table.csv"
    summary_path = OUT_DIR / "s1_ri_sensitivity_summary.csv"
    score_table.to_csv(score_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    print("\nWeighting sensitivity written to {} and {}".format(score_path.name, summary_path.name))
    print(summary_table[[
        "scheme_id", "pearson_r_vs_baseline",
        "spearman_r_vs_baseline", "mae_vs_baseline",
        "pct_cities_tier_changed_vs_baseline",
    ]].to_string(index=False))

    # B. +/-1-scene tolerance
    tol_table = run_scene_tolerance(df)
    tol_path  = OUT_DIR / "s1_ri_scene_tolerance_table.csv"
    tol_table.to_csv(tol_path, index=False)
    print("\n+/-1-scene tolerance written to {}".format(tol_path.name))
    plus1_mae  = float(tol_table["s1_ri_delta_plus1"].abs().mean())
    minus1_mae = float(tol_table["s1_ri_delta_minus1"].abs().mean())
    plus1_max  = float(tol_table["s1_ri_delta_plus1"].abs().max())
    minus1_max = float(tol_table["s1_ri_delta_minus1"].abs().max())
    print("  +1 scene: MAE={:.6f}, max delta={:.6f}".format(plus1_mae, plus1_max))
    print("  -1 scene: MAE={:.6f}, max delta={:.6f}".format(minus1_mae, minus1_max))

    # QC summary JSON
    qc_schemes = []
    for _, row in summary_table.iterrows():
        qc_schemes.append({
            "scheme_id":                       row["scheme_id"],
            "pearson_r_vs_baseline":           row["pearson_r_vs_baseline"],
            "spearman_r_vs_baseline":          row["spearman_r_vs_baseline"],
            "kendall_tau_vs_baseline":         row["kendall_tau_vs_baseline"],
            "mae_vs_baseline":                 row["mae_vs_baseline"],
            "pct_cities_tier_changed":         row["pct_cities_tier_changed_vs_baseline"],
        })

    qc = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_cities":          len(df),
        "source_ri_csv":     str(RI_CSV),
        "weighting_schemes": qc_schemes,
        "scene_tolerance": {
            "plus1_scene_mae":       round(plus1_mae,  6),
            "plus1_scene_max_delta": round(plus1_max,  6),
            "minus1_scene_mae":      round(minus1_mae, 6),
            "minus1_scene_max_delta": round(minus1_max, 6),
        },
        "outputs": {
            "sensitivity_table":    str(score_path),
            "sensitivity_summary":  str(summary_path),
            "scene_tolerance_table": str(tol_path),
        },
    }
    qc_path = OUT_DIR / "s1_ri_sensitivity_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print("\nQC summary written to {}".format(qc_path.name))
    print("\nDone.")


if __name__ == "__main__":
    main()
