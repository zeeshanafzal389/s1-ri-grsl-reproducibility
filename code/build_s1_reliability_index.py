#!/usr/bin/env python3
"""Compute city-level Sentinel-1 reliability metrics and S1-RI scores."""

from __future__ import annotations

import argparse
import calendar
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_NAME = "Sentinel-1 Observation Reliability & Continuity Index"
SOURCE_VERSION = "copernicus_odata_s1_iw_grdh_201701_202605_v1"
START_YM = "2017-01"
END_YM = "2026-05"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def archive_if_exists(path: Path, archive_root: Path) -> None:
    if not path.exists():
        return
    dest_dir = archive_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest_dir / path.name))


def month_days(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1]


def longest_zero_gap_days(group: pd.DataFrame) -> int:
    max_days = 0
    cur_days = 0
    for _, row in group.sort_values(["year", "month"]).iterrows():
        if int(row["n_total"]) <= 0:
            cur_days += month_days(int(row["year"]), int(row["month"]))
            max_days = max(max_days, cur_days)
        else:
            cur_days = 0
    return int(max_days)


def ratio_or_nan(num: pd.Series, denom: pd.Series) -> pd.Series:
    out = num / denom.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def era_mean(df: pd.DataFrame, start: str, end: str, value_col: str = "n_total") -> pd.Series:
    subset = df[(df["year_month"] >= start) & (df["year_month"] <= end)]
    return subset.groupby("city_id")[value_col].mean()


def era_fraction(df: pd.DataFrame, start: str, end: str, bool_col: str) -> pd.Series:
    subset = df[(df["year_month"] >= start) & (df["year_month"] <= end)]
    return subset.groupby("city_id")[bool_col].mean()


def build_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    project = root if (root / "data").exists() else root / PROJECT_NAME
    counts_path = project / "data" / "city_counts" / "s1_city_monthly_counts_2017_2026.csv"
    if (project / "data" / "source_city_data").exists():
        city_table_path = project / "data" / "source_city_data" / "statistics_v2" / "supplementary_table1_all_cities_v2.csv"
    else:
        city_table_path = root / "analysis" / "statistics_v2" / "supplementary_table1_all_cities_v2.csv"

    counts = pd.read_csv(counts_path)
    city_table = pd.read_csv(city_table_path, low_memory=False)
    city_table = city_table[city_table["data_quality_flag"] == "v2_endpoint_shards"].copy()

    counts["year_month"] = counts["year"].astype(str) + "-" + counts["month"].astype(str).str.zfill(2)
    counts["has_observation"] = counts["has_observation"].astype(bool)
    counts["has_dual_pol_month"] = counts["n_dual_pol"] > 0
    counts["has_both_orbits_month"] = (counts["n_asc"] > 0) & (counts["n_desc"] > 0)
    counts["has_s1c_or_s1d_month"] = (counts["n_s1c"] + counts["n_s1d"]) > 0

    g = counts.groupby("city_id", sort=False)
    metrics = pd.DataFrame(
        {
            "city_id": g.size().index,
            "n_months": g.size().values,
            "mean_monthly_acquisitions_2017_2026": g["n_total"].mean().values,
            "median_monthly_acquisitions_2017_2026": g["n_total"].median().values,
            "min_monthly_acquisitions_2017_2026": g["n_total"].min().values,
            "max_monthly_acquisitions_2017_2026": g["n_total"].max().values,
            "fraction_months_with_observation": g["has_observation"].mean().values,
            "fraction_months_with_dual_pol_coverage": g["has_dual_pol_month"].mean().values,
            "fraction_months_with_both_asc_desc": g["has_both_orbits_month"].mean().values,
            "mean_monthly_s1a": g["n_s1a"].mean().values,
            "mean_monthly_s1b": g["n_s1b"].mean().values,
            "mean_monthly_s1c": g["n_s1c"].mean().values,
            "mean_monthly_s1d": g["n_s1d"].mean().values,
        }
    )

    gap = g.apply(longest_zero_gap_days, include_groups=False).rename("longest_observation_gap_days")
    metrics = metrics.merge(gap.reset_index(), on="city_id", how="left")
    metrics["longest_gap_metric_basis"] = "monthly_zero_observation_run_days"

    era_series = {
        "pre_1b_baseline_mean_monthly_count_2019_2021": era_mean(counts, "2019-01", "2021-12"),
        "single_satellite_era_mean_monthly_count_2022_2024": era_mean(counts, "2022-01", "2024-12"),
        "recovery_era_mean_monthly_count_2025_2026": era_mean(counts, "2025-01", "2026-05"),
        "pre_1b_fraction_months_observed_2019_2021": era_fraction(counts, "2019-01", "2021-12", "has_observation"),
        "single_satellite_fraction_months_observed_2022_2024": era_fraction(counts, "2022-01", "2024-12", "has_observation"),
        "recovery_fraction_months_observed_2025_2026": era_fraction(counts, "2025-01", "2026-05", "has_observation"),
        "recovery_fraction_months_with_s1c_or_s1d_2025_2026": era_fraction(counts, "2025-01", "2026-05", "has_s1c_or_s1d_month"),
    }
    for col, series in era_series.items():
        metrics = metrics.merge(series.rename(col).reset_index(), on="city_id", how="left")

    metrics["loss_after_1b_failure_2022_2024_minus_2019_2021"] = (
        metrics["single_satellite_era_mean_monthly_count_2022_2024"]
        - metrics["pre_1b_baseline_mean_monthly_count_2019_2021"]
    )
    metrics["relative_loss_after_1b_failure"] = ratio_or_nan(
        -metrics["loss_after_1b_failure_2022_2024_minus_2019_2021"],
        metrics["pre_1b_baseline_mean_monthly_count_2019_2021"],
    ).clip(lower=0)
    metrics["single_to_pre_1b_ratio"] = ratio_or_nan(
        metrics["single_satellite_era_mean_monthly_count_2022_2024"],
        metrics["pre_1b_baseline_mean_monthly_count_2019_2021"],
    )
    metrics["recovery_delta_2025_2026_minus_2022_2024"] = (
        metrics["recovery_era_mean_monthly_count_2025_2026"]
        - metrics["single_satellite_era_mean_monthly_count_2022_2024"]
    )
    metrics["recovery_to_single_satellite_ratio"] = ratio_or_nan(
        metrics["recovery_era_mean_monthly_count_2025_2026"],
        metrics["single_satellite_era_mean_monthly_count_2022_2024"],
    )

    # Conservative, interpretable S1-RI: coverage and continuity dominate; orbit,
    # polarization, and 1B-shock resilience contribute smaller terms.
    metrics["score_component_mean_acquisition"] = (metrics["mean_monthly_acquisitions_2017_2026"] / 12.0).clip(0, 1)
    metrics["score_component_observation_continuity"] = metrics["fraction_months_with_observation"].clip(0, 1)
    metrics["score_component_gap"] = (1 - (metrics["longest_observation_gap_days"] / 180.0).clip(0, 1)).clip(0, 1)
    metrics["score_component_dual_pol"] = metrics["fraction_months_with_dual_pol_coverage"].clip(0, 1)
    metrics["score_component_orbit_diversity"] = metrics["fraction_months_with_both_asc_desc"].clip(0, 1)
    metrics["score_component_1b_shock_resilience"] = metrics["single_to_pre_1b_ratio"].fillna(0).clip(0, 1)
    metrics["s1_ri_score"] = (
        0.25 * metrics["score_component_mean_acquisition"]
        + 0.25 * metrics["score_component_observation_continuity"]
        + 0.20 * metrics["score_component_gap"]
        + 0.15 * metrics["score_component_dual_pol"]
        + 0.10 * metrics["score_component_orbit_diversity"]
        + 0.05 * metrics["score_component_1b_shock_resilience"]
    ).clip(0, 1)
    metrics["s1_ri_score_equal_weight_sensitivity"] = metrics[
        [
            "score_component_mean_acquisition",
            "score_component_observation_continuity",
            "score_component_gap",
            "score_component_dual_pol",
            "score_component_orbit_diversity",
            "score_component_1b_shock_resilience",
        ]
    ].mean(axis=1)

    metadata_cols = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "centroid_lon",
        "centroid_lat",
        "population_2020",
        "area_km2",
        "dominant_mode",
        "archetype",
        "data_quality_flag",
    ]
    out = city_table[metadata_cols].merge(metrics, on="city_id", how="inner")
    out["source_inventory_version"] = SOURCE_VERSION
    out["analysis_month_start"] = START_YM
    out["analysis_month_end"] = END_YM

    table1 = (
        out.groupby("continent", dropna=False)
        .agg(
            n_cities=("city_id", "size"),
            mean_s1_ri=("s1_ri_score", "mean"),
            median_s1_ri=("s1_ri_score", "median"),
            mean_monthly_acquisitions=("mean_monthly_acquisitions_2017_2026", "mean"),
            mean_pre_1b_2019_2021=("pre_1b_baseline_mean_monthly_count_2019_2021", "mean"),
            mean_single_satellite_2022_2024=("single_satellite_era_mean_monthly_count_2022_2024", "mean"),
            mean_loss_after_1b=("loss_after_1b_failure_2022_2024_minus_2019_2021", "mean"),
            mean_recovery_2025_2026=("recovery_era_mean_monthly_count_2025_2026", "mean"),
            mean_recovery_delta=("recovery_delta_2025_2026_minus_2022_2024", "mean"),
            mean_fraction_months_observed=("fraction_months_with_observation", "mean"),
            median_longest_gap_days=("longest_observation_gap_days", "median"),
        )
        .reset_index()
    )

    summary = {
        "n_cities": int(len(out)),
        "n_months": int(out["n_months"].iloc[0]) if len(out) else 0,
        "analysis_month_start": START_YM,
        "analysis_month_end": END_YM,
        "mean_s1_ri_score": round(float(out["s1_ri_score"].mean()), 6),
        "median_s1_ri_score": round(float(out["s1_ri_score"].median()), 6),
        "min_s1_ri_score": round(float(out["s1_ri_score"].min()), 6),
        "max_s1_ri_score": round(float(out["s1_ri_score"].max()), 6),
        "mean_monthly_acquisitions": round(float(out["mean_monthly_acquisitions_2017_2026"].mean()), 6),
        "mean_fraction_months_with_observation": round(float(out["fraction_months_with_observation"].mean()), 6),
        "median_longest_gap_days": round(float(out["longest_observation_gap_days"].median()), 6),
        "cities_with_any_zero_observation_month": int((out["longest_observation_gap_days"] > 0).sum()),
        "gap_metric_basis": "monthly_zero_observation_run_days",
        "score_weights": {
            "mean_acquisition_capped_at_12_per_month": 0.25,
            "observation_continuity": 0.25,
            "gap_score_capped_at_180_days": 0.20,
            "dual_polarization_month_fraction": 0.15,
            "both_asc_desc_month_fraction": 0.10,
            "single_to_pre_1b_ratio": 0.05,
        },
    }
    return out, table1, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.cwd()))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    project = root if (root / "data").exists() else root / PROJECT_NAME
    archive_root = project / "archive"
    out_path = project / "data" / "ri_outputs" / "s1_city_reliability_index.csv"
    table1_path = project / "data" / "ri_outputs" / "s1_reliability_metrics_by_region_and_era.csv"
    summary_path = project / "qa" / "s1_reliability_index_qc_summary.json"
    state_path = root / "S1RI_PIPELINE_STATE.json"

    for path in [out_path, table1_path, summary_path]:
        archive_if_exists(path, archive_root)

    out, table1, summary = build_metrics(root)
    out.to_csv(out_path, index=False)
    table1.to_csv(table1_path, index=False)
    summary["output_path"] = str(out_path)
    summary["table1_path"] = str(table1_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        state["last_updated"] = utc_now()
        state["current_phase"] = "phase_4_reliability_metrics"
        state["reliability_index_status"] = "complete"
        state["qc_summary"] = {
            **(state.get("qc_summary") or {}),
            "reliability_index": summary,
        }
        state["next_session_priority"] = "Run Phase 5 publication figures and tables."
        state["estimated_completion_pct"] = 72
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
