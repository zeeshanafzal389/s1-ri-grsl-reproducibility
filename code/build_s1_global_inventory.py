#!/usr/bin/env python3
"""Build the Phase 2 cleaned global Sentinel-1 inventory and QC tables."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_NAME = "Sentinel-1 Observation Reliability & Continuity Index"
INVENTORY_START = "2017-01"
INVENTORY_END = "2026-05"
SOURCE_VERSION = "copernicus_odata_s1_iw_grdh_201701_202605_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_iter(start_ym: str, end_ym: str) -> list[str]:
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    out: list[str] = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def archive_if_exists(path: Path, archive_root: Path) -> None:
    if not path.exists():
        return
    dest_dir = archive_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest_dir / path.name))


def product_key_from_name(name: str) -> str:
    key = (name or "").replace(".SAFE", "")
    key = key.replace("_COG", "")
    return key


def acquisition_id_from_name(name: str) -> str:
    stem = product_key_from_name(name)
    parts = stem.split("_")
    if len(parts) >= 7:
        return "_".join(parts[:7])
    return stem


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_month(path: Path, ym: str) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    year, month = map(int, ym.split("-"))
    df["year"] = year
    df["month"] = month
    df["year_month"] = ym
    df["source_inventory_version"] = SOURCE_VERSION

    for col in ["is_cog", "has_footprint", "has_vv", "has_vh", "has_hh", "has_hv", "is_dual_pol"]:
        if col in df.columns:
            df[col] = normalize_bool(df[col])

    df["platform"] = df["Name"].astype(str).str.extract(r"^(S1[A-D])_", expand=False).fillna(df.get("platform"))
    df["product_key"] = df["Name"].astype(str).map(product_key_from_name)
    df["acquisition_id"] = df["Name"].astype(str).map(acquisition_id_from_name)
    df["content_start"] = pd.to_datetime(df["content_start"], errors="coerce", utc=True)
    df["content_end"] = pd.to_datetime(df["content_end"], errors="coerce", utc=True)
    df["content_date"] = df["content_start"].dt.date.astype("string")
    return df


def summarize_month(df: pd.DataFrame, expected: int | None) -> dict:
    n = len(df)
    return {
        "month": df["year_month"].iloc[0],
        "n_products_raw": n,
        "expected_products": expected,
        "count_matches_catalogue": n == expected if expected is not None else None,
        "n_s1a": int((df["platform"] == "S1A").sum()),
        "n_s1b": int((df["platform"] == "S1B").sum()),
        "n_s1c": int((df["platform"] == "S1C").sum()),
        "n_s1d": int((df["platform"] == "S1D").sum()),
        "n_dual_pol": int(df["is_dual_pol"].sum()),
        "n_vv": int(df["has_vv"].sum()),
        "n_vh": int(df["has_vh"].sum()),
        "n_hh": int(df["has_hh"].sum()),
        "n_hv": int(df["has_hv"].sum()),
        "n_asc": int((df["orbit_direction"] == "ASCENDING").sum()),
        "n_desc": int((df["orbit_direction"] == "DESCENDING").sum()),
        "n_orbit_missing": int(df["orbit_direction"].isna().sum() + (df["orbit_direction"].astype(str).str.len() == 0).sum()),
        "n_cog": int(df["is_cog"].sum()),
        "cog_duplicate_rate": round(float(df["is_cog"].mean()), 8) if n else 0,
        "missing_footprint_rate": round(float((~df["has_footprint"]).mean()), 8) if n else 0,
        "missing_date_rate": round(float(df["content_start"].isna().mean()), 8) if n else 0,
        "n_unique_product_keys": int(df["product_key"].nunique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.cwd()))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    project = root if (root / "data").exists() else root / PROJECT_NAME
    processed_dir = project / "data" / "global_s1_metadata" / "processed_monthly"
    archive_dir = project / "archive"
    state_path = root / "S1RI_PIPELINE_STATE.json"

    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    months = month_iter(INVENTORY_START, INVENTORY_END)
    completed = set(state.get("completed_months", []))
    missing = []
    paths: list[tuple[str, Path]] = []
    for ym in months:
        path = processed_dir / f"s1_iw_grdh_{ym.replace('-', '_')}.csv.gz"
        if ym not in completed or not path.exists():
            missing.append(ym)
        else:
            paths.append((ym, path))
    if missing:
        raise RuntimeError(f"Missing required completed monthly files for Phase 2: {missing[:20]}")

    inventory_path = project / "data" / "global_s1_metadata" / "s1_global_inventory_2017_2026.parquet"
    dedup_inventory_path = project / "data" / "global_s1_metadata" / "s1_global_inventory_2017_2026_dedup.parquet"
    duplicate_summary_path = project / "data" / "global_s1_metadata" / "s1_global_inventory_duplicate_summary.csv"
    qc_path = project / "data" / "global_s1_metadata" / "s1_global_inventory_qc.csv"
    summary_path = project / "data" / "global_s1_metadata" / "s1_global_inventory_qc_summary.json"

    for out in [inventory_path, dedup_inventory_path, duplicate_summary_path, qc_path, summary_path]:
        archive_if_exists(out, archive_dir)

    monthly_qc = []
    chunks = []
    for ym, path in paths:
        df = load_month(path, ym)
        expected = (state.get("month_expected_counts") or {}).get(ym)
        monthly_qc.append(summarize_month(df, int(expected) if expected is not None else None))
        chunks.append(df)

    all_df = pd.concat(chunks, ignore_index=True)
    all_df.sort_values(["content_start", "Name", "Id"], inplace=True, kind="mergesort")

    duplicate_counts = all_df.groupby("product_key", dropna=False).agg(
        n_versions=("Id", "size"),
        n_cog=("is_cog", "sum"),
        n_non_cog=("is_cog", lambda x: int((~x).sum())),
        first_month=("year_month", "min"),
        first_name=("Name", "first"),
    )
    duplicate_keys = duplicate_counts[duplicate_counts["n_versions"] > 1].reset_index()
    duplicate_keys.to_csv(duplicate_summary_path, index=False)

    dedup_df = all_df.sort_values(["product_key", "is_cog", "content_start"], kind="mergesort").drop_duplicates(
        subset=["product_key"], keep="first"
    )
    dedup_df.sort_values(["content_start", "Name", "Id"], inplace=True, kind="mergesort")

    all_df.to_parquet(inventory_path, index=False)
    dedup_df.to_parquet(dedup_inventory_path, index=False)

    qc_df = pd.DataFrame(monthly_qc)
    qc_df["n_products_dedup"] = dedup_df.groupby("year_month").size().reindex(qc_df["month"]).fillna(0).astype(int).values
    qc_df["n_removed_as_duplicates"] = qc_df["n_products_raw"] - qc_df["n_products_dedup"]
    qc_df.to_csv(qc_path, index=False)

    pre_1b = qc_df[(qc_df["month"] >= "2019-01") & (qc_df["month"] <= "2021-12")]["n_products_dedup"].mean()
    single_sat = qc_df[(qc_df["month"] >= "2022-01") & (qc_df["month"] <= "2024-12")]["n_products_dedup"].mean()
    recovery = qc_df[(qc_df["month"] >= "2025-01") & (qc_df["month"] <= "2026-05")]["n_products_dedup"].mean()
    s1c_2025_2026 = int(dedup_df[(dedup_df["platform"] == "S1C") & (dedup_df["year_month"] >= "2025-01")].shape[0])
    s1d_2026 = int(dedup_df[(dedup_df["platform"] == "S1D") & (dedup_df["year_month"] >= "2026-01")].shape[0])

    summary = {
        "inventory_version": SOURCE_VERSION,
        "inventory_month_start": INVENTORY_START,
        "inventory_month_end": INVENTORY_END,
        "n_months": len(months),
        "n_products_raw": int(len(all_df)),
        "n_products_dedup": int(len(dedup_df)),
        "n_duplicate_product_keys": int(len(duplicate_keys)),
        "n_removed_as_duplicates": int(len(all_df) - len(dedup_df)),
        "cog_duplicate_rate_raw": round(float(all_df["is_cog"].mean()), 8),
        "missing_footprint_rate_raw": round(float((~all_df["has_footprint"]).mean()), 8),
        "missing_date_rate_raw": round(float(all_df["content_start"].isna().mean()), 8),
        "zero_count_months": qc_df.loc[qc_df["n_products_raw"] == 0, "month"].tolist(),
        "pre_1b_mean_monthly_dedup_2019_2021": round(float(pre_1b), 2),
        "single_satellite_mean_monthly_dedup_2022_2024": round(float(single_sat), 2),
        "recovery_mean_monthly_dedup_2025_2026_may": round(float(recovery), 2),
        "s1b_loss_visible_after_dec_2021": bool(single_sat < pre_1b * 0.8),
        "s1c_recovery_visible_2025_2026": bool(s1c_2025_2026 > 0 and recovery > single_sat * 1.2),
        "s1d_visible_2026": bool(s1d_2026 > 0),
        "n_s1c_2025_2026": s1c_2025_2026,
        "n_s1d_2026": s1d_2026,
        "inventory_path": str(inventory_path),
        "dedup_inventory_path": str(dedup_inventory_path),
        "duplicate_summary_path": str(duplicate_summary_path),
        "qc_path": str(qc_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    state["last_updated"] = utc_now()
    state["current_phase"] = "phase_2_metadata_cleaning_deduplication"
    state["running_month"] = None
    state["current_nextLink"] = None
    state["failed_months"] = [
        f for f in state.get("failed_months", []) if not (isinstance(f, dict) and f.get("month") == "2026-06")
    ]
    state["global_inventory_status"] = "complete_2017_01_to_2026_05"
    state["total_products_collected"] = int(summary["n_products_raw"])
    state["qc_summary"] = {
        **(state.get("qc_summary") or {}),
        "inventory_scope_note": "User confirmed 2026-06 is not needed on 2026-06-07; Phase 2 uses completed months through 2026-05.",
        "phase2_summary": summary,
        "qc_failures": [],
    }
    state["blockers"] = []
    state["next_session_priority"] = "Run Phase 3 city intersection using the deduplicated global inventory."
    state["estimated_completion_pct"] = 45
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
