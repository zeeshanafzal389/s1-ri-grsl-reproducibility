#!/usr/bin/env python3
"""Intersect the deduplicated Sentinel-1 inventory with city boundaries."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape


PROJECT_NAME = "Sentinel-1 Observation Reliability & Continuity Index"
SOURCE_VERSION = "copernicus_odata_s1_iw_grdh_201701_202605_v1"
START_MONTH = "2017-01"
END_MONTH = "2026-05"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_iter(start_ym: str, end_ym: str) -> list[str]:
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    out = []
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


def load_main_cities(city_geojson: Path, city_table: Path) -> gpd.GeoDataFrame:
    if not city_geojson.exists():
        raise FileNotFoundError(f"City-boundary GeoJSON not found: {city_geojson}")
    if not city_table.exists():
        raise FileNotFoundError(f"City attribute table not found: {city_table}")
    cities = gpd.read_file(city_geojson)
    table = pd.read_csv(city_table, low_memory=False)
    main_ids = table.loc[table["data_quality_flag"] == "v2_endpoint_shards", ["city_id"]]
    cities = cities.merge(main_ids, on="city_id", how="inner")
    cities = cities[["city_id", "geometry"]].copy()
    cities = cities.set_crs("EPSG:4326", allow_override=True)
    cities["geometry"] = cities.geometry.make_valid()
    return cities


def scene_geometries(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometries = []
    keep = []
    for idx, text in enumerate(df["footprint_json"].fillna("")):
        if not text:
            continue
        try:
            geometries.append(shape(json.loads(text)))
            keep.append(idx)
        except Exception:
            continue
    subset = df.iloc[keep].copy()
    subset["geometry"] = geometries
    return gpd.GeoDataFrame(subset, geometry="geometry", crs="EPSG:4326")


def count_month(df: pd.DataFrame, cities: gpd.GeoDataFrame, ym: str, pass_level: bool = False) -> pd.DataFrame:
    scenes = scene_geometries(df)
    year, month = map(int, ym.split("-"))
    if scenes.empty:
        return pd.DataFrame(columns=["city_id", "year", "month"])

    joined = gpd.sjoin(
        scenes[
            [
                "platform",
                "orbit_direction",
                "is_dual_pol",
                "has_vv",
                "has_vh",
                "orbit_number",
                "content_date",
                "geometry",
            ]
        ],
        cities,
        how="inner",
        predicate="intersects",
    )
    # Pass-level deduplication: collapse contiguous along-track slice products
    # from the same satellite pass (unique absolute orbit) over a city into one
    # observation, so a single overpass is not counted multiple times.
    if pass_level and not joined.empty:
        joined = joined.drop_duplicates(
            subset=["city_id", "platform", "orbit_number", "content_date"]
        )
    if joined.empty:
        return pd.DataFrame(columns=["city_id", "year", "month"])

    flags = pd.DataFrame(
        {
            "city_id": joined["city_id"].values,
            "year": year,
            "month": month,
            "is_s1a": (joined["platform"] == "S1A").astype(int).values,
            "is_s1b": (joined["platform"] == "S1B").astype(int).values,
            "is_s1c": (joined["platform"] == "S1C").astype(int).values,
            "is_s1d": (joined["platform"] == "S1D").astype(int).values,
            "is_dual_pol": joined["is_dual_pol"].astype(bool).astype(int).values,
            "has_vv": joined["has_vv"].astype(bool).astype(int).values,
            "has_vh": joined["has_vh"].astype(bool).astype(int).values,
            "is_asc": (joined["orbit_direction"] == "ASCENDING").astype(int).values,
            "is_desc": (joined["orbit_direction"] == "DESCENDING").astype(int).values,
        }
    )
    out = (
        flags.groupby(["city_id", "year", "month"], as_index=False)
        .agg(
            n_total=("city_id", "size"),
            n_s1a=("is_s1a", "sum"),
            n_s1b=("is_s1b", "sum"),
            n_s1c=("is_s1c", "sum"),
            n_s1d=("is_s1d", "sum"),
            n_dual_pol=("is_dual_pol", "sum"),
            n_vv=("has_vv", "sum"),
            n_vh=("has_vh", "sum"),
            n_asc=("is_asc", "sum"),
            n_desc=("is_desc", "sum"),
        )
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--city-geojson", required=True, help="Local city-boundary GeoJSON path.")
    parser.add_argument(
        "--city-table",
        default="",
        help="City attribute table path. Defaults to the packaged source-city statistics table.",
    )
    parser.add_argument("--max-months", type=int, default=0, help="0 means all months.")
    parser.add_argument(
        "--pass-level",
        action="store_true",
        help="Count one observation per satellite pass (city, platform, absolute "
        "orbit, date) instead of per product, removing along-track slice multiplicity.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    project = root if (root / "data").exists() else root / PROJECT_NAME
    inventory = project / "data" / "global_s1_metadata" / "s1_global_inventory_2017_2026_dedup.parquet"
    city_geojson = Path(args.city_geojson).resolve()
    city_table = (
        Path(args.city_table).resolve()
        if args.city_table
        else project / "data" / "source_city_data" / "statistics_v2" / "supplementary_table1_all_cities_v2.csv"
    )
    suffix = "_passlevel" if args.pass_level else ""
    out_path = project / "data" / "city_counts" / f"s1_city_monthly_counts_2017_2026{suffix}.csv"
    progress_path = project / "data" / "city_counts" / f"s1_city_monthly_counts_progress{suffix}.csv"
    summary_path = project / "data" / "city_counts" / f"s1_city_intersection_summary{suffix}.csv"
    state_path = root / "S1RI_PIPELINE_STATE.json"

    archive_if_exists(out_path, project / "archive")
    archive_if_exists(progress_path, project / "archive")
    archive_if_exists(summary_path, project / "archive")

    cities = load_main_cities(city_geojson, city_table)
    months = month_iter(START_MONTH, END_MONTH)
    if args.max_months:
        months = months[: args.max_months]

    columns = [
        "Id",
        "platform",
        "orbit_direction",
        "is_dual_pol",
        "has_vv",
        "has_vh",
        "orbit_number",
        "content_date",
        "footprint_json",
        "year_month",
    ]
    all_counts = []
    qc_rows = []
    for ym in months:
        df = pd.read_parquet(inventory, columns=columns, filters=[("year_month", "==", ym)])
        month_counts = count_month(df, cities, ym, pass_level=args.pass_level)
        all_counts.append(month_counts)
        qc_rows.append(
            {
                "month": ym,
                "n_inventory_scenes": int(len(df)),
                "n_city_scene_intersections": int(month_counts["n_total"].sum()) if not month_counts.empty else 0,
                "n_cities_with_observation": int(month_counts["city_id"].nunique()) if not month_counts.empty else 0,
                "n_main_cities": int(len(cities)),
            }
        )
        pd.DataFrame(qc_rows).to_csv(progress_path, index=False)

    counts = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    full_index = pd.MultiIndex.from_product(
        [cities["city_id"].sort_values(), months], names=["city_id", "year_month"]
    ).to_frame(index=False)
    full_index["year"] = full_index["year_month"].str.slice(0, 4).astype(int)
    full_index["month"] = full_index["year_month"].str.slice(5, 7).astype(int)
    if not counts.empty:
        merged = full_index.merge(counts, on=["city_id", "year", "month"], how="left")
    else:
        merged = full_index.copy()

    count_cols = [
        "n_total",
        "n_s1a",
        "n_s1b",
        "n_s1c",
        "n_s1d",
        "n_dual_pol",
        "n_vv",
        "n_vh",
        "n_asc",
        "n_desc",
    ]
    for col in count_cols:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)
    merged["has_observation"] = merged["n_total"] > 0
    merged["source_inventory_version"] = SOURCE_VERSION
    merged = merged[
        [
            "city_id",
            "year",
            "month",
            *count_cols,
            "has_observation",
            "source_inventory_version",
        ]
    ].sort_values(["city_id", "year", "month"])
    merged.to_csv(out_path, index=False)
    pd.DataFrame(qc_rows).to_csv(summary_path, index=False)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        state["last_updated"] = utc_now()
        state["current_phase"] = "phase_3_city_intersection"
        state["city_intersection_status"] = "complete" if not args.max_months else f"test_complete_{len(months)}_months"
        state["qc_summary"] = {
            **(state.get("qc_summary") or {}),
            "city_intersection": {
                "months_processed": len(months),
                "main_cities": int(len(cities)),
                "output_rows": int(len(merged)),
                "output_path": str(out_path),
                "summary_path": str(summary_path),
                "test_mode": bool(args.max_months),
            },
        }
        state["next_session_priority"] = (
            "Run Phase 4 reliability metrics." if not args.max_months else "Run full Phase 3 city intersection."
        )
        state["estimated_completion_pct"] = 60 if not args.max_months else max(state.get("estimated_completion_pct", 45), 48)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "months_processed": len(months),
                "main_cities": int(len(cities)),
                "output_rows": int(len(merged)),
                "output_path": str(out_path),
                "qa_path": str(qa_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
