#!/usr/bin/env python3
"""Collect Sentinel-1 IW GRDH metadata from Copernicus Data Space OData.

This script is intentionally page-checkpointed so it can be run repeatedly by
the pipeline without redoing completed months.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import calendar
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
PROJECT_NAME = "Sentinel-1 Observation Reliability & Continuity Index"
START_MONTH = "2017-01"
END_MONTH = "2026-06"
TOP = 1000
MAX_RETRIES = 3
REQUEST_TIMEOUT = 90


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


def next_month(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    m += 1
    if m == 13:
        y += 1
        m = 1
    return f"{y:04d}-{m:02d}"


def ensure_dirs(project_dir: Path) -> dict[str, Path]:
    paths = {
        "raw": project_dir / "data" / "global_s1_metadata" / "raw_monthly",
        "processed": project_dir / "data" / "global_s1_metadata" / "processed_monthly",
        "archive": project_dir / "archive",
        "qa": project_dir / "qa",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {
        "last_updated": utc_now(),
        "current_phase": "phase_1_global_metadata_inventory",
        "completed_months": [],
        "running_month": None,
        "failed_months": [],
        "current_nextLink": None,
        "pages_completed_this_run": 0,
        "products_collected_this_run": 0,
        "total_products_collected": 0,
        "global_inventory_status": "initialized",
        "city_intersection_status": "not_started",
        "reliability_index_status": "not_started",
        "figures_status": "not_started",
        "qc_summary": {},
        "blockers": [],
        "next_session_priority": "Collect Sentinel-1 IW GRDH monthly metadata.",
        "estimated_completion_pct": 1,
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["last_updated"] = utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def archive_file(path: Path, archive_root: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = archive_root / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    shutil.move(str(path), str(dest))
    return dest


def request_json(url: str | None = None, params: dict[str, str] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if url:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            else:
                resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:500]}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - exact error is persisted in state.
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)
    raise RuntimeError(str(last_error))


def build_params(start_iso: str, end_iso: str, *, count: bool = False) -> dict[str, str]:
    flt = (
        "Collection/Name eq 'SENTINEL-1' "
        "and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S') "
        f"and ContentDate/Start ge {start_iso} "
        f"and ContentDate/Start lt {end_iso}"
    )
    params = {
        "$filter": flt,
        "$expand": "Attributes",
        "$top": str(TOP),
    }
    if count:
        params["$count"] = "true"
    return params


def month_bounds(ym: str) -> tuple[str, str]:
    return f"{ym}-01T00:00:00.000Z", f"{next_month(ym)}-01T00:00:00.000Z"


def day_windows(ym: str) -> list[tuple[str, str]]:
    year, month = map(int, ym.split("-"))
    ndays = calendar.monthrange(year, month)[1]
    windows = []
    for day in range(1, ndays + 1):
        start = f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000Z"
        if day == ndays:
            end = f"{next_month(ym)}-01T00:00:00.000Z"
        else:
            end = f"{year:04d}-{month:02d}-{day + 1:02d}T00:00:00.000Z"
        windows.append((start, end))
    return windows


def expected_month_count(ym: str) -> int:
    start, end = month_bounds(ym)
    data = request_json(params=build_params(start, end, count=True) | {"$top": "0"})
    return int(data.get("@odata.count", 0))


def attr_map(product: dict[str, Any]) -> dict[str, Any]:
    attrs = {}
    for item in product.get("Attributes") or []:
        name = item.get("Name")
        if name:
            attrs[name] = item.get("Value")
    return attrs


def parse_name(name: str) -> dict[str, Any]:
    base = name.replace(".SAFE", "")
    parts = base.split("_")
    platform = parts[0] if parts else None
    product_type = "_".join(parts[1:4]) if len(parts) >= 4 else None
    pol_code = parts[3] if len(parts) >= 4 else None
    start_name = parts[4] if len(parts) >= 5 else None
    end_name = parts[5] if len(parts) >= 6 else None
    is_cog = "_COG" in name or name.endswith("_COG.SAFE")
    return {
        "platform": platform if platform in {"S1A", "S1B", "S1C", "S1D"} else None,
        "product_type_from_name": product_type,
        "pol_code": pol_code,
        "name_start": start_name,
        "name_end": end_name,
        "is_cog": is_cog,
    }


def polarization_flags(pol_code: str | None, channels: str | None) -> dict[str, Any]:
    text = " ".join([pol_code or "", channels or ""]).upper()
    code = (pol_code or "").upper()
    has_vv = "VV" in text or code.endswith("SV") or code.endswith("DV")
    has_vh = "VH" in text or code.endswith("DV")
    has_hh = "HH" in text or code.endswith("SH") or code.endswith("DH")
    has_hv = "HV" in text or code.endswith("DH")
    dual = ("DV" in text) or ("DH" in text) or ("VV" in text and "VH" in text) or ("HH" in text and "HV" in text)
    return {
        "has_vv": has_vv,
        "has_vh": has_vh,
        "has_hh": has_hh,
        "has_hv": has_hv,
        "is_dual_pol": dual,
    }


def compact_row(product: dict[str, Any]) -> dict[str, Any]:
    attrs = attr_map(product)
    parsed = parse_name(product.get("Name", ""))
    pol = polarization_flags(parsed.get("pol_code"), attrs.get("polarisationChannels"))
    content = product.get("ContentDate") or {}
    footprint = product.get("GeoFootprint")
    return {
        "Id": product.get("Id"),
        "Name": product.get("Name"),
        "S3Path": product.get("S3Path"),
        "content_start": content.get("Start"),
        "content_end": content.get("End"),
        "platform": parsed.get("platform"),
        "product_type": attrs.get("productType") or parsed.get("product_type_from_name"),
        "pol_code": parsed.get("pol_code"),
        "polarisation_channels": attrs.get("polarisationChannels"),
        "orbit_direction": attrs.get("orbitDirection"),
        "relative_orbit": attrs.get("relativeOrbitNumber"),
        "orbit_number": attrs.get("orbitNumber"),
        "is_cog": parsed.get("is_cog"),
        "has_footprint": footprint is not None,
        "footprint_json": json.dumps(footprint, separators=(",", ":"), ensure_ascii=False) if footprint else None,
        **pol,
    }


def write_processed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "Id",
        "Name",
        "S3Path",
        "content_start",
        "content_end",
        "platform",
        "product_type",
        "pol_code",
        "polarisation_channels",
        "orbit_direction",
        "relative_orbit",
        "orbit_number",
        "is_cog",
        "has_footprint",
        "has_vv",
        "has_vh",
        "has_hh",
        "has_hv",
        "is_dual_pol",
        "footprint_json",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def raw_file_is_readable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    return True
    except Exception:
        return False
    return False


def count_raw_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def month_complete(ym: str, raw_dir: Path, processed_dir: Path, state: dict[str, Any]) -> bool:
    raw_path = raw_dir / f"s1_iw_grdh_{ym.replace('-', '_')}.jsonl"
    processed_path = processed_dir / f"s1_iw_grdh_{ym.replace('-', '_')}.csv.gz"
    monthly_qc = (state.get("monthly_qc") or {}).get(ym) or {}
    if not monthly_qc.get("attributes_expanded"):
        return False
    expected = (state.get("month_expected_counts") or {}).get(ym)
    if expected is not None and count_raw_lines(raw_path) != int(expected):
        return False
    return (
        ym in set(state.get("completed_months", []))
        and raw_file_is_readable(raw_path)
        and processed_path.exists()
        and processed_path.stat().st_size > 0
    )


def collect_month(
    ym: str,
    paths: dict[str, Path],
    state: dict[str, Any],
    state_path: Path,
) -> tuple[str, int, int, str | None]:
    raw_path = paths["raw"] / f"s1_iw_grdh_{ym.replace('-', '_')}.jsonl"
    processed_path = paths["processed"] / f"s1_iw_grdh_{ym.replace('-', '_')}.csv.gz"

    if raw_path.exists():
        archive_file(raw_path, paths["archive"])
    if processed_path.exists():
        archive_file(processed_path, paths["archive"])

    expected_count = expected_month_count(ym)
    month_expected_counts = state.get("month_expected_counts") or {}
    month_expected_counts[ym] = expected_count
    state["month_expected_counts"] = month_expected_counts

    rows: list[dict[str, Any]] = []
    page = 0
    count = 0
    next_link: str | None = None
    state["running_month"] = ym
    state["current_nextLink"] = None
    state["global_inventory_status"] = "running"
    save_state(state_path, state)

    try:
        with raw_path.open("w", encoding="utf-8", newline="\n") as raw:
            for start_iso, end_iso in day_windows(ym):
                next_link = None
                while True:
                    if next_link:
                        data = request_json(url=next_link)
                    else:
                        data = request_json(params=build_params(start_iso, end_iso))

                    products = data.get("value", [])
                    page += 1
                    for product in products:
                        raw.write(json.dumps(product, ensure_ascii=False, separators=(",", ":")) + "\n")
                        rows.append(compact_row(product))
                    raw.flush()

                    count += len(products)
                    next_link = data.get("@odata.nextLink")
                    state["current_nextLink"] = next_link
                    state["pages_completed_this_run"] = int(state.get("pages_completed_this_run", 0)) + 1
                    state["products_collected_this_run"] = int(state.get("products_collected_this_run", 0)) + len(products)
                    state["total_products_collected"] = int(state.get("total_products_collected", 0)) + len(products)
                    state["qc_summary"] = {
                        **(state.get("qc_summary") or {}),
                        "last_month_page_checkpoint": {
                            "month": ym,
                            "window_start": start_iso,
                            "window_end": end_iso,
                            "page": page,
                            "products_in_page": len(products),
                            "products_in_month_so_far": count,
                            "expected_products_in_month": expected_count,
                        },
                    }
                    save_state(state_path, state)

                    if not next_link:
                        break

        if count != expected_count:
            raise RuntimeError(f"Monthly count mismatch for {ym}: collected {count}, expected {expected_count}")

        write_processed_csv(processed_path, rows)
        completed = list(dict.fromkeys([*state.get("completed_months", []), ym]))
        state["completed_months"] = completed
        state["running_month"] = None
        state["current_nextLink"] = None
        monthly_qc = state.get("monthly_qc") or {}
        monthly_qc[ym] = {
            "expected_products": expected_count,
            "collected_products": count,
            "pages": page,
            "status": "complete",
            "processed_format": "csv.gz",
            "attributes_expanded": any(row.get("orbit_direction") for row in rows),
        }
        state["monthly_qc"] = monthly_qc
        state["global_inventory_status"] = "phase_1_in_progress"
        save_state(state_path, state)
        return "complete", count, page, None
    except Exception as exc:  # noqa: BLE001 - exact error is part of the required checkpoint.
        failed = state.get("failed_months", [])
        entry = {"month": ym, "error": repr(exc), "timestamp": utc_now()}
        failed.append(entry)
        state["failed_months"] = failed
        state["running_month"] = None
        state["global_inventory_status"] = "phase_1_in_progress_with_failures"
        state["current_nextLink"] = next_link
        save_state(state_path, state)
        return "failed", count, page, repr(exc)


def update_completion_state(state: dict[str, Any], all_months: list[str]) -> None:
    completed = set(state.get("completed_months", []))
    failed_entries = state.get("failed_months", [])
    failed = {x.get("month") for x in failed_entries if isinstance(x, dict)}
    n_complete = len([m for m in all_months if m in completed])
    state["current_phase"] = "phase_1_global_metadata_inventory"
    if n_complete == len(all_months):
        state["global_inventory_status"] = "monthly_metadata_complete_needs_global_merge"
    elif failed:
        state["global_inventory_status"] = "phase_1_in_progress_with_failures"
    else:
        state["global_inventory_status"] = "phase_1_in_progress"
    state["estimated_completion_pct"] = max(1, min(35, round((n_complete / len(all_months)) * 35, 1)))
    remaining = [m for m in all_months if m not in completed and m not in failed]
    state["next_session_priority"] = (
        f"Continue Phase 1 monthly metadata pull at {remaining[0]}." if remaining else "Merge monthly inventory and run QC."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--max-months", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    project_dir = root if (root / "data").exists() else root / PROJECT_NAME
    state_path = root / "S1RI_PIPELINE_STATE.json"
    paths = ensure_dirs(project_dir)
    state = load_state(state_path)
    state["pages_completed_this_run"] = 0
    state["products_collected_this_run"] = 0
    save_state(state_path, state)

    months = month_iter(START_MONTH, END_MONTH)
    completed_months = list(state.get("completed_months", []))
    partial_months: dict[str, Any] = {}
    expected_counts = state.get("month_expected_counts") or {}
    for ym in list(completed_months):
        raw_path = paths["raw"] / f"s1_iw_grdh_{ym.replace('-', '_')}.jsonl"
        processed_path = paths["processed"] / f"s1_iw_grdh_{ym.replace('-', '_')}.csv.gz"
        if not raw_path.exists() or not processed_path.exists():
            completed_months.remove(ym)
            partial_months[ym] = {"reason": "missing raw or processed file"}
            continue
        if ym not in expected_counts:
            expected_counts[ym] = expected_month_count(ym)
        lines = count_raw_lines(raw_path)
        if lines != int(expected_counts[ym]):
            completed_months.remove(ym)
            partial_months[ym] = {
                "reason": "raw line count does not match catalogue count",
                "raw_lines": lines,
                "expected_products": int(expected_counts[ym]),
            }
    state["completed_months"] = completed_months
    state["month_expected_counts"] = expected_counts
    if partial_months:
        state["qc_summary"] = {**(state.get("qc_summary") or {}), "partial_months_reset": partial_months}
        save_state(state_path, state)
    failed_months = {x.get("month") for x in state.get("failed_months", []) if isinstance(x, dict)}
    candidates = [
        m
        for m in months
        if not month_complete(m, paths["raw"], paths["processed"], state)
        and m not in failed_months
    ]
    chosen = candidates[: max(0, args.max_months)]

    summary = {
        "chosen_months": chosen,
        "completed": [],
        "failed": [],
        "products": 0,
        "pages": 0,
    }

    for ym in chosen:
        status, count, pages, error = collect_month(ym, paths, state, state_path)
        summary["products"] += count
        summary["pages"] += pages
        if status == "complete":
            summary["completed"].append({"month": ym, "products": count, "pages": pages})
        else:
            summary["failed"].append({"month": ym, "products_before_failure": count, "pages": pages, "error": error})
            continue

    update_completion_state(state, months)
    save_state(state_path, state)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
