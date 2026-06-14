#!/usr/bin/env python3
"""Generate manuscript Figures 1-2 and publication Table I."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


PROJECT_NAME = "Sentinel-1 Observation Reliability & Continuity Index"
FIGURE_DPI = 600

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

S1RI_CMAP = LinearSegmentedColormap.from_list(
    "ocean_coral",
    ["#0a1f3c", "#1f4e6b", "#3f86a0", "#7fb6a3", "#e3a07a", "#f6d061"],
    N=256,
)

REGION_COLORS = {
    "Africa": "#2f6f4e",
    "Latin_America": "#c84b31",
    "South_Asia": "#345995",
    "Southeast_Asia": "#8a5a9e",
}


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    outputs = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_basemap(ax: plt.Axes, extent: tuple[float, float, float, float]):
    """Draw a light geographic basemap when Basemap is available."""
    try:
        from mpl_toolkits.basemap import Basemap
    except Exception:
        return None

    lon0, lon1, lat0, lat1 = extent
    m = Basemap(
        projection="cyl",
        llcrnrlon=lon0,
        urcrnrlon=lon1,
        llcrnrlat=lat0,
        urcrnrlat=lat1,
        resolution="l",
        ax=ax,
    )
    m.drawmapboundary(fill_color="#f4f8fb", linewidth=0.5)
    m.fillcontinents(color="#e9ecef", lake_color="#f4f8fb", zorder=0)
    m.drawcoastlines(linewidth=0.35, color="#9aa3ad", zorder=1)
    m.drawcountries(linewidth=0.2, color="#c2c8cf", zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#b8bec5")
    return m


def add_event_markers(ax: plt.Axes) -> None:
    events = [
        (pd.Timestamp("2021-12-01"), "S1B failure"),
        (pd.Timestamp("2025-04-01"), "S1C recovery"),
        (pd.Timestamp("2026-04-01"), "S1D opening"),
    ]
    ymax = ax.get_ylim()[1]
    for date, label in events:
        ax.axvline(date, color="#404040", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.text(
            date,
            ymax * 0.98,
            label,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="#303030",
        )


def build_figure_1(
    ri: pd.DataFrame,
    figure_dir: Path,
    input_dir: Path,
) -> tuple[list[str], str]:
    columns = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "centroid_lon",
        "centroid_lat",
        "s1_ri_score",
        "mean_monthly_acquisitions_2017_2026",
        "fraction_months_with_observation",
    ]
    points = ri[columns].copy()
    input_path = input_dir / "figure1_s1_ri_city_points.csv"
    points.to_csv(input_path, index=False)

    extent = (-125, 155, -42, 42)
    fig, ax = plt.subplots(figsize=(6.5, 2.25))
    basemap = draw_basemap(ax, extent)
    if basemap is not None:
        xs, ys = basemap(points["centroid_lon"].values, points["centroid_lat"].values)
    else:
        xs, ys = points["centroid_lon"].values, points["centroid_lat"].values
        style_axes(ax)

    scatter = ax.scatter(
        xs,
        ys,
        c=points["s1_ri_score"],
        cmap=S1RI_CMAP,
        s=10,
        linewidths=0.2,
        edgecolors="white",
        alpha=0.92,
        vmin=0,
        vmax=1,
        zorder=3,
    )
    ax.set_xticks([-100, -50, 0, 50, 100, 150])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_xlim(*extent[:2])
    ax.set_ylim(*extent[2:])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.012, aspect=20)
    colorbar.set_label("S1-RI score", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.outline.set_linewidth(0.5)

    outputs = save_figure(fig, figure_dir / "figure1_global_s1_ri_map")
    return outputs, str(input_path)


def build_figure_2(
    counts: pd.DataFrame,
    city_meta: pd.DataFrame,
    figure_dir: Path,
    input_dir: Path,
) -> tuple[list[str], str]:
    counts = counts.merge(city_meta[["city_id", "continent"]], on="city_id", how="left")
    counts["date"] = pd.to_datetime(
        counts["year"].astype(str)
        + "-"
        + counts["month"].astype(str).str.zfill(2)
        + "-01"
    )
    timeseries = (
        counts.groupby(["continent", "date"], as_index=False)
        .agg(
            mean_monthly_acquisitions=("n_total", "mean"),
            median_monthly_acquisitions=("n_total", "median"),
            fraction_cities_observed=("has_observation", "mean"),
            n_cities=("city_id", "nunique"),
        )
        .sort_values(["continent", "date"])
    )
    input_path = input_dir / "figure2_region_monthly_timeseries.csv"
    timeseries.to_csv(input_path, index=False)

    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    for region, group in timeseries.groupby("continent"):
        ax.plot(
            group["date"],
            group["mean_monthly_acquisitions"],
            label=region.replace("_", " "),
            color=REGION_COLORS.get(region, "#555555"),
            linewidth=1.25,
        )

    ax.set_ylabel("Mean acquisitions per city-month")
    ax.set_xlabel("Acquisition month")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axes(ax)
    add_event_markers(ax)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=0.9,
        handlelength=1.8,
    )
    fig.tight_layout(pad=0.45)

    outputs = save_figure(fig, figure_dir / "figure2_regional_monthly_acquisitions")
    return outputs, str(input_path)


def publication_table(table: pd.DataFrame, output_path: Path) -> str:
    output = table.copy()
    numeric = output.select_dtypes(include=[np.number]).columns
    output[numeric] = output[numeric].round(3)
    output.to_csv(output_path, index=False)
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.cwd()))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    project = root if (root / "data").exists() else root / PROJECT_NAME
    figure_dir = project / "figures"
    input_dir = project / "data" / "ri_outputs" / "figure_inputs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    ri = pd.read_csv(project / "data" / "ri_outputs" / "s1_city_reliability_index.csv")
    counts = pd.read_csv(
        project / "data" / "city_counts" / "s1_city_monthly_counts_2017_2026.csv"
    )
    table = pd.read_csv(
        project
        / "data"
        / "ri_outputs"
        / "s1_reliability_metrics_by_region_and_era.csv"
    )
    city_meta = ri[["city_id", "continent"]].copy()

    figure_1, input_1 = build_figure_1(ri, figure_dir, input_dir)
    figure_2, input_2 = build_figure_2(counts, city_meta, figure_dir, input_dir)
    table_path = publication_table(
        table,
        project
        / "data"
        / "ri_outputs"
        / "table1_reliability_metrics_by_region_and_era_publication.csv",
    )

    print(
        json.dumps(
            {
                "figure1": {"files": figure_1, "input": input_1},
                "figure2": {"files": figure_2, "input": input_2},
                "table1_publication_csv": table_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
