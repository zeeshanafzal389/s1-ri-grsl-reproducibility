# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Fast offline downsampling analysis for Sentinel-1 per-scene city means."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
PERSCENE_CSV = PROJECT / "data" / "downsampling" / "s1_perscene_city_means_full.csv"
OUT_DIR = PROJECT / "data" / "downsampling"
FIG_DIR = PROJECT / "figures"
VALIDATION_CSV = PROJECT / "data" / "ri_outputs" / "s1_ri_detectability_validation_city_table.csv"

B = 100
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
RNG = np.random.default_rng(20260609)


def load_exports() -> pd.DataFrame:
    if not PERSCENE_CSV.exists():
        raise FileNotFoundError(f"Per-scene city means table not found: {PERSCENE_CSV}")
    out = pd.read_csv(PERSCENE_CSV)
    keep = [
        "city_id",
        "continent",
        "country",
        "area_km2",
        "population_2020",
        "date",
        "orbit",
        "rel_orbit",
        "platform",
        "window",
        "vv_db_mean",
        "vh_db_mean",
        "valid_frac",
    ]
    out = out[keep].dropna(subset=["city_id", "date", "orbit", "window", "vv_db_mean", "vh_db_mean"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["city_id", "date", "orbit", "rel_orbit", "platform", "window"])
    return out


def draw_mean(values: np.ndarray, k: int) -> float:
    idx = RNG.choice(len(values), size=k, replace=False)
    return float(values[idx].mean())


def synthetic_ri_from_k(k: int, both_windows: bool = True) -> float:
    # Deterministic approximation for a thinned endpoint design. Density is the
    # main experimental variable; continuity/gap are represented by expected
    # monthly occupancy under random draws across 12-month and 24-month windows.
    draws = k * (2 if both_windows else 1)
    span_months = 36 if both_windows else 24
    density = min((draws / span_months) / 6.0, 1.0)
    expected_month_fraction = 1.0 - ((span_months - 1) / span_months) ** max(draws, 1)
    continuity = min(expected_month_fraction, 1.0)
    gap_score = min(np.log1p(draws) / np.log1p(32), 1.0)
    orbit_div = 0.5
    return float(np.mean([density, continuity, gap_score, orbit_div]))


def fixed_effect_slope(df: pd.DataFrame, x_col: str, y_col: str = "abs_bias_db") -> dict[str, float]:
    work = df[["city_id", x_col, y_col]].dropna().copy()
    if work["city_id"].nunique() < 3:
        return {"coef": np.nan, "se": np.nan, "t": np.nan, "p_value": np.nan, "n_obs": len(work), "n_cities": work["city_id"].nunique()}
    work["_x"] = work[x_col] - work.groupby("city_id")[x_col].transform("mean")
    work["_y"] = work[y_col] - work.groupby("city_id")[y_col].transform("mean")
    x = work["_x"].to_numpy()
    y = work["_y"].to_numpy()
    denom = float(np.dot(x, x))
    coef = float(np.dot(x, y) / denom) if denom > 0 else np.nan
    resid = y - coef * x
    n = len(work)
    g = work["city_id"].nunique()
    # Cluster-robust SE by city for one regressor after within transform.
    meat = 0.0
    for _, sub in work.assign(_resid=resid).groupby("city_id"):
        sxu = float(np.dot(sub["_x"].to_numpy(), sub["_resid"].to_numpy()))
        meat += sxu * sxu
    var = meat / (denom * denom) if denom > 0 else np.nan
    if g > 1 and n > 1:
        var *= (g / (g - 1)) * ((n - 1) / max(n - 2, 1))
    se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
    t = coef / se if se and se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=max(g - 1, 1))) if np.isfinite(t) else np.nan
    return {"coef": coef, "se": se, "t": float(t), "p_value": p, "n_obs": int(n), "n_cities": int(g)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ps = load_exports()
    validation = pd.read_csv(VALIDATION_CSV)[["city_id", "mean_delta_built_fraction"]].dropna()

    summary_rows = []
    model_rows = []
    city_mean_rows = []
    n_replicate_rows = 0

    for (city_id, continent), city in ps.groupby(["city_id", "continent"], sort=False):
        for orbit in ("ASCENDING", "DESCENDING"):
            orbit_df = city[city["orbit"] == orbit]
            for pol in ("vv_db_mean", "vh_db_mean"):
                s_vals = orbit_df.loc[orbit_df["window"] == "S", pol].dropna().to_numpy(float)
                e_vals = orbit_df.loc[orbit_df["window"] == "E", pol].dropna().to_numpy(float)
                if len(s_vals) < max(K_GRID) or len(e_vals) < max(K_GRID):
                    continue
                ref = float(e_vals.mean() - s_vals.mean())
                for scheme in ("both_thinned", "end_thinned", "start_thinned"):
                    for k in K_GRID:
                        if len(s_vals) < k or len(e_vals) < k:
                            continue
                        deltas = np.empty(B, dtype=float)
                        abs_bias = np.empty(B, dtype=float)
                        for rep in range(B):
                            if scheme == "both_thinned":
                                delta = draw_mean(e_vals, k) - draw_mean(s_vals, k)
                            elif scheme == "end_thinned":
                                delta = draw_mean(e_vals, k) - float(s_vals.mean())
                            else:
                                delta = float(e_vals.mean()) - draw_mean(s_vals, k)
                            deltas[rep] = delta
                            abs_bias[rep] = abs(delta - ref)
                        syn_ri = synthetic_ri_from_k(k, both_windows=(scheme == "both_thinned"))
                        n_replicate_rows += B
                        summary_rows.append(
                            {
                                "scheme": scheme,
                                "pol": pol,
                                "orbit": orbit,
                                "k": k,
                                "city_id": city_id,
                                "continent": continent,
                                "n_replicates": B,
                                "mean_delta_db": float(deltas.mean()),
                                "mean_abs_bias_db": float(abs_bias.mean()),
                                "median_abs_bias_db": float(np.median(abs_bias)),
                                "instability_db": float(deltas.std(ddof=1)),
                                "p90_abs_bias_db": float(np.quantile(abs_bias, 0.9)),
                                "synthetic_ri": syn_ri,
                            }
                        )
                        city_mean_rows.append(
                            {
                                "city_id": city_id,
                                "scheme": scheme,
                                "pol": pol,
                                "orbit": orbit,
                                "k": k,
                                "mean_delta_db": float(deltas.mean()),
                            }
                        )

    city_summary = pd.DataFrame(summary_rows)
    if city_summary.empty:
        raise RuntimeError("No eligible downsampling groups found.")

    dose = (
        city_summary.groupby(["scheme", "pol", "orbit", "k"], as_index=False)
        .agg(
            n_cities=("city_id", "nunique"),
            n_replicates=("n_replicates", "sum"),
            mean_abs_bias_db=("mean_abs_bias_db", "mean"),
            median_abs_bias_db=("median_abs_bias_db", "median"),
            instability_db=("instability_db", "mean"),
            p90_abs_bias_db=("p90_abs_bias_db", "mean"),
            mean_synthetic_ri=("synthetic_ri", "mean"),
        )
    )

    city_delta = pd.DataFrame(city_mean_rows).merge(validation, on="city_id", how="inner")
    agreement_rows = []
    for keys, sub in city_delta.groupby(["scheme", "pol", "orbit", "k"]):
        if len(sub) >= 10:
            r, p = stats.spearmanr(sub["mean_delta_db"], sub["mean_delta_built_fraction"])
        else:
            r, p = np.nan, np.nan
        agreement_rows.append(
            {
                "scheme": keys[0],
                "pol": keys[1],
                "orbit": keys[2],
                "k": keys[3],
                "n_cities_agreement": len(sub),
                "spearman_delta_vs_ob_built": r,
                "spearman_p": p,
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    dose = dose.merge(agreement, on=["scheme", "pol", "orbit", "k"], how="left")

    model_input = city_summary.loc[:, ["city_id", "scheme", "pol", "orbit", "k", "mean_abs_bias_db", "synthetic_ri"]].copy()
    model_input = model_input.rename(columns={"mean_abs_bias_db": "abs_bias_db"})
    model_input["logk"] = np.log(model_input["k"])
    for keys, sub in model_input.groupby(["scheme", "pol", "orbit"]):
        for x_col, label in (("logk", "abs_bias_db ~ log(k) + city FE"), ("synthetic_ri", "abs_bias_db ~ synthetic_ri + city FE")):
            fit = fixed_effect_slope(sub, x_col=x_col)
            fit.update({"scheme": keys[0], "pol": keys[1], "orbit": keys[2], "model": label, "term": x_col})
            model_rows.append(fit)
    models = pd.DataFrame(model_rows)

    city_summary.to_csv(OUT_DIR / "downsample_city_summary_full.csv", index=False)
    dose.to_csv(OUT_DIR / "dose_response_summary_full.csv", index=False)
    agreement.to_csv(OUT_DIR / "downsample_ob_agreement_full.csv", index=False)
    models.to_csv(OUT_DIR / "synthetic_ri_fixedeffect_summary_full.csv", index=False)

    plot_dose(dose)

    qc = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_table": str(PERSCENE_CSV),
        "n_scene_rows": int(len(ps)),
        "n_cities_exports": int(ps["city_id"].nunique()),
        "n_cities_replicates": int(city_summary["city_id"].nunique()),
        "n_virtual_replicate_rows": int(n_replicate_rows),
    }
    (OUT_DIR / "s1_ri_downsampling_run_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


def plot_dose(dose: pd.DataFrame) -> None:
    primary = dose[dose["scheme"] == "both_thinned"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4))
    styles = {
        ("vv_db_mean", "ASCENDING"): ("#2166ac", "VV asc."),
        ("vv_db_mean", "DESCENDING"): ("#67a9cf", "VV desc."),
        ("vh_db_mean", "ASCENDING"): ("#b2182b", "VH asc."),
        ("vh_db_mean", "DESCENDING"): ("#ef8a62", "VH desc."),
    }
    for (pol, orbit), sub in primary.groupby(["pol", "orbit"]):
        sub = sub.sort_values("k")
        color, label = styles[(pol, orbit)]
        axes[0].plot(
            sub["k"],
            sub["mean_abs_bias_db"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=label,
        )
        axes[1].plot(
            sub["k"],
            sub["instability_db"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
        )
        axes[2].plot(
            sub["k"],
            sub["spearman_delta_vs_ob_built"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
        )
    axes[0].set_ylabel("Mean absolute bias (dB)")
    axes[1].set_ylabel("Mean within-city SD (dB)")
    axes[2].set_ylabel("Spearman r")
    for letter, ax in zip("abc", axes):
        ax.set_xlabel("Retained scenes per window (k)")
        ax.set_xticks([5, 10, 15])
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=1.2, pad=0.35)
    figure_path = FIG_DIR / "figure3_downsampling_dose_response.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
