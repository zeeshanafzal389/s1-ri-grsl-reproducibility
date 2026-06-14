"""
Phase C - S1-RI downstream validation against Open Buildings 2.5D structural change.

Tests whether the Sentinel-1 Reliability Index (S1-RI) predicts/moderates the
agreement between Sentinel-1 endpoint backscatter change and independent
Open Buildings structural change and reliability-bin summaries.

Outputs (S1-RI project):
  data/ri_outputs/s1_ri_detectability_validation_city_table.csv
  data/ri_outputs/s1_ri_validation_by_reliability_bin.csv
  data/ri_outputs/s1_ri_moderation_model_summary.csv
  data/ri_outputs/s1_ri_failure_mode_model_summary.csv
  data/ri_outputs/s1_ri_detectability_validation_summary.json
  figures/validation_s1_ri_detectability.png
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1RI = Path(__file__).resolve().parents[1]
ROOT = S1RI.parent
DET = S1RI / "data" / "external" / "sentinel1_open_buildings_detectability_city_table.csv"
OUT = S1RI / "data/ri_outputs"
FIG = S1RI / "figures"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DET)
n_all = len(df)

# ---- Validation sample: cities with S1-RI score ----
val = df[df["s1_ri_score"].notna()].copy()
n_ri = len(val)

S1_IDX = "s1_backscatter_change_index"
OB_IDX = "openbuildings_structural_change_index"
ORBIT_METRICS = [
    "s1_ascending_dvv_db_mean", "s1_ascending_dvh_db_mean",
    "s1_descending_dvv_db_mean", "s1_descending_dvh_db_mean",
]
OB_REF = "mean_delta_built_fraction"

summary = {"n_all_rows": int(n_all), "n_with_s1_ri": int(n_ri)}

def spearman(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())

# =====================================================================
# TEST 1 - RI-bin agreement test
# =====================================================================
bins = ["low", "medium", "high"]
rows = []
for b in bins:
    sub = val[val["s1_reliability_bin"] == b]
    # primary: combined S1 change index vs OB structural change index
    r, p, n = spearman(sub[S1_IDX], sub[OB_IDX])
    # agreement rates
    flag = sub["s1_ob_binary_agreement_flag"]
    agree_rate = float(flag.mean(skipna=True)) if flag.notna().any() else np.nan
    insuff = sub["s1_core_change_metric_unavailable_flag"]
    insuff_rate = float(insuff.mean()) if len(sub) else np.nan
    quad = sub["s1_ob_agreement_quadrant"].value_counts(normalize=True)
    row = {
        "ri_bin": b, "n_cities": len(sub),
        "spearman_s1idx_vs_obidx": r, "spearman_p": p, "n_pairs": n,
        "binary_agreement_rate": agree_rate,
        "insufficient_s1_metric_rate": insuff_rate,
        "false_positive_rate": float(quad.get("false_positive_s1_change_without_ob_growth", 0.0)),
        "false_negative_rate": float(quad.get("false_negative_ob_growth_s1_low", 0.0)),
    }
    # secondary: orbit-specific metrics vs OB built-fraction change (kept separate)
    for m in ORBIT_METRICS:
        rr, pp, nn = spearman(sub[m].abs(), sub[OB_REF].abs())
        row[f"spearman_abs_{m}_vs_obbuilt"] = rr
    rows.append(row)
bin_tbl = pd.DataFrame(rows)
bin_tbl.to_csv(OUT / "s1_ri_validation_by_reliability_bin.csv", index=False)
summary["test1_ri_bin"] = bin_tbl.to_dict(orient="records")

# Continuous correlation across full sample for reference
r_all, p_all, n_all_pairs = spearman(val[S1_IDX], val[OB_IDX])
summary["overall_spearman_s1idx_vs_obidx"] = {"r": r_all, "p": p_all, "n": n_all_pairs}
# Does RI itself correlate with agreement?
r_ri_agree, p_ri_agree, n_ri_agree = spearman(
    val["s1_ri_score"], val["s1_ob_binary_agreement_flag"].astype("float")
)
summary["spearman_ri_vs_binary_agreement"] = {"r": r_ri_agree, "p": p_ri_agree, "n": n_ri_agree}

# =====================================================================
# TEST 2 - Moderation regression (does RI moderate S1 -> OB relationship?)
# =====================================================================
mod = val.dropna(subset=[S1_IDX, OB_IDX, "s1_ri_score", "continent"]).copy()
# standardize predictors for interpretable interaction
for c in [S1_IDX, "s1_ri_score"]:
    mod[c + "_z"] = (mod[c] - mod[c].mean()) / mod[c].std()
mod_formula = f"{OB_IDX} ~ {S1_IDX}_z * s1_ri_score_z + C(continent)"
m2 = smf.ols(mod_formula, data=mod).fit()
mod_rows = []
for name in m2.params.index:
    mod_rows.append({
        "term": name, "coef": float(m2.params[name]),
        "std_err": float(m2.bse[name]), "t": float(m2.tvalues[name]),
        "p_value": float(m2.pvalues[name]),
    })
mod_df = pd.DataFrame(mod_rows)
mod_df.to_csv(OUT / "s1_ri_moderation_model_summary.csv", index=False)
inter_term = f"{S1_IDX}_z:s1_ri_score_z"
summary["test2_moderation"] = {
    "formula": mod_formula, "n": int(m2.nobs), "r_squared": float(m2.rsquared),
    "interaction_term": inter_term,
    "interaction_coef": float(m2.params.get(inter_term, np.nan)),
    "interaction_p": float(m2.pvalues.get(inter_term, np.nan)),
    "s1_main_coef": float(m2.params.get(f"{S1_IDX}_z", np.nan)),
    "s1_main_p": float(m2.pvalues.get(f"{S1_IDX}_z", np.nan)),
}

# =====================================================================
# TEST 3 - Failure-mode model (do RI components predict insufficient metric?)
# =====================================================================
comp_cols = [
    "s1_reliability_score_component_mean_acquisition",
    "s1_reliability_score_component_observation_continuity",
    "s1_reliability_score_component_gap",
    "s1_reliability_score_component_dual_pol",
    "s1_reliability_score_component_orbit_diversity",
    "s1_reliability_score_component_1b_shock_resilience",
]
fm = val.dropna(subset=comp_cols + ["s1_core_change_metric_unavailable_flag"]).copy()
fm["y"] = fm["s1_core_change_metric_unavailable_flag"].astype(int)
Xf = sm.add_constant(fm[comp_cols])
m3 = sm.Logit(fm["y"], Xf).fit(disp=0)
fm_rows = []
for name in m3.params.index:
    fm_rows.append({
        "term": name, "coef": float(m3.params[name]),
        "odds_ratio": float(np.exp(m3.params[name])),
        "std_err": float(m3.bse[name]), "z": float(m3.tvalues[name]),
        "p_value": float(m3.pvalues[name]),
    })
fm_df = pd.DataFrame(fm_rows)
fm_df.to_csv(OUT / "s1_ri_failure_mode_model_summary.csv", index=False)
# simple RI-score-only logistic for an interpretable headline OR
fm["ri_z"] = (fm["s1_ri_score"] - fm["s1_ri_score"].mean()) / fm["s1_ri_score"].std()
m3b = sm.Logit(fm["y"], sm.add_constant(fm[["ri_z"]])).fit(disp=0)
summary["test3_failure_mode"] = {
    "n": int(m3.nobs), "pseudo_r2": float(m3.prsquared),
    "ri_score_only_OR_per_sd": float(np.exp(m3b.params["ri_z"])),
    "ri_score_only_p": float(m3b.pvalues["ri_z"]),
}

# =====================================================================
# TEST 4 - Holdout check (fit on 80pct, evaluate on 20pct)
# =====================================================================
tr = val[(val["s1_analysis_split"] == "analysis_80pct")].dropna(subset=comp_cols + ["s1_core_change_metric_unavailable_flag"]).copy()
te = val[(val["s1_analysis_split"] == "holdout_20pct")].dropna(subset=comp_cols + ["s1_core_change_metric_unavailable_flag"]).copy()
tr["y"] = tr["s1_core_change_metric_unavailable_flag"].astype(int)
te["y"] = te["s1_core_change_metric_unavailable_flag"].astype(int)
mh = sm.Logit(tr["y"], sm.add_constant(tr[comp_cols])).fit(disp=0)
te_pred = mh.predict(sm.add_constant(te[comp_cols], has_constant="add"))
try:
    from sklearn.metrics import roc_auc_score
    auc_tr = roc_auc_score(tr["y"], mh.predict(sm.add_constant(tr[comp_cols])))
    auc_te = roc_auc_score(te["y"], te_pred)
except Exception:
    # manual AUC
    def auc(y, s):
        y = np.asarray(y); s = np.asarray(s)
        pos = s[y == 1]; neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        return float((pos[:, None] > neg[None, :]).mean())
    auc_tr = auc(tr["y"], mh.predict(sm.add_constant(tr[comp_cols])))
    auc_te = auc(te["y"], te_pred)
# also RI-bin agreement reproduced on holdout
ho_bin = []
for b in bins:
    sub = te[te["s1_reliability_bin"] == b]
    ho_bin.append({"ri_bin": b, "n": len(sub),
                   "insufficient_rate": float(sub["y"].mean()) if len(sub) else np.nan})
summary["test4_holdout"] = {
    "n_train": int(len(tr)), "n_test": int(len(te)),
    "auc_train": float(auc_tr), "auc_holdout": float(auc_te),
    "holdout_bin_insufficient_rate": ho_bin,
}

# =====================================================================
# Save validation city table (overlap sample, key columns)
# =====================================================================
keep = ["city_id", "city_name", "country", "continent", "population_2020",
        "dominant_mode", "archetype", "s1_ri_score", "s1_reliability_bin",
        S1_IDX, OB_IDX, "mean_delta_built_fraction", "total_delta_volume_proxy",
        "s1_ob_agreement_quadrant", "s1_ob_binary_agreement_flag",
        "s1_core_change_metric_unavailable_flag", "s1_analysis_split"] + comp_cols
val[keep].to_csv(OUT / "s1_ri_detectability_validation_city_table.csv", index=False)

with open(OUT / "s1_ri_detectability_validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# =====================================================================
# Validation diagnostic panel
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
order = ["low", "medium", "high"]
colors = ["#c0392b", "#e6a817", "#1f6f8b"]

# (a) Spearman S1 vs OB by RI bin
rs = [bin_tbl.loc[bin_tbl.ri_bin == b, "spearman_s1idx_vs_obidx"].values[0] for b in order]
ax[0].bar(order, rs, color=colors)
ax[0].set_title("(a) S1-OB change agreement\nby reliability tier")
ax[0].set_ylabel("Spearman r (S1 vs OpenBuildings)")
ax[0].axhline(0, color="k", lw=0.6)
for i, v in enumerate(rs):
    ax[0].text(i, v + (0.005 if v >= 0 else -0.02), f"{v:.3f}", ha="center", fontsize=9)

# (b) insufficient-metric rate by RI bin
ins = [bin_tbl.loc[bin_tbl.ri_bin == b, "insufficient_s1_metric_rate"].values[0] for b in order]
ax[1].bar(order, [v * 100 for v in ins], color=colors)
ax[1].set_title("(b) Insufficient S1 change metric\nby reliability tier")
ax[1].set_ylabel("% cities with insufficient S1 metric")
for i, v in enumerate(ins):
    ax[1].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=9)

# (c) binary agreement rate by RI bin
agr = [bin_tbl.loc[bin_tbl.ri_bin == b, "binary_agreement_rate"].values[0] for b in order]
ax[2].bar(order, [v * 100 for v in agr], color=colors)
ax[2].set_title("(c) S1-OB agreement rate\nby reliability tier")
ax[2].set_ylabel("% cities in agreement")
for i, v in enumerate(agr):
    ax[2].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=9)

for a in ax:
    a.set_xlabel("S1-RI tier")
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "validation_s1_ri_detectability.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "validation_s1_ri_detectability.pdf", bbox_inches="tight")

print(json.dumps(summary, indent=2))
