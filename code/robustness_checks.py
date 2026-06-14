"""Robustness checks for the S1-RI validation."""
import numpy as np, pandas as pd, json
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
DET = PROJECT / "data" / "external" / "sentinel1_open_buildings_detectability_city_table.csv"
df = pd.read_csv(DET)
v = df[df.s1_ri_score.notna()].copy()
v["y"] = v.s1_core_change_metric_unavailable_flag.astype(int)
out = {}

# ---------------------------------------------------------------
# 1. Is the insufficiency flag mechanically the endpoint scene counts?
# ---------------------------------------------------------------
endp = ["s1_ascending_obs_count_2017_mean","s1_ascending_obs_count_2023_mean",
        "s1_descending_obs_count_2017_mean","s1_descending_obs_count_2023_mean"]
v["endpoint_total_obs"] = v[endp].sum(axis=1)
g = v.groupby("y")["endpoint_total_obs"].describe()[["mean","50%","min","max"]]
out["endpoint_obs_by_flag"] = g.round(2).to_dict()
# how separable is the flag from endpoint counts alone?
auc_endp_only = roc_auc_score(v.y, -v.endpoint_total_obs)  # fewer obs -> more likely unavailable
out["auc_flag_from_endpoint_obs_only"] = round(float(auc_endp_only), 4)

# ---------------------------------------------------------------
# 2. INCREMENTAL VALUE: does S1-RI predict the flag BEYOND endpoint counts + size + region?
# ---------------------------------------------------------------
v["log_pop"] = np.log10(v.population_2020.clip(lower=1))
v["log_area"] = np.log10(v.area_km2.clip(lower=0.01))
base_num = v[endp + ["log_pop","log_area"]].copy()
cont = pd.get_dummies(v.continent, prefix="c", drop_first=True).astype(float)
# standardize numeric
for c in base_num.columns:
    base_num[c] = (base_num[c]-base_num[c].mean())/base_num[c].std()
ri = (v.s1_ri_score - v.s1_ri_score.mean())/v.s1_ri_score.std()

def fit_auc(X, y):
    X = sm.add_constant(X.astype(float), has_constant="add")
    m = sm.Logit(y, X).fit(disp=0)
    return m, roc_auc_score(y, m.predict(X))

X0 = pd.concat([base_num, cont], axis=1)              # endpoint counts + size + region
X1 = pd.concat([base_num, cont, ri.rename("s1_ri_z")], axis=1)  # + S1-RI
m0, auc0 = fit_auc(X0, v.y)
m1, auc1 = fit_auc(X1, v.y)
lr_stat = 2*(m1.llf - m0.llf); lr_p = stats.chi2.sf(lr_stat, 1)
out["incremental_value"] = {
    "auc_base_endpoint+size+region": round(auc0,4),
    "auc_base+S1RI": round(auc1,4),
    "auc_gain": round(auc1-auc0,4),
    "LR_chi2_for_S1RI": round(float(lr_stat),2), "LR_p": float(lr_p),
    "S1RI_coef_in_full": round(float(m1.params.get("s1_ri_z",np.nan)),3),
    "S1RI_p_in_full": float(m1.pvalues.get("s1_ri_z",np.nan)),
}

# Same test but WITHOUT endpoint counts (only structural covariates) to bound the other side
X0b = pd.concat([v[["log_pop","log_area"]].apply(lambda s:(s-s.mean())/s.std()), cont], axis=1)
X1b = pd.concat([X0b, ri.rename("s1_ri_z")], axis=1)
m0b, auc0b = fit_auc(X0b, v.y); m1b, auc1b = fit_auc(X1b, v.y)
out["incremental_value_no_endpoint_controls"] = {
    "auc_size+region": round(auc0b,4), "auc_+S1RI": round(auc1b,4), "auc_gain": round(auc1b-auc0b,4)}

# ---------------------------------------------------------------
# 3. Trend test: insufficiency monotone across RI tiers (Cochran-Armitage)
# ---------------------------------------------------------------
order = ["low","medium","high"]
tab = []
for b in order:
    s = v[v.s1_reliability_bin==b]
    tab.append([int(s.y.sum()), int((1-s.y).sum())])
tab = np.array(tab)  # rows tiers, cols [insuff, ok]
# Cochran-Armitage trend
scores = np.array([0,1,2])
n = tab.sum(); row = tab.sum(1); col = tab.sum(0)
p = tab[:,0]
T = np.sum(scores*(tab[:,0]*row.sum()/1))  # use standard CA formula below
# standard CA
N = tab.sum(); Rk = tab.sum(1); Ck = tab.sum(0)
p_bar = Ck[0]/N
num = np.sum(scores*(tab[:,0] - Rk*p_bar))
s_bar = np.sum(Rk*scores)/N
var = p_bar*(1-p_bar)*(np.sum(Rk*scores**2) - N*s_bar**2)
z = num/np.sqrt(var); ca_p = 2*stats.norm.sf(abs(z))
out["trend_test"] = {"tiers_insuff_rate":[round(tab[i,0]/Rk[i],4) for i in range(3)],
                     "cochran_armitage_z": round(float(z),3), "p": float(ca_p)}

# ---------------------------------------------------------------
# 4. Agreement-fidelity null: check it isn't an abs-transform artifact
# ---------------------------------------------------------------
sub = v.dropna(subset=["s1_backscatter_change_index","openbuildings_structural_change_index"])
r_abs,_ = stats.spearmanr(sub.s1_backscatter_change_index, sub.openbuildings_structural_change_index)
# raw directional: ascending dVV vs delta built fraction (both signed)
sub2 = v.dropna(subset=["s1_ascending_dvv_db_mean","mean_delta_built_fraction"])
r_dir,_ = stats.spearmanr(sub2.s1_ascending_dvv_db_mean, sub2.mean_delta_built_fraction)
# restrict to genuine growth cities (OB built fraction increase) - does S1 track it?
grow = sub2[sub2.mean_delta_built_fraction>sub2.mean_delta_built_fraction.quantile(0.75)]
r_grow,_ = stats.spearmanr(grow.s1_ascending_dvv_db_mean.abs(), grow.mean_delta_built_fraction.abs())
out["agreement_null_robustness"] = {"spearman_abs_indices": round(float(r_abs),4),
    "spearman_directional_dvv_vs_built": round(float(r_dir),4),
    "spearman_growth_quartile": round(float(r_grow),4), "n_growth": int(len(grow))}

# ---------------------------------------------------------------
# 5. Outlier audit
# ---------------------------------------------------------------
ri_csv = PROJECT / "data" / "ri_outputs" / "s1_city_reliability_index.csv"
d = pd.read_csv(ri_csv)
lowri = d[d.s1_ri_score<0.1]
biggap = d[d.longest_observation_gap_days>1500]
out["outlier_audit"] = {
    "n_ri_below_0.1": int(len(lowri)),
    "ri_below_0.1_median_pop": float(lowri.population_2020.median()) if len(lowri) else None,
    "ri_below_0.1_median_area_km2": float(lowri.area_km2.median()) if len(lowri) else None,
    "ri_below_0.1_median_mean_monthly_acq": float(lowri.mean_monthly_acquisitions_2017_2026.median()) if len(lowri) else None,
    "n_gap_over_1500d": int(len(biggap)),
    "gap_over_1500d_median_pop": float(biggap.population_2020.median()) if len(biggap) else None,
    "gap_over_1500d_median_area_km2": float(biggap.area_km2.median()) if len(biggap) else None,
    "all_cities_median_pop": float(d.population_2020.median()),
    "all_cities_median_area_km2": float(d.area_km2.median()),
}

print(json.dumps(out, indent=2, default=str))
Path(PROJECT / "data" / "ri_outputs" / "s1_ri_robustness_checks.json").write_text(json.dumps(out, indent=2, default=str))
