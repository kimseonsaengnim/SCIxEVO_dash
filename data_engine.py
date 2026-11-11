"""
Data & Statistical Engine for the Psychometric & Epistemic BI Dashboard.
Handles read-only CSV ingestion (explicitly dropping Religion), dynamic TG.csv & TGcodes.csv
parsing, cross-validation, multi-mode aspect scoring (Mean 1-5, Raw Sum, Z-Score), and
parametric/non-parametric psychometric statistical modeling.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    sm = None
    smf = None
    HAS_STATSMODELS = False

try:
    import streamlit as st
    cache_decorator = st.cache_data
except Exception:
    def cache_decorator(func):
        return func

from config import (
    ACTIVE_DEMOGRAPHICS,
    ALL_EVOLUTION_ITEMS,
    ALL_SCIENCE_ITEMS,
    EVOLUTION_CRITERIA,
    EVO_ALL_ASPECTS,
    EVO_ASPECT_ITEMS,
    EVO_CRD_ITEMS,
    EVO_FCT_ITEMS,
    EVO_OVERLAPPING_ASPECTS,
    EVO_SPLIT_SUBASPECT_ITEMS,
    SCIENCE_AFFECTIVE_ITEMS,
    SCIENCE_ALL_ASPECTS,
    SCIENCE_ASPECT_ITEMS,
    SCIENCE_COGNITIVE_ASPECTS,
    SCIENCE_COGNITIVE_ITEMS,
    SCIENCE_PREDICTORS,
    load_tg_codes,
    resolve_input_file,
)


# ==============================================================================
# 1. CACHED DATA LOADING & DYNAMIC PARSING
# ==============================================================================
@cache_decorator
def load_and_validate_metadata() -> Dict[str, Any]:
    """
    Dynamically load and parse TG.csv and TGcodes.csv, cross-validating against
    theoretical construct and aspect mappings.
    """
    tg_path = resolve_input_file("TG.csv")
    tg_df = pd.read_csv(tg_path)
    tg_df.columns = [c.strip() for c in tg_df.columns]
    for col in ["Item#", "prj.3.0 Construct", "prj.3.0 Aspect"]:
        tg_df[col] = tg_df[col].astype(str).str.strip()

    codes_df, code_to_meaning, code_to_label = load_tg_codes()

    # Dynamic extraction from TG.csv
    dynamic_sci_aspects: Dict[str, List[str]] = {}
    dynamic_evo_aspects: Dict[str, List[str]] = {}
    dynamic_fct_items: List[str] = []
    dynamic_crd_items: List[str] = []
    item_meta: Dict[str, Dict[str, str]] = {}

    for _, row in tg_df.iterrows():
        item_id = row["Item#"]
        construct = row["prj.3.0 Construct"]
        aspect = row["prj.3.0 Aspect"]
        item_meta[item_id] = {
            "key": str(row["key"]),
            "construct": construct,
            "construct_meaning": code_to_meaning.get(construct, construct),
            "construct_label": code_to_label.get(construct, construct),
            "aspect": aspect,
            "aspect_meaning": code_to_meaning.get(aspect, aspect),
            "aspect_label": code_to_label.get(aspect, aspect),
        }
        if item_id.startswith("S"):
            dynamic_sci_aspects.setdefault(aspect, []).append(item_id)
        elif item_id.startswith("E"):
            dynamic_evo_aspects.setdefault(aspect, []).append(item_id)
            if construct == "FCT":
                dynamic_fct_items.append(item_id)
            elif construct == "CRD":
                dynamic_crd_items.append(item_id)

    # Cross-validate against theoretical ground truth
    for asp, expected_items in SCIENCE_ASPECT_ITEMS.items():
        if sorted(dynamic_sci_aspects.get(asp, [])) != sorted(expected_items):
            raise ValueError(f"Science aspect mismatch for {asp}: {dynamic_sci_aspects.get(asp)} vs {expected_items}")

    for asp, expected_items in EVO_ASPECT_ITEMS.items():
        if sorted(dynamic_evo_aspects.get(asp, [])) != sorted(expected_items):
            raise ValueError(f"Evolution aspect mismatch for {asp}: {dynamic_evo_aspects.get(asp)} vs {expected_items}")

    if sorted(dynamic_fct_items) != sorted(EVO_FCT_ITEMS):
        raise ValueError("FCT items mismatch between TG.csv and theoretical specification.")
    if sorted(dynamic_crd_items) != sorted(EVO_CRD_ITEMS):
        raise ValueError("CRD items mismatch between TG.csv and theoretical specification.")

    return {
        "tg_df": tg_df,
        "codes_df": codes_df,
        "code_to_meaning": code_to_meaning,
        "code_to_label": code_to_label,
        "item_meta": item_meta,
    }


@cache_decorator
def load_survey_data() -> pd.DataFrame:
    """
    Load 2026data_rev.csv (N=121), strictly drop the 'Religion' column on load,
    and compute raw sums and 1-5 item means for all constructs, aspects, and
    epistemic split sub-aspects.
    """
    data_path = resolve_input_file("2026data_rev.csv")
    df = pd.read_csv(data_path)

    # STRICT GUARDRAIL: Completely exclude and drop 'Religion' column upon load
    if "Religion" in df.columns:
        df = df.drop(columns=["Religion"])
    for forbidden_col in [c for c in df.columns if "religion" in c.lower() and c != "Religiosity"]:
        df = df.drop(columns=[forbidden_col])

    # Clean string demographic columns
    df["ID"] = df["ID"].astype(str).str.strip()
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["Religiosity"] = df["Religiosity"].astype(str).str.strip()
    df["Age"] = df["Age"].astype(int)

    # Compute FCT and CRD raw composite sums & verify pre-computed composites
    df["FCT_sum"] = df[EVO_FCT_ITEMS].sum(axis=1)
    df["CRD_sum"] = df[EVO_CRD_ITEMS].sum(axis=1)
    df["FCT"] = df["FCT_sum"]
    df["CRD"] = df["CRD_sum"]

    # Store explicit _sum and _mean columns for primary composites
    df["SCI_sum"] = df["SCI"]
    df["SCI_mean"] = df[ALL_SCIENCE_ITEMS].mean(axis=1)

    df["SKSI_sum"] = df["SKSI"]
    df["SKSI_mean"] = df[SCIENCE_COGNITIVE_ITEMS].mean(axis=1)

    df["ATS_sum"] = df["ATS"]
    df["ATS_mean"] = df[SCIENCE_AFFECTIVE_ITEMS].mean(axis=1)

    df["EVO_sum"] = df["EVO"]
    df["EVO_mean"] = df[ALL_EVOLUTION_ITEMS].mean(axis=1)

    df["FCT_mean"] = df[EVO_FCT_ITEMS].mean(axis=1)
    df["CRD_mean"] = df[EVO_CRD_ITEMS].mean(axis=1)

    # Compute 14 Science Aspect scores (Raw Sum and 1-5 Mean)
    for asp, items in SCIENCE_ASPECT_ITEMS.items():
        df[f"{asp}_sum"] = df[items].sum(axis=1)
        df[f"{asp}_mean"] = df[items].mean(axis=1)

    # Compute 6 Evolution Aspect scores (Raw Sum and 1-5 Mean)
    for asp, items in EVO_ASPECT_ITEMS.items():
        df[f"{asp}_sum"] = df[items].sum(axis=1)
        df[f"{asp}_mean"] = df[items].mean(axis=1)

    # Compute 10 Evolution Epistemic Split Sub-Aspect scores (Raw Sum and 1-5 Mean)
    for sub_asp, items in EVO_SPLIT_SUBASPECT_ITEMS.items():
        df[f"{sub_asp}_sum"] = df[items].sum(axis=1)
        df[f"{sub_asp}_mean"] = df[items].mean(axis=1)

    return df


# ==============================================================================
# 2. FILTERING & MULTI-MODE SCORING ENGINE
# ==============================================================================
def filter_cohort(
    df: pd.DataFrame,
    genders: List[str] | None = None,
    religiosities: List[str] | None = None,
    age_range: Tuple[int, int] = (14, 18),
) -> pd.DataFrame:
    """Filter respondent dataframe by active demographic slicers (Gender, Religiosity, Age)."""
    out = df.copy()
    if genders is not None:
        out = out[out["Gender"].isin(genders)]
    if religiosities is not None:
        out = out[out["Religiosity"].isin(religiosities)]
    if age_range is not None:
        out = out[(out["Age"] >= age_range[0]) & (out["Age"] <= age_range[1])]
    return out.reset_index(drop=True)


def get_scored_dataframe(df: pd.DataFrame, scoring_mode: str = "Mean (1-5)") -> pd.DataFrame:
    """
    Return a dataframe where all primary composites, 14 Science aspects,
    6 Evolution aspects, and 10 Epistemic sub-aspects are populated according
    to the user-selected aggregation mode:
      - 'Mean (1-5)': Item mean on the 1-5 Likert scale
      - 'Raw Sum': Direct summation of items within each construct/aspect
      - 'Z-Score': Standardized (Mean=0, SD=1) across the active sample
    """
    scored = df.copy()
    all_keys = (
        SCIENCE_PREDICTORS
        + EVOLUTION_CRITERIA
        + SCIENCE_ALL_ASPECTS
        + EVO_ALL_ASPECTS
        + list(EVO_SPLIT_SUBASPECT_ITEMS.keys())
    )

    for key in all_keys:
        if scoring_mode == "Mean (1-5)":
            scored[key] = scored[f"{key}_mean"]
        elif scoring_mode == "Raw Sum":
            scored[key] = scored[f"{key}_sum"]
        elif scoring_mode in ("Z-Score", "Z-Score Standardized"):
            vals = scored[f"{key}_mean"]
            sd = vals.std(ddof=1)
            if pd.isna(sd) or sd == 0:
                scored[key] = 0.0
            else:
                scored[key] = (vals - vals.mean()) / sd
        else:
            scored[key] = scored[f"{key}_mean"]

    return scored


# ==============================================================================
# 3. STATISTICAL HELPER FUNCTIONS (SCIPY & STATSMODELS)
# ==============================================================================
def format_sig_stars(p_val: float) -> str:
    """Return significance stars for two-tailed p-values."""
    if pd.isna(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return ""


def compute_bivariate_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: str = "Pearson (r)",
) -> Dict[str, Any]:
    """Compute Pearson (r) or Spearman (rho) correlation with exact two-tailed p-value."""
    sub = df[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 3 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return {"r": np.nan, "p": np.nan, "stars": "", "n": n, "method_symbol": "r"}

    if "Spearman" in method or method.lower() == "spearman":
        stat, p_val = stats.spearmanr(sub[x_col], sub[y_col])
        sym = "ρ"
    else:
        stat, p_val = stats.pearsonr(sub[x_col], sub[y_col])
        sym = "r"

    return {
        "r": float(stat),
        "p": float(p_val),
        "stars": format_sig_stars(float(p_val)),
        "n": n,
        "method_symbol": sym,
    }


def compute_correlation_matrix(
    df: pd.DataFrame,
    row_cols: List[str],
    col_cols: List[str],
    method: str = "Pearson (r)",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute rectangular correlation matrix, p-value matrix, and annotated text matrix
    (e.g., '0.45**') between row_cols and col_cols.
    """
    r_mat = pd.DataFrame(index=row_cols, columns=col_cols, dtype=float)
    p_mat = pd.DataFrame(index=row_cols, columns=col_cols, dtype=float)
    annot_mat = pd.DataFrame(index=row_cols, columns=col_cols, dtype=str)

    for r_col in row_cols:
        for c_col in col_cols:
            res = compute_bivariate_correlation(df, r_col, c_col, method=method)
            r_val, p_val, stars = res["r"], res["p"], res["stars"]
            r_mat.loc[r_col, c_col] = r_val
            p_mat.loc[r_col, c_col] = p_val
            if pd.isna(r_val):
                annot_mat.loc[r_col, c_col] = "N/A"
            else:
                annot_mat.loc[r_col, c_col] = f"{r_val:.2f}{stars}"

    return r_mat, p_mat, annot_mat


def _matrix_ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Exact ordinary least squares computation via numpy/scipy matching statsmodels.OLS."""
    n, p = X.shape
    df_resid = n - p
    df_model = p - 1
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ssr / sst) if sst > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else np.nan
    mse = ssr / df_resid if df_resid > 0 else np.nan
    cov_beta = mse * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    t_vals = np.where(se > 0, beta / se, 0.0)
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=df_resid) if df_resid > 0 else np.full(p, np.nan)

    msm = (sst - ssr) / df_model if df_model > 0 else np.nan
    f_stat = msm / mse if (mse and mse > 0) else np.nan
    f_pval = float(stats.f.sf(f_stat, df_model, df_resid)) if not pd.isna(f_stat) else np.nan

    return {
        "beta": beta,
        "se": se,
        "t_vals": t_vals,
        "p_vals": p_vals,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": float(f_stat),
        "f_pval": f_pval,
        "cov_beta": cov_beta,
        "df_resid": df_resid,
    }


def fit_ols_regression(df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
    """
    Fit OLS linear regression Y ~ X using statsmodels (or exact scipy/numpy OLS engine),
    returning slope (beta), intercept, R^2, p-value, standard error, and 95% CI band grid.
    """
    sub = df[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 3 or sub[x_col].nunique() < 2:
        return {
            "beta": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "p": np.nan,
            "f_stat": np.nan,
            "se": np.nan,
            "n": n,
            "x_grid": np.array([]),
            "y_hat": np.array([]),
            "ci_low": np.array([]),
            "ci_high": np.array([]),
        }

    x_arr = sub[x_col].astype(float).to_numpy()
    y_arr = sub[y_col].astype(float).to_numpy()
    x_grid = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)

    if HAS_STATSMODELS and sm is not None:
        X = sm.add_constant(sub[x_col].astype(float))
        model = sm.OLS(sub[y_col].astype(float), X).fit()
        X_pred = sm.add_constant(x_grid, has_constant="add")
        pred_summary = model.get_prediction(X_pred).summary_frame(alpha=0.05)
        return {
            "beta": float(model.params[x_col]),
            "intercept": float(model.params["const"]),
            "r2": float(model.rsquared),
            "p": float(model.pvalues[x_col]),
            "f_stat": float(model.fvalue) if model.fvalue is not None else np.nan,
            "se": float(model.bse[x_col]),
            "n": n,
            "x_grid": x_grid,
            "y_hat": pred_summary["mean"].to_numpy(),
            "ci_low": pred_summary["mean_ci_lower"].to_numpy(),
            "ci_high": pred_summary["mean_ci_upper"].to_numpy(),
        }

    X_mat = np.column_stack([np.ones(n), x_arr])
    res = _matrix_ols_fit(X_mat, y_arr)
    X_pred = np.column_stack([np.ones(len(x_grid)), x_grid])
    y_hat = X_pred @ res["beta"]
    pred_se = np.sqrt(np.sum((X_pred @ res["cov_beta"]) * X_pred, axis=1))
    t_crit = float(stats.t.ppf(0.975, df=res["df_resid"]))

    return {
        "beta": float(res["beta"][1]),
        "intercept": float(res["beta"][0]),
        "r2": float(res["r2"]),
        "p": float(res["p_vals"][1]),
        "f_stat": float(res["f_stat"]),
        "se": float(res["se"][1]),
        "n": n,
        "x_grid": x_grid,
        "y_hat": y_hat,
        "ci_low": y_hat - t_crit * pred_se,
        "ci_high": y_hat + t_crit * pred_se,
    }


def fit_moderation_ols(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    moderator: str,
) -> Dict[str, Any]:
    """
    Fit OLS moderation model Y ~ X * Moderator using statsmodels (with exact scipy/numpy fallback).
    Supports categorical moderators ('Gender', 'Religiosity') and continuous/ordinal ('Age').
    Returns overall model stats, interaction term beta/p-value, and subgroup slope breakdowns.
    """
    if moderator not in ACTIVE_DEMOGRAPHICS:
        raise ValueError(f"Unsupported demographic moderator: {moderator}")

    sub = df[[x_col, y_col, moderator]].dropna().copy()
    n = len(sub)
    if n < 6 or sub[x_col].nunique() < 2 or sub[moderator].nunique() < 2:
        return {
            "valid": False,
            "reason": "Insufficient subgroup variance in active filtered sample.",
            "n": n,
            "subgroup_slopes": [],
        }

    sub["X_val"] = sub[x_col].astype(float)
    sub["Y_val"] = sub[y_col].astype(float)

    if moderator in ("Gender", "Religiosity"):
        formula = f"Y_val ~ X_val * C({moderator})"
    else:
        formula = f"Y_val ~ X_val * {moderator}"

    if HAS_STATSMODELS and smf is not None:
        model = smf.ols(formula, data=sub).fit()
        interaction_terms = [t for t in model.params.index if ":" in t]
        if interaction_terms:
            int_term = interaction_terms[0]
            beta_int = float(model.params[int_term])
            p_int = float(model.pvalues[int_term])
            t_int = float(model.tvalues[int_term])
            se_int = float(model.bse[int_term])
        else:
            int_term, beta_int, p_int, t_int, se_int = "N/A", np.nan, np.nan, np.nan, np.nan
        r2 = float(model.rsquared)
        adj_r2 = float(model.rsquared_adj)
        f_stat = float(model.fvalue) if model.fvalue is not None else np.nan
        f_pval = float(model.f_pvalue) if model.f_pvalue is not None else np.nan
    else:
        x_arr = sub["X_val"].to_numpy()
        y_arr = sub["Y_val"].to_numpy()
        if moderator in ("Gender", "Religiosity"):
            levels = sorted(sub[moderator].unique())
            mod_arr = (sub[moderator] == levels[1]).astype(float).to_numpy()
            int_term = f"X_val:C({moderator})[T.{levels[1]}]"
        else:
            mod_arr = sub[moderator].astype(float).to_numpy()
            int_term = f"X_val:{moderator}"

        X_mat = np.column_stack([np.ones(n), x_arr, mod_arr, x_arr * mod_arr])
        res = _matrix_ols_fit(X_mat, y_arr)
        beta_int = float(res["beta"][3])
        se_int = float(res["se"][3])
        t_int = float(res["t_vals"][3])
        p_int = float(res["p_vals"][3])
        r2 = float(res["r2"])
        adj_r2 = float(res["adj_r2"])
        f_stat = float(res["f_stat"])
        f_pval = float(res["f_pval"])

    # Compute per-level OLS slopes
    subgroup_slopes: List[Dict[str, Any]] = []
    for level, grp in sub.groupby(moderator):
        ols_res = fit_ols_regression(grp, x_col, y_col)
        corr_res = compute_bivariate_correlation(grp, x_col, y_col, method="Pearson (r)")
        subgroup_slopes.append({
            "level": str(level),
            "n": len(grp),
            "beta": ols_res["beta"],
            "intercept": ols_res["intercept"],
            "r": corr_res["r"],
            "r2": ols_res["r2"],
            "p": ols_res["p"],
            "se": ols_res["se"],
            "x_grid": ols_res["x_grid"],
            "y_hat": ols_res["y_hat"],
            "ci_low": ols_res["ci_low"],
            "ci_high": ols_res["ci_high"],
        })

    return {
        "valid": True,
        "moderator": moderator,
        "formula": formula,
        "n": n,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_pval": f_pval,
        "interaction_term": int_term,
        "beta_interaction": beta_int,
        "se_interaction": se_int,
        "t_interaction": t_int,
        "p_interaction": p_int,
        "stars_interaction": format_sig_stars(p_int),
        "subgroup_slopes": subgroup_slopes,
    }


def compute_paired_epistemic_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare normalized FCT (Fact, 1-5 mean) vs. CRD (Credibility, 1-5 mean) within
    respondents using both Paired t-test and Wilcoxon signed-rank test.
    """
    sub = df[["FCT_mean", "CRD_mean"]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "n": n,
            "fct_mean": np.nan,
            "fct_sd": np.nan,
            "crd_mean": np.nan,
            "crd_sd": np.nan,
            "diff_mean": np.nan,
            "diff_sd": np.nan,
            "t_stat": np.nan,
            "t_pval": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_pval": np.nan,
            "cohens_dz": np.nan,
        }

    fct = sub["FCT_mean"].to_numpy()
    crd = sub["CRD_mean"].to_numpy()
    diff = fct - crd

    t_res = stats.ttest_rel(fct, crd)
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) >= 3:
        w_res = stats.wilcoxon(non_zero_diff)
        w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
    else:
        w_stat, w_pval = np.nan, np.nan

    diff_sd = float(np.std(diff, ddof=1))
    cohens_dz = float(np.mean(diff) / diff_sd) if diff_sd > 0 else 0.0

    return {
        "n": n,
        "fct_mean": float(np.mean(fct)),
        "fct_sd": float(np.std(fct, ddof=1)),
        "crd_mean": float(np.mean(crd)),
        "crd_sd": float(np.std(crd, ddof=1)),
        "diff_mean": float(np.mean(diff)),
        "diff_sd": diff_sd,
        "t_stat": float(t_res.statistic),
        "t_pval": float(t_res.pvalue),
        "t_stars": format_sig_stars(float(t_res.pvalue)),
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval,
        "cohens_dz": cohens_dz,
    }


def compute_epistemic_dumbbell_table(
    df: pd.DataFrame,
    code_to_label: Dict[str, str],
) -> pd.DataFrame:
    """
    Build summary table for the Epistemic Divergence Dumbbell Plot across the 4
    overlapping EVO aspects (val, pro, evi, bel) plus Overall (FCT vs. CRD).
    """
    rows: List[Dict[str, Any]] = []
    for asp in EVO_OVERLAPPING_ASPECTS:
        fct_col = f"{asp}_FCT_mean"
        crd_col = f"{asp}_CRD_mean"
        sub = df[[fct_col, crd_col]].dropna()
        if len(sub) >= 3:
            t_res = stats.ttest_rel(sub[fct_col], sub[crd_col])
            t_stat, p_val = float(t_res.statistic), float(t_res.pvalue)
        else:
            t_stat, p_val = np.nan, np.nan
        fct_m = float(sub[fct_col].mean()) if len(sub) else np.nan
        crd_m = float(sub[crd_col].mean()) if len(sub) else np.nan
        gap = fct_m - crd_m if not (pd.isna(fct_m) or pd.isna(crd_m)) else np.nan
        rows.append({
            "aspect_code": asp,
            "aspect_label": code_to_label.get(asp, asp),
            "fct_items": ", ".join(EVO_SPLIT_SUBASPECT_ITEMS[f"{asp}_FCT"]),
            "crd_items": ", ".join(EVO_SPLIT_SUBASPECT_ITEMS[f"{asp}_CRD"]),
            "fct_mean": fct_m,
            "crd_mean": crd_m,
            "gap_fct_minus_crd": gap,
            "abs_gap": abs(gap) if not pd.isna(gap) else np.nan,
            "t_stat": t_stat,
            "p_val": p_val,
            "stars": format_sig_stars(p_val),
        })

    # Add Overall EVO row
    overall = compute_paired_epistemic_test(df)
    rows.append({
        "aspect_code": "OVERALL",
        "aspect_label": "Overall Evolution (FCT vs. CRD)",
        "fct_items": "10 FCT Items",
        "crd_items": "10 CRD Items",
        "fct_mean": overall["fct_mean"],
        "crd_mean": overall["crd_mean"],
        "gap_fct_minus_crd": overall["diff_mean"],
        "abs_gap": abs(overall["diff_mean"]) if not pd.isna(overall["diff_mean"]) else np.nan,
        "t_stat": overall["t_stat"],
        "p_val": overall["t_pval"],
        "stars": overall.get("t_stars", ""),
    })

    return pd.DataFrame(rows)


def compute_likert_distribution_table(
    df: pd.DataFrame,
    item_meta: Dict[str, Dict[str, str]],
    construct_filter: List[str] | None = None,
    sort_by: str = "Aspect Meaning",
) -> pd.DataFrame:
    """
    Compute 1-5 Likert percentage and count distributions across all 48 items
    (28 Science + 20 Evolution), filterable by Construct (SCI, ATS, FCT, CRD).
    """
    all_items = ALL_SCIENCE_ITEMS + ALL_EVOLUTION_ITEMS
    rows: List[Dict[str, Any]] = []

    for item in all_items:
        meta = item_meta.get(item, {})
        c_code = meta.get("construct", "")
        if construct_filter and c_code not in construct_filter:
            continue

        series = df[item].dropna().astype(int)
        n_valid = len(series)
        counts = {k: int((series == k).sum()) for k in range(1, 6)}
        pcts = {k: (counts[k] / n_valid * 100.0 if n_valid > 0 else 0.0) for k in range(1, 6)}

        neg_pct = pcts[1] + pcts[2]
        pos_pct = pcts[4] + pcts[5]

        rows.append({
            "item": item,
            "key_num": int(meta.get("key", 0)),
            "construct": c_code,
            "construct_label": meta.get("construct_label", c_code),
            "aspect": meta.get("aspect", ""),
            "aspect_meaning": meta.get("aspect_meaning", ""),
            "aspect_label": meta.get("aspect_label", ""),
            "display_label": f"{item} | {meta.get('aspect_meaning', '')} ({meta.get('aspect', '')})",
            "mean": float(series.mean()) if n_valid > 0 else np.nan,
            "sd": float(series.std(ddof=1)) if n_valid > 1 else np.nan,
            "n": n_valid,
            "count_1": counts[1],
            "count_2": counts[2],
            "count_3": counts[3],
            "count_4": counts[4],
            "count_5": counts[5],
            "pct_1": pcts[1],
            "pct_2": pcts[2],
            "pct_3": pcts[3],
            "pct_4": pcts[4],
            "pct_5": pcts[5],
            "disagree_pct": neg_pct,
            "agree_pct": pos_pct,
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    if sort_by == "Mean Endorsement (High to Low)":
        out_df = out_df.sort_values(by=["mean", "key_num"], ascending=[False, True])
    elif sort_by == "Mean Endorsement (Low to High)":
        out_df = out_df.sort_values(by=["mean", "key_num"], ascending=[True, True])
    elif sort_by == "Agreement % (4+5)":
        out_df = out_df.sort_values(by=["agree_pct", "mean"], ascending=[False, False])
    elif sort_by == "Item Number (S1-E20)":
        out_df = out_df.sort_values(by="key_num", ascending=True)
    else:
        out_df = out_df.sort_values(by=["aspect_meaning", "key_num"], ascending=[True, True])

    return out_df.reset_index(drop=True)


def compute_science_internal_consistency(
    df: pd.DataFrame,
    code_to_label: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute inter-item correlations, Cronbach's alpha, and Spearman-Brown reliability
    for the 13 multi-item cognitive Science aspects to detect item-pair divergence or
    reverse-wording polarity effects.
    """
    rows: List[Dict[str, Any]] = []
    for asp in SCIENCE_COGNITIVE_ASPECTS:
        items = SCIENCE_ASPECT_ITEMS[asp]
        k = len(items)
        sub = df[items].dropna()
        n = len(sub)

        if n < 3:
            continue

        corr_mat = sub.corr(method="pearson")
        pair_corrs: List[float] = []
        pair_labels: List[str] = []
        for i in range(k):
            for j in range(i + 1, k):
                r_ij = float(corr_mat.iloc[i, j])
                pair_corrs.append(r_ij)
                pair_labels.append(f"{items[i]}-{items[j]}: r={r_ij:.2f}")

        mean_r = float(np.mean(pair_corrs)) if pair_corrs else np.nan

        item_vars = sub.var(axis=0, ddof=1).sum()
        total_var = sub.sum(axis=1).var(ddof=1)
        if total_var > 0 and k > 1:
            cronbach_alpha = float((k / (k - 1.0)) * (1.0 - (item_vars / total_var)))
        else:
            cronbach_alpha = np.nan

        if not pd.isna(mean_r) and (1.0 + (k - 1.0) * mean_r) != 0:
            spearman_brown = float((k * mean_r) / (1.0 + (k - 1.0) * mean_r))
        else:
            spearman_brown = np.nan

        if mean_r < 0:
            diagnostic_status = "Negative Polarity / Reverse-Wording Divergence"
        elif mean_r < 0.15:
            diagnostic_status = "Orthogonal / Multi-Faceted Aspect"
        else:
            diagnostic_status = "Convergent / Internally Consistent"

        rows.append({
            "aspect_code": asp,
            "aspect_label": code_to_label.get(asp, asp),
            "items": ", ".join(items),
            "k_items": k,
            "mean_inter_item_r": mean_r,
            "cronbach_alpha": cronbach_alpha,
            "spearman_brown": spearman_brown,
            "pairwise_detail": " | ".join(pair_labels),
            "aspect_mean": float(sub.mean(axis=1).mean()),
            "aspect_sd": float(sub.mean(axis=1).std(ddof=1)),
            "diagnostic_status": diagnostic_status,
        })

    out_df = pd.DataFrame(rows)
    return out_df.sort_values(by="mean_inter_item_r", ascending=True).reset_index(drop=True)
