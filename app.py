"""
Main Streamlit BI Dashboard Application (`app.py`)
Investigates the relationship between Views on Science (`SCI`, `SKSI`, `ATS`) and
Views on Evolution (`EVO`, split into Fact `FCT` and Credibility `CRD`) across N = 121
respondents, moderated by Age, Gender, and Religiosity (Religion strictly excluded).
Styled in the Ocean Beach Palette (https://pikka.app/palettes/ocean-beach-palette).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

    class _DummyContext:
        def __enter__(self) -> "_DummyContext":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            return getattr(st, name)

    class _DummyStreamlit:
        sidebar = _DummyContext()

        def set_page_config(self, **kwargs: Any) -> None:
            pass

        def markdown(self, *args: Any, **kwargs: Any) -> None:
            pass

        def title(self, *args: Any, **kwargs: Any) -> None:
            pass

        def subheader(self, *args: Any, **kwargs: Any) -> None:
            pass

        def caption(self, *args: Any, **kwargs: Any) -> None:
            pass

        def info(self, *args: Any, **kwargs: Any) -> None:
            pass

        def warning(self, *args: Any, **kwargs: Any) -> None:
            pass

        def metric(self, *args: Any, **kwargs: Any) -> None:
            pass

        def plotly_chart(self, *args: Any, **kwargs: Any) -> None:
            pass

        def dataframe(self, *args: Any, **kwargs: Any) -> None:
            pass

        def multiselect(self, label: str, options: List[Any], default: Any = None, **kwargs: Any) -> Any:
            return default if default is not None else list(options)

        def slider(self, label: str, min_value: Any, max_value: Any, value: Any = None, **kwargs: Any) -> Any:
            return value if value is not None else (min_value, max_value)

        def radio(self, label: str, options: List[Any], index: int = 0, **kwargs: Any) -> Any:
            return options[index]

        def segmented_control(
            self,
            label: str,
            options: List[Any],
            default: Any = None,
            **kwargs: Any,
        ) -> Any:
            return default if default is not None else options[0]

        def selectbox(self, label: str, options: List[Any], index: int = 0, **kwargs: Any) -> Any:
            return options[index]

        def columns(self, spec: Any, **kwargs: Any) -> List[_DummyContext]:
            count = spec if isinstance(spec, int) else len(spec)
            return [_DummyContext() for _ in range(count)]

        def tabs(self, tab_names: List[str]) -> List[_DummyContext]:
            return [_DummyContext() for _ in tab_names]

        def expander(self, *args: Any, **kwargs: Any) -> _DummyContext:
            return _DummyContext()

    st = _DummyStreamlit()  # type: ignore

from config import (
    ACTIVE_DEMOGRAPHICS,
    EVOLUTION_CRITERIA,
    EVO_ALL_ASPECTS,
    EVO_SPLIT_SUBASPECT_ITEMS,
    OCEAN_BEACH_PALETTE,
    SCIENCE_ALL_ASPECTS,
    SCIENCE_PREDICTORS,
    UI_THEME,
)
from data_engine import (
    compute_bivariate_correlation,
    compute_correlation_matrix,
    compute_epistemic_dumbbell_table,
    compute_likert_distribution_table,
    compute_science_internal_consistency,
    filter_cohort,
    fit_moderation_ols,
    fit_ols_regression,
    get_scored_dataframe,
    load_and_validate_metadata,
    load_survey_data,
)
from visuals import (
    plot_aspect_correlation_heatmap,
    plot_bivariate_aspect_drilldown,
    plot_dual_slope_epistemic_overlay,
    plot_epistemic_dumbbell,
    plot_faceted_moderation_regressions,
    plot_joint_scatter_with_marginals,
    plot_likert_diverging_bars,
    plot_multi_aspect_radar,
    plot_science_internal_consistency,
    plot_subgroup_distributions,
)


# ==============================================================================
# 1. CUSTOM OCEAN BEACH CSS STYLING
# ==============================================================================
def inject_ocean_beach_css() -> None:
    """Inject custom Ocean Beach CSS variables and card styling into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {UI_THEME['bg_main']};
            color: {UI_THEME['text_primary']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {UI_THEME['bg_sidebar']};
            border-right: 1px solid {UI_THEME['border_color']};
        }}
        h1, h2, h3, h4 {{
            color: {UI_THEME['text_primary']} !important;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        div[data-testid="stTabs"] [role="tablist"] {{
            align-items: flex-end;
            border-bottom: 1px solid {UI_THEME['border_color']} !important;
            gap: 0.35rem !important;
            padding: 0 0.35rem !important;
        }}
        div[data-testid="stTabs"] [role="tab"] {{
            background: linear-gradient(180deg, #ffffff 0%, {UI_THEME['bg_card']} 100%) !important;
            border: 1px solid {UI_THEME['border_color']} !important;
            border-bottom-color: {UI_THEME['border_color']} !important;
            border-left: 4px solid {UI_THEME['border_color']} !important;
            border-radius: 8px 8px 0 0 !important;
            color: {UI_THEME['text_primary']} !important;
            cursor: pointer !important;
            font-weight: 600 !important;
            margin-bottom: -1px !important;
            min-height: 2.8rem !important;
            padding: 0.6rem 0.9rem !important;
            transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
        }}
        div[data-testid="stTabs"] [role="tab"] * {{
            color: inherit !important;
        }}
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            display: none !important;
        }}
        div[data-testid="stTabs"] [role="tab"]:hover {{
            border-color: {OCEAN_BEACH_PALETTE['warm_tangerine']} !important;
            border-bottom-color: {UI_THEME['border_color']} !important;
        }}
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            background: linear-gradient(180deg, #ffffff 0%, {UI_THEME['bg_card']} 100%) !important;
            border-color: {UI_THEME['border_color']} !important;
            border-bottom-color: {UI_THEME['bg_main']} !important;
            border-left: 4px solid {OCEAN_BEACH_PALETTE['deep_ocean_blue']} !important;
            box-shadow: 0 -2px 6px rgba(0, 63, 92, 0.06) !important;
            color: {UI_THEME['text_primary']} !important;
        }}
        div[data-testid="stTabs"] [role="tab"]:focus-visible {{
            outline: 3px solid {OCEAN_BEACH_PALETTE['sunset_crimson']} !important;
            outline-offset: 2px !important;
        }}
        .kpi-card {{
            background: linear-gradient(180deg, #ffffff 0%, {UI_THEME['bg_card']} 100%);
            border: 1px solid {UI_THEME['border_color']};
            border-left: 5px solid {OCEAN_BEACH_PALETTE['deep_ocean_blue']};
            border-radius: 10px;
            box-sizing: border-box;
            padding: 14px 16px;
            box-shadow: 0 2px 6px rgba(0, 63, 92, 0.06);
            display: flex;
            flex-direction: column;
            height: 150px;
            min-height: 150px;
        }}
        .kpi-title {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {UI_THEME['text_secondary']};
            margin-bottom: 4px;
        }}
        .kpi-value {{
            font-size: 1.50rem;
            font-weight: 800;
            color: {UI_THEME['text_primary']};
            line-height: 1.2;
        }}
        .kpi-sub {{
            font-size: 0.80rem;
            color: {UI_THEME['text_secondary']};
            margin-top: auto;
        }}
        .insight-banner {{
            background-color: {UI_THEME['bg_sidebar']};
            border: 1px solid {UI_THEME['border_color']};
            border-left: 4px solid {OCEAN_BEACH_PALETTE['warm_tangerine']};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 14px;
            color: {UI_THEME['text_primary']};
            font-size: 0.92rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_plotly_chart(fig: Any) -> None:
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)


# ==============================================================================
# 2. SIDEBAR CONTROLLER
# ==============================================================================
def render_sidebar_controls(
    raw_df: pd.DataFrame,
    code_to_label: Dict[str, str],
) -> Dict[str, Any]:
    """Render global sidebar slicers and psychometric configuration controls."""
    st.sidebar.markdown(
        f"<h1 style='color:{UI_THEME['text_primary']};margin-bottom:0;'>Cohort Controls</h2>"
        f"<p style='color:{UI_THEME['text_secondary']};font-size:0.84rem;margin-top:2px;'>"
        "INSERT DESCRIPTION HERE</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Demographic Slicers")
    all_genders = sorted(raw_df["Gender"].unique().tolist())
    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=all_genders,
        default=all_genders,
        help="Filter respondents by self-reported Gender (F, M).",
    )

    all_religiosities = sorted(raw_df["Religiosity"].unique().tolist())
    selected_religiosities = st.sidebar.multiselect(
        "Religiosity",
        options=all_religiosities,
        default=all_religiosities,
        help="Filter respondents by Religiosity (Religious vs. Non Religious). Religion denomination is excluded by design.",
    )

    min_age, max_age = int(raw_df["Age"].min()), int(raw_df["Age"].max())
    selected_age_range: Tuple[int, int] = st.sidebar.slider(
        "Age Range (Years)",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1,
    )

    st.sidebar.markdown("### Primary Construct Selectors")
    predictor_code = st.sidebar.radio(
        "Primary Science Predictor (X)",
        options=SCIENCE_PREDICTORS,
        format_func=lambda c: code_to_label.get(c, c),
        index=0,
    )

    criterion_code = st.sidebar.radio(
        "Primary Evolution Criterion (Y)",
        options=EVOLUTION_CRITERIA,
        format_func=lambda c: code_to_label.get(c, c),
        index=0,
    )

    st.sidebar.markdown("### Psychometric & Statistical Mode")
    scoring_mode = st.sidebar.radio(
        "Scoring Metric Aggregation",
        options=["Mean (1-5)", "Raw Sum", "Z-Score"],
        index=0,
        help="Mean (1-5) normalizes subscales of different item lengths to a comparable 1–5 Likert scale.",
    )

    corr_method = st.sidebar.radio(
        "Correlation Estimator",
        options=["Pearson (r)", "Spearman (rho)"],
        index=0,
    )

    return {
        "genders": selected_genders,
        "religiosities": selected_religiosities,
        "age_range": selected_age_range,
        "predictor_code": predictor_code,
        "criterion_code": criterion_code,
        "scoring_mode": scoring_mode,
        "corr_method": corr_method,
    }


# ==============================================================================
# 3. MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main() -> None:
    """Launch the 4-tab Streamlit Psychometric & Epistemic BI Dashboard."""
    st.set_page_config(
        page_title="Psychometric & Epistemic BI Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_ocean_beach_css()

    # Load validated metadata and survey data (Religion strictly dropped)
    meta = load_and_validate_metadata()
    code_to_meaning: Dict[str, str] = meta["code_to_meaning"]
    code_to_label: Dict[str, str] = meta["code_to_label"]
    item_meta: Dict[str, Dict[str, str]] = meta["item_meta"]

    raw_df = load_survey_data()
    total_n = len(raw_df)

    controls = render_sidebar_controls(raw_df, code_to_label)

    # Filter cohort and apply selected scoring aggregation
    filtered_raw = filter_cohort(
        raw_df,
        genders=controls["genders"],
        religiosities=controls["religiosities"],
        age_range=controls["age_range"],
    )
    scored_df = get_scored_dataframe(filtered_raw, scoring_mode=controls["scoring_mode"])
    active_n = len(scored_df)

    # Top Dashboard Header
    st.title("Psychometric & Epistemic Business Intelligence Dashboard")
    st.caption(
        f"**Questionnaires:** {code_to_label.get('VNOS', 'VNOS')} & {code_to_label.get('VOTE', 'VOTE')} | "
        f"**Active Sample:** n = {active_n} / {total_n} | "
        f"**Scoring Mode:** {controls['scoring_mode']} | **Correlation:** {controls['corr_method']}"
    )
    if active_n < 3:
        st.warning("Please select at least 3 respondents in the sidebar filters to compute psychometric models.")
        return

    pred_code = controls["predictor_code"]
    crit_code = controls["criterion_code"]
    pred_label = code_to_label.get(pred_code, pred_code)
    crit_label = code_to_label.get(crit_code, crit_code)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Macro Relationship & Epistemic Split",
        "Aspect Cross-Correlation Matrix",
        "Item Diagnostics & Epistemic Gap",
        "Demographic Moderation & Profiles",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: MACRO RELATIONSHIP & EPISTEMIC SPLIT (EXECUTIVE OVERVIEW)
    # --------------------------------------------------------------------------
    with tab1:
        primary_corr = compute_bivariate_correlation(
            scored_df, pred_code, crit_code, method=controls["corr_method"]
        )

        pred_mean = float(scored_df[pred_code].mean())
        pred_sd = float(scored_df[pred_code].std(ddof=1))
        crit_mean = float(scored_df[crit_code].mean())
        crit_sd = float(scored_df[crit_code].std(ddof=1))

        # 4 KPI Metric Ribbon Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">Active Sample Cohort</div>
                <div class="kpi-value">{active_n} / {total_n}</div>
                <div class="kpi-sub">{active_n / total_n * 100:.1f}% of N=121 respondents</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with k2:
            r_str = f"{primary_corr['r']:.4f}{primary_corr['stars']}" if not pd.isna(primary_corr["r"]) else "N/A"
            p_str = f"p = {primary_corr['p']:.2e}" if not pd.isna(primary_corr["p"]) else "N/A"
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">{pred_code} × {crit_code} ({primary_corr['method_symbol']})</div>
                <div class="kpi-value">{r_str}</div>
                <div class="kpi-sub">Two-tailed {p_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">{pred_label}</div>
                <div class="kpi-value">{pred_mean:.2f} <span style="font-size:0.95rem;font-weight:500;">(SD {pred_sd:.2f})</span></div>
                <div class="kpi-sub">Metric: {controls['scoring_mode']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-title">{crit_label}</div>
                <div class="kpi-value">{crit_mean:.2f} <span style="font-size:0.95rem;font-weight:500;">(SD {crit_sd:.2f})</span></div>
                <div class="kpi-sub">Metric: {controls['scoring_mode']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        scatter_color = getattr(st, "session_state", {}).get("joint_scatter_color", "None")
        fig_joint = plot_joint_scatter_with_marginals(
            scored_df,
            x_col=pred_code,
            y_col=crit_code,
            x_label=f"{pred_label} [{controls['scoring_mode']}]",
            y_label=f"{crit_label} [{controls['scoring_mode']}]",
            color_by=scatter_color,
            corr_method=controls["corr_method"],
        )
        joint_header = fig_joint.layout.title.text
        fig_joint.update_layout(title_text="", margin=dict(l=65, r=40, t=45, b=60))
        st.markdown(
            f"<div style='font-family:Inter, Segoe UI, sans-serif;font-size:17px;line-height:1.35;color:{UI_THEME['text_primary']};margin-bottom:0.5rem;'>{joint_header}</div>",
            unsafe_allow_html=True,
        )
        st.segmented_control(
            "Color Joint Scatterplot By Moderator:",
            options=["None", "Religiosity", "Gender", "Age"],
            default="None",
            selection_mode="single",
            width="stretch",
            key="joint_scatter_color",
        )
        render_plotly_chart(fig_joint)

        fig_dual = plot_dual_slope_epistemic_overlay(
            scored_df,
            predictor_code=pred_code,
            predictor_label=pred_label,
        )
        render_plotly_chart(fig_dual)

    # --------------------------------------------------------------------------
    # TAB 2: ASPECT-TO-ASPECT CROSS-CORRELATION MATRIX (GRANULAR EXPLORER)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Granular Aspect-to-Aspect Cross-Correlation Matrix")
        matrix_view = st.radio(
            "Select Evolution Matrix Columns View:",
            options=[
                "Standard View (6 EVO Aspects + FCT, CRD, EVO)",
                "Epistemic Split View (10 EVO Sub-Aspects: FCT vs. CRD)",
            ],
            horizontal=True,
        )

        if "Standard View" in matrix_view:
            target_evo_cols = EVO_ALL_ASPECTS + ["FCT", "CRD", "EVO"]
        else:
            target_evo_cols = list(EVO_SPLIT_SUBASPECT_ITEMS.keys())

        r_mat, p_mat, annot_mat = compute_correlation_matrix(
            scored_df,
            row_cols=SCIENCE_ALL_ASPECTS,
            col_cols=target_evo_cols,
            method=controls["corr_method"],
        )

        fig_heatmap = plot_aspect_correlation_heatmap(
            r_mat=r_mat,
            p_mat=p_mat,
            annot_mat=annot_mat,
            code_to_label=code_to_label,
            code_to_meaning=code_to_meaning,
            view_mode=matrix_view,
            corr_method=controls["corr_method"],
        )
        render_plotly_chart(fig_heatmap)

        st.subheader("Bivariate Aspect Drill-Down Inspector")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            sel_sci_aspect = st.selectbox(
                "Select Science Aspect (X):",
                options=SCIENCE_ALL_ASPECTS,
                format_func=lambda c: code_to_label.get(c, c),
                index=0,
            )
        with d_col2:
            drill_evo_options = EVO_ALL_ASPECTS + list(EVO_SPLIT_SUBASPECT_ITEMS.keys())
            sel_evo_aspect = st.selectbox(
                "Select Evolution Aspect or Epistemic Sub-Aspect (Y):",
                options=drill_evo_options,
                format_func=lambda c: code_to_label.get(c, c),
                index=0,
            )

        fig_drill = plot_bivariate_aspect_drilldown(
            scored_df,
            sci_aspect=sel_sci_aspect,
            evo_aspect=sel_evo_aspect,
            sci_label=code_to_label.get(sel_sci_aspect, sel_sci_aspect),
            evo_label=code_to_label.get(sel_evo_aspect, sel_evo_aspect),
            corr_method=controls["corr_method"],
        )
        render_plotly_chart(fig_drill)

    # --------------------------------------------------------------------------
    # TAB 3: ITEM DIAGNOSTICS, POLARITY & EPISTEMIC GAP (48-ITEM DEEP DIVE)
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("1. Epistemic Divergence Dumbbell Plot (Fact vs. Credibility Gap)")
        dumbbell_df = compute_epistemic_dumbbell_table(scored_df, code_to_label)
        fig_dumbbell = plot_epistemic_dumbbell(dumbbell_df)
        render_plotly_chart(fig_dumbbell)

        with st.expander("View Paired Epistemic Divergence Summary Table (val, pro, evi, bel & Overall)"):
            st.dataframe(dumbbell_df, use_container_width=True)

        st.subheader("2. 48-Item Diverging Stacked Bar Chart (1–5 Likert Response Distribution)")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            selected_constructs = st.multiselect(
                "Filter Items by Construct:",
                options=["SCI", "ATS", "FCT", "CRD"],
                default=["SCI", "ATS", "FCT", "CRD"],
                format_func=lambda c: code_to_label.get(c, c),
            )
        with f_col2:
            likert_sort = st.selectbox(
                "Sort Items By:",
                options=[
                    "Aspect Meaning",
                    "Mean Endorsement (High to Low)",
                    "Mean Endorsement (Low to High)",
                    "Agreement % (4+5)",
                    "Item Number (S1-E20)",
                ],
                index=0,
            )

        likert_df = compute_likert_distribution_table(
            scored_df,
            item_meta=item_meta,
            construct_filter=selected_constructs,
            sort_by=likert_sort,
        )
        fig_likert = plot_likert_diverging_bars(likert_df)
        render_plotly_chart(fig_likert)

        st.subheader("3. Science Aspect Internal Consistency & Reverse-Wording Diagnostic")
        consistency_df = compute_science_internal_consistency(scored_df, code_to_label)
        fig_consistency = plot_science_internal_consistency(consistency_df)
        render_plotly_chart(fig_consistency)

        with st.expander("View Science Aspect Inter-Item Correlation & Reliability Table"):
            st.dataframe(consistency_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: DEMOGRAPHIC MODERATION & SUBGROUP PROFILES
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Demographic Moderation & Subgroup Epistemic Profiles")
        mod_col1, _ = st.columns([1, 2])
        with mod_col1:
            selected_moderator = st.selectbox(
                "Select Demographic Moderator:",
                options=ACTIVE_DEMOGRAPHICS,
                index=2,  # Default to Religiosity
            )

        mod_res = fit_moderation_ols(
            scored_df,
            x_col=pred_code,
            y_col=crit_code,
            moderator=selected_moderator,
        )

        if mod_res.get("valid"):
            st.markdown(
                f"""<div class="insight-banner">
                <b>Automated OLS Moderation Readout ({mod_res['formula']}):</b><br>
                • <b>Interaction Term ({mod_res['interaction_term']}):</b> β = {mod_res['beta_interaction']:.4f}{mod_res['stars_interaction']}
                (SE = {mod_res['se_interaction']:.4f}, t = {mod_res['t_interaction']:.3f}, two-tailed p = {mod_res['p_interaction']:.4f})<br>
                • <b>Overall Model Fit:</b> R² = {mod_res['r2']:.4f} (Adjusted R² = {mod_res['adj_r2']:.4f}),
                F = {mod_res['f_stat']:.3f} (p = {mod_res['f_pval']:.3e}), Active n = {mod_res['n']}
                </div>""",
                unsafe_allow_html=True,
            )

        fig_mod = plot_faceted_moderation_regressions(
            scored_df,
            x_col=pred_code,
            y_col=crit_code,
            x_label=f"{pred_label} [{controls['scoring_mode']}]",
            y_label=f"{crit_label} [{controls['scoring_mode']}]",
            moderator=selected_moderator,
            mod_res=mod_res,
        )
        render_plotly_chart(fig_mod)

        fig_dist = plot_subgroup_distributions(
            scored_df,
            moderator=selected_moderator,
            code_to_label=code_to_label,
        )
        render_plotly_chart(fig_dist)

        fig_radar = plot_multi_aspect_radar(
            scored_df,
            code_to_label=code_to_label,
            moderator=selected_moderator,
        )
        render_plotly_chart(fig_radar)


if __name__ == "__main__":
    main()
elif HAS_STREAMLIT:
    main()
