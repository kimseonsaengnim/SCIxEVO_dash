"""
Reusable Plotly Visualization Library for the Psychometric & Epistemic BI Dashboard.
All charts are styled in the Ocean Beach Palette (https://pikka.app/palettes/ocean-beach-palette)
and dynamically labeled using TGcodes.csv meanings.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

    class _DummyTrace(dict):
        def __init__(self, trace_type: str, **kwargs: Any) -> None:
            super().__init__(trace_type=trace_type, **kwargs)

    class _DummyFigure:
        def __init__(self, data: Any = None, layout: Any = None) -> None:
            self.data: List[Any] = list(data) if data else []
            self.layout: Dict[str, Any] = dict(layout) if layout else {}
            self.annotations: List[Any] = []

        def add_trace(self, trace: Any, row: int | None = None, col: int | None = None, **kwargs: Any) -> "_DummyFigure":
            self.data.append({"trace": trace, "row": row, "col": col, **kwargs})
            return self

        def update_layout(self, **kwargs: Any) -> "_DummyFigure":
            self.layout.update(kwargs)
            return self

        def update_xaxes(self, **kwargs: Any) -> "_DummyFigure":
            self.layout.setdefault("xaxes_updates", []).append(kwargs)
            return self

        def update_yaxes(self, **kwargs: Any) -> "_DummyFigure":
            self.layout.setdefault("yaxes_updates", []).append(kwargs)
            return self

        def add_vline(self, **kwargs: Any) -> "_DummyFigure":
            self.layout.setdefault("vlines", []).append(kwargs)
            return self

        def add_hline(self, **kwargs: Any) -> "_DummyFigure":
            self.layout.setdefault("hlines", []).append(kwargs)
            return self

        def add_annotation(self, **kwargs: Any) -> "_DummyFigure":
            self.annotations.append(kwargs)
            return self

    class _DummyGo:
        Figure = _DummyFigure
        Scatter = staticmethod(lambda **kw: _DummyTrace("Scatter", **kw))
        Scatterpolar = staticmethod(lambda **kw: _DummyTrace("Scatterpolar", **kw))
        Bar = staticmethod(lambda **kw: _DummyTrace("Bar", **kw))
        Box = staticmethod(lambda **kw: _DummyTrace("Box", **kw))
        Violin = staticmethod(lambda **kw: _DummyTrace("Violin", **kw))
        Heatmap = staticmethod(lambda **kw: _DummyTrace("Heatmap", **kw))
        Histogram2d = staticmethod(lambda **kw: _DummyTrace("Histogram2d", **kw))
        Histogram2dContour = staticmethod(lambda **kw: _DummyTrace("Histogram2dContour", **kw))

    go = _DummyGo()  # type: ignore

    def make_subplots(**kwargs: Any) -> _DummyFigure:  # type: ignore
        return _DummyFigure(layout={"subplots": kwargs})

from config import (
    EVO_ALL_ASPECTS,
    LIKERT_COLORS,
    LIKERT_LABELS,
    OCEAN_BEACH_PALETTE,
    OCEAN_BEACH_SEQUENCE,
    SCIENCE_ALL_ASPECTS,
    UI_THEME,
)
from data_engine import (
    compute_bivariate_correlation,
    fit_ols_regression,
)


# ==============================================================================
# 1. GLOBAL OCEAN BEACH THEME LAYOUT HELPER
# ==============================================================================
OCEAN_BEACH_COLORSCALE: List[List[Any]] = [
    [0.00, "#003f5c"],
    [0.25, "#2f4b7c"],
    [0.50, "#f8f9fa"],
    [0.75, "#f95d6a"],
    [0.90, "#ff7c43"],
    [1.00, "#ffa600"],
]

OCEAN_BEACH_SEQUENTIAL_DENSITY: List[List[Any]] = [
    [0.00, "#f8f9fa"],
    [0.25, "#eef2f5"],
    [0.50, "#665191"],
    [0.75, "#d45087"],
    [1.00, "#ffa600"],
]

MODERATOR_COLOR_MAPS: Dict[str, Dict[str, str]] = {
    "Religiosity": {
        "Religious": "#003f5c",
        "Non Religious": "#ff7c43",
    },
    "Gender": {
        "F": "#d45087",
        "M": "#003f5c",
    },
    "Age": {
        "14": "#003f5c",
        "15": "#2f4b7c",
        "16": "#665191",
        "17": "#f95d6a",
        "18": "#ffa600",
    },
}


def apply_ocean_beach_layout(
    fig: Any,
    title: str,
    subtitle: str = "",
    height: int = 520,
) -> Any:
    """Apply consistent Ocean Beach Palette styling, typography, and surface backgrounds."""
    full_title = (
        f"<b>{title}</b><br><br><span style='font-size:12px;color:{UI_THEME['text_secondary']}'>{subtitle}</span>"
        if subtitle
        else f"<b>{title}</b>"
    )
    fig.update_layout(
        title=dict(
            text=full_title,
            font=dict(size=17, color=UI_THEME["text_primary"], family="Inter, Segoe UI, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=UI_THEME["bg_main"],
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color=UI_THEME["text_primary"], size=12),
        height=height + 55,
        margin=dict(l=65, r=40, t=140, b=60),
        legend=dict(
            bgcolor="rgba(255, 249, 240, 0.90)",
            bordercolor=UI_THEME["border_color"],
            borderwidth=1,
            font=dict(size=11, color=UI_THEME["text_primary"]),
        ),
        hoverlabel=dict(
            bgcolor="#003f5c",
            font_size=12,
            font_family="Inter, Segoe UI, sans-serif",
            font_color="#ffffff",
        ),
    )
    fig.update_xaxes(
        gridcolor="#e9ecef",
        linecolor=UI_THEME["border_color"],
        zerolinecolor="#ced4da",
        title_font=dict(size=13, color=UI_THEME["text_primary"]),
        tickfont=dict(size=11, color=UI_THEME["text_secondary"]),
    )
    fig.update_yaxes(
        gridcolor="#e9ecef",
        linecolor=UI_THEME["border_color"],
        zerolinecolor="#ced4da",
        title_font=dict(size=13, color=UI_THEME["text_primary"]),
        tickfont=dict(size=11, color=UI_THEME["text_secondary"]),
    )
    return fig


# ==============================================================================
# 2. TAB 1 CHARTS: EXECUTIVE OVERVIEW & EPISTEMIC SPLIT
# ==============================================================================
def plot_joint_scatter_with_marginals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    color_by: str = "Religiosity",
    corr_method: str = "Pearson (r)",
) -> Any:
    """
    Joint Scatterplot with Marginal Box/Violin Distributions, OLS regression trendline,
    95% confidence band, and optional demographic color-coding (Religiosity, Gender, Age).
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.80, 0.20],
        row_heights=[0.22, 0.78],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )

    corr_res = compute_bivariate_correlation(df, x_col, y_col, method=corr_method)
    ols_res = fit_ols_regression(df, x_col, y_col)

    if color_by in ("Religiosity", "Gender", "Age"):
        palette_map = MODERATOR_COLOR_MAPS.get(color_by, {})
        groups = sorted(df[color_by].astype(str).unique())
        for idx, grp_val in enumerate(groups):
            sub = df[df[color_by].astype(str) == grp_val]
            color = palette_map.get(grp_val, OCEAN_BEACH_SEQUENCE[idx % len(OCEAN_BEACH_SEQUENCE)])

            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode="markers",
                    name=f"{color_by}: {grp_val} (n={len(sub)})",
                    legendgroup=grp_val,
                    marker=dict(
                        color=color,
                        size=9,
                        opacity=0.85,
                        line=dict(width=1, color="#ffffff"),
                    ),
                    customdata=np.column_stack([sub["ID"], sub["Gender"], sub["Religiosity"], sub["Age"]]),
                    hovertemplate=(
                        "<b>Respondent ID: %{customdata[0]}</b><br>"
                        f"{x_label}: %{{x:.2f}}<br>"
                        f"{y_label}: %{{y:.2f}}<br>"
                        "Gender: %{customdata[1]} | Religiosity: %{customdata[2]} | Age: %{customdata[3]}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Violin(
                    x=sub[x_col],
                    name=f"{grp_val}",
                    legendgroup=grp_val,
                    showlegend=False,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.65,
                    box_visible=True,
                    meanline_visible=True,
                    orientation="h",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Violin(
                    y=sub[y_col],
                    name=f"{grp_val}",
                    legendgroup=grp_val,
                    showlegend=False,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.65,
                    box_visible=True,
                    meanline_visible=True,
                    orientation="v",
                ),
                row=2,
                col=2,
            )
    else:
        color = UI_THEME["accent_fct"]
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                name=f"Respondents (n={len(df)})",
                marker=dict(color=color, size=9, opacity=0.85, line=dict(width=1, color="#ffffff")),
                customdata=np.column_stack([df["ID"], df["Gender"], df["Religiosity"], df["Age"]]),
                hovertemplate=(
                    "<b>Respondent ID: %{customdata[0]}</b><br>"
                    f"{x_label}: %{{x:.2f}}<br>"
                    f"{y_label}: %{{y:.2f}}<br>"
                    "Gender: %{customdata[1]} | Religiosity: %{customdata[2]} | Age: %{customdata[3]}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Violin(
                x=df[x_col],
                showlegend=False,
                line_color=color,
                fillcolor=color,
                opacity=0.65,
                box_visible=True,
                meanline_visible=True,
                orientation="h",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Violin(
                y=df[y_col],
                showlegend=False,
                line_color=UI_THEME["accent_crd"],
                fillcolor=UI_THEME["accent_crd"],
                opacity=0.65,
                box_visible=True,
                meanline_visible=True,
                orientation="v",
            ),
            row=2,
            col=2,
        )

    if len(ols_res["x_grid"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([ols_res["x_grid"], ols_res["x_grid"][::-1]]),
                y=np.concatenate([ols_res["ci_high"], ols_res["ci_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(249, 93, 106, 0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% OLS Confidence Band",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ols_res["x_grid"],
                y=ols_res["y_hat"],
                mode="lines",
                name=f"OLS Trend (β={ols_res['beta']:.2f}, R²={ols_res['r2']:.3f})",
                line=dict(color=OCEAN_BEACH_PALETTE["sunset_crimson"], width=3),
            ),
            row=2,
            col=1,
        )

    subtitle = (
        f"{corr_res['method_symbol']} = {corr_res['r']:.4f}{corr_res['stars']} "
        f"(p = {corr_res['p']:.2e}) | OLS β = {ols_res['beta']:.3f}, R² = {ols_res['r2']:.3f} | Active n = {len(df)}"
        if not pd.isna(corr_res["r"])
        else f"Active n = {len(df)}"
    )
    apply_ocean_beach_layout(
        fig,
        title=f"Joint Distribution: {x_label} vs. {y_label}",
        subtitle=subtitle,
        height=580,
    )
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_yaxes(title_text=y_label, row=2, col=1)
    fig.update_xaxes(showticklabels=False, title_text="", row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="", row=1, col=1)
    fig.update_xaxes(showticklabels=False, title_text="", row=2, col=2)
    fig.update_yaxes(showticklabels=False, title_text="", row=2, col=2)
    return fig


def plot_dual_slope_epistemic_overlay(
    df: pd.DataFrame,
    predictor_code: str,
    predictor_label: str,
) -> Any:
    """
    Dual-Slope Epistemic Overlay Canvas: Plots Science Predictor on X-axis (1-5 mean scale)
    and overlays BOTH FCT (Deep Ocean #003f5c) and CRD (Sunset Crimson #f95d6a) on Y-axis
    (normalized to a 1-5 mean scale) with separate OLS regression lines and 95% CI bands.
    """
    x_col = f"{predictor_code}_mean"
    fct_col = "FCT_mean"
    crd_col = "CRD_mean"

    ols_fct = fit_ols_regression(df, x_col, fct_col)
    ols_crd = fit_ols_regression(df, x_col, crd_col)
    corr_fct = compute_bivariate_correlation(df, x_col, fct_col)
    corr_crd = compute_bivariate_correlation(df, x_col, crd_col)

    fig = go.Figure()

    if len(ols_fct["x_grid"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([ols_fct["x_grid"], ols_fct["x_grid"][::-1]]),
                y=np.concatenate([ols_fct["ci_high"], ols_fct["ci_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(0, 63, 92, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Fact (FCT) 95% CI",
            )
        )

    if len(ols_crd["x_grid"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([ols_crd["x_grid"], ols_crd["x_grid"][::-1]]),
                y=np.concatenate([ols_crd["ci_high"], ols_crd["ci_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(249, 93, 106, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Credibility (CRD) 95% CI",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[fct_col],
            mode="markers",
            name=f"Fact (FCT) Items Mean (r={corr_fct['r']:.3f}{corr_fct['stars']})",
            marker=dict(
                color=UI_THEME["accent_fct"],
                size=8,
                symbol="circle",
                opacity=0.75,
                line=dict(width=1, color="#ffffff"),
            ),
            customdata=df["ID"],
            hovertemplate=(
                "<b>ID: %{customdata}</b> | Construct: Fact (FCT)<br>"
                f"{predictor_label} (1-5): %{{x:.2f}}<br>"
                "FCT Mean (1-5): %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[crd_col],
            mode="markers",
            name=f"Credibility (CRD) Items Mean (r={corr_crd['r']:.3f}{corr_crd['stars']})",
            marker=dict(
                color=UI_THEME["accent_crd"],
                size=8,
                symbol="diamond",
                opacity=0.75,
                line=dict(width=1, color="#ffffff"),
            ),
            customdata=df["ID"],
            hovertemplate=(
                "<b>ID: %{customdata}</b> | Construct: Credibility (CRD)<br>"
                f"{predictor_label} (1-5): %{{x:.2f}}<br>"
                "CRD Mean (1-5): %{y:.2f}<extra></extra>"
            ),
        )
    )

    if len(ols_fct["x_grid"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=ols_fct["x_grid"],
                y=ols_fct["y_hat"],
                mode="lines",
                name=f"FCT OLS Slope (β={ols_fct['beta']:.3f}, R²={ols_fct['r2']:.3f})",
                line=dict(color=UI_THEME["accent_fct"], width=3.5, dash="solid"),
            )
        )
    if len(ols_crd["x_grid"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=ols_crd["x_grid"],
                y=ols_crd["y_hat"],
                mode="lines",
                name=f"CRD OLS Slope (β={ols_crd['beta']:.3f}, R²={ols_crd['r2']:.3f})",
                line=dict(color=UI_THEME["accent_crd"], width=3.5, dash="dash"),
            )
        )

    subtitle = (
        f"Normalized 1–5 Likert Scale | FCT Slope β = {ols_fct['beta']:.3f} (p = {ols_fct['p']:.2e}) vs. "
        f"CRD Slope β = {ols_crd['beta']:.3f} (p = {ols_crd['p']:.2e})"
        if not pd.isna(ols_fct["beta"])
        else "Normalized 1–5 Likert Scale"
    )
    apply_ocean_beach_layout(
        fig,
        title=f"Dual-Slope Epistemic Overlay: {predictor_label} → Fact (FCT) vs. Credibility (CRD)",
        subtitle=subtitle,
        height=540,
    )
    fig.update_xaxes(title_text=f"{predictor_label} (Normalized 1–5 Item Mean)")
    fig.update_yaxes(title_text="Evolution Endorsement (Normalized 1–5 Item Mean)")
    return fig


# ==============================================================================
# 3. TAB 2 CHARTS: ASPECT CROSS-CORRELATION MATRIX & DRILL-DOWN INSPECTOR
# ==============================================================================
def plot_aspect_correlation_heatmap(
    r_mat: pd.DataFrame,
    p_mat: pd.DataFrame,
    annot_mat: pd.DataFrame,
    code_to_label: Dict[str, str],
    code_to_meaning: Dict[str, str],
    view_mode: str = "Standard View (6 EVO Aspects)",
    corr_method: str = "Pearson (r)",
) -> Any:
    """
    Interactive Correlation Heatmap between the 14 Science Aspects (rows) and
    either the 6 EVO Aspects (+ FCT, CRD, EVO summaries) or 10 Epistemic Split EVO Sub-Aspects.
    """
    row_labels = [code_to_label.get(r, r) for r in r_mat.index]
    col_labels = [code_to_label.get(c, c) for c in r_mat.columns]

    hover_matrix: List[List[str]] = []
    for r_code in r_mat.index:
        hover_row: List[str] = []
        r_full = code_to_label.get(r_code, r_code)
        for c_code in r_mat.columns:
            c_full = code_to_label.get(c_code, c_code)
            r_val = r_mat.loc[r_code, c_code]
            p_val = p_mat.loc[r_code, c_code]
            annot_str = annot_mat.loc[r_code, c_code]
            hover_row.append(
                f"<b>Science Aspect:</b> {r_full}<br>"
                f"<b>Evolution Target:</b> {c_full}<br>"
                f"<b>{corr_method}:</b> {annot_str} (r = {r_val:.4f})<br>"
                f"<b>Two-tailed p-value:</b> {p_val:.4e}"
            )
        hover_matrix.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=r_mat.values,
            x=col_labels,
            y=row_labels,
            text=annot_mat.values,
            texttemplate="%{text}",
            textfont=dict(size=11, family="Inter, Segoe UI, sans-serif"),
            hovertext=hover_matrix,
            hoverinfo="text",
            colorscale=OCEAN_BEACH_COLORSCALE,
            zmin=-0.6,
            zmax=0.6,
            colorbar=dict(
                title=dict(text=corr_method, font=dict(size=12, color=UI_THEME["text_primary"])),
                tickvals=[-0.5, -0.25, 0.0, 0.25, 0.5],
            ),
        )
    )

    apply_ocean_beach_layout(
        fig,
        title=f"Aspect-to-Aspect Cross-Correlation Matrix ({view_mode})",
        subtitle=f"{corr_method} | Significance: * p < .05, ** p < .01, *** p < .001 | Ocean Beach Diverging Scale",
        height=640,
    )
    fig.update_xaxes(tickangle=-30, side="bottom")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_bivariate_aspect_drilldown(
    df: pd.DataFrame,
    sci_aspect: str,
    evo_aspect: str,
    sci_label: str,
    evo_label: str,
    corr_method: str = "Pearson (r)",
) -> Any:
    """
    Bivariate Aspect Drill-Down Inspector: Side-by-side Jittered Scatterplot with OLS
    trendline & student ID tooltips (left) and 2D Contingency Density Heatmap (right).
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Jittered Scatterplot & OLS Trendline ({sci_aspect} × {evo_aspect})",
            f"2D Contingency Density Surface ({sci_aspect} × {evo_aspect})",
        ),
        horizontal_spacing=0.10,
    )

    corr_res = compute_bivariate_correlation(df, sci_aspect, evo_aspect, method=corr_method)
    ols_res = fit_ols_regression(df, sci_aspect, evo_aspect)

    rng = np.random.default_rng(seed=42)
    x_raw = df[sci_aspect].astype(float).to_numpy()
    y_raw = df[evo_aspect].astype(float).to_numpy()
    x_span = max(float(np.ptp(x_raw)), 1.0)
    y_span = max(float(np.ptp(y_raw)), 1.0)
    x_jit = x_raw + rng.uniform(-0.022 * x_span, 0.022 * x_span, size=len(df))
    y_jit = y_raw + rng.uniform(-0.022 * y_span, 0.022 * y_span, size=len(df))

    for grp_val, color in MODERATOR_COLOR_MAPS["Religiosity"].items():
        mask = (df["Religiosity"] == grp_val).to_numpy()
        if not np.any(mask):
            continue
        sub = df.loc[mask]
        fig.add_trace(
            go.Scatter(
                x=x_jit[mask],
                y=y_jit[mask],
                mode="markers",
                name=f"{grp_val} (n={len(sub)})",
                marker=dict(color=color, size=9, opacity=0.82, line=dict(width=1, color="#ffffff")),
                customdata=np.column_stack([
                    sub["ID"],
                    sub[sci_aspect],
                    sub[evo_aspect],
                    sub["Gender"],
                    sub["Age"],
                ]),
                hovertemplate=(
                    "<b>Student ID: %{customdata[0]}</b><br>"
                    f"{sci_label} (exact): %{{customdata[1]:.2f}}<br>"
                    f"{evo_label} (exact): %{{customdata[2]:.2f}}<br>"
                    f"Religiosity: {grp_val} | Gender: %{{customdata[3]}} | Age: %{{customdata[4]}}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if len(ols_res["x_grid"]) > 0:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([ols_res["x_grid"], ols_res["x_grid"][::-1]]),
                y=np.concatenate([ols_res["ci_high"], ols_res["ci_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(249, 93, 106, 0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% OLS CI",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ols_res["x_grid"],
                y=ols_res["y_hat"],
                mode="lines",
                name=f"OLS Line (β={ols_res['beta']:.2f}, p={ols_res['p']:.3f})",
                line=dict(color=OCEAN_BEACH_PALETTE["sunset_crimson"], width=3),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram2d(
            x=x_raw,
            y=y_raw,
            showlegend=False,
            colorscale=OCEAN_BEACH_SEQUENTIAL_DENSITY,
            nbinsx=10,
            nbinsy=10,
            colorbar=dict(title="Count", x=1.02, len=0.85),
            hovertemplate=(
                f"{sci_label} Bin: %{{x}}<br>"
                f"{evo_label} Bin: %{{y}}<br>"
                "Respondents: %{z}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    subtitle = (
        f"{corr_res['method_symbol']} = {corr_res['r']:.4f}{corr_res['stars']} (p = {corr_res['p']:.3e}) | "
        f"OLS Slope β = {ols_res['beta']:.3f}, R² = {ols_res['r2']:.3f}"
        if not pd.isna(corr_res["r"])
        else "Insufficient variance for OLS"
    )
    apply_ocean_beach_layout(
        fig,
        title=f"Bivariate Aspect Drill-Down: {sci_label} × {evo_label}",
        subtitle=subtitle,
        height=650,
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.22,
            yanchor="top",
        ),
        margin=dict(l=65, r=65, t=200, b=120),
    )
    fig.update_xaxes(title_text=sci_label, row=1, col=1)
    fig.update_yaxes(title_text=evo_label, row=1, col=1)
    fig.update_xaxes(title_text=sci_label, row=1, col=2)
    fig.update_yaxes(title_text=evo_label, row=1, col=2)
    return fig


# ==============================================================================
# 4. TAB 3 CHARTS: ITEM DIAGNOSTICS, POLARITY & EPISTEMIC GAP
# ==============================================================================
def plot_epistemic_dumbbell(dumbbell_df: pd.DataFrame) -> Any:
    """
    Epistemic Divergence Dumbbell Plot across the 4 overlapping EVO aspects
    (val, pro, evi, bel) plus Overall (FCT vs. CRD), comparing mean Fact (#003f5c)
    endorsement vs. mean Credibility (#ff7c43) endorsement.
    """
    fig = go.Figure()

    for _, row in dumbbell_df.iterrows():
        y_label = row["aspect_label"]
        fct_val = row["fct_mean"]
        crd_val = row["crd_mean"]
        gap = row["gap_fct_minus_crd"]
        stars = row["stars"]

        fig.add_trace(
            go.Scatter(
                x=[fct_val, crd_val],
                y=[y_label, y_label],
                mode="lines",
                line=dict(color=OCEAN_BEACH_PALETTE["dusk_purple"], width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        mid_x = (fct_val + crd_val) / 2.0
        fig.add_annotation(
            x=mid_x,
            y=y_label,
            text=f"<b>Δ = {gap:+.2f}{stars}</b>",
            showarrow=False,
            yshift=16,
            font=dict(size=11, color=UI_THEME["text_primary"]),
        )

    fig.add_trace(
        go.Scatter(
            x=dumbbell_df["fct_mean"],
            y=dumbbell_df["aspect_label"],
            mode="markers+text",
            name="Fact Subscale (FCT Mean)",
            text=[f"{v:.2f}" for v in dumbbell_df["fct_mean"]],
            textposition="bottom center",
            marker=dict(color=OCEAN_BEACH_PALETTE["deep_ocean_blue"], size=15, symbol="circle", line=dict(width=2, color="#ffffff")),
            customdata=np.column_stack([dumbbell_df["fct_items"], dumbbell_df["gap_fct_minus_crd"], dumbbell_df["p_val"]]),
            hovertemplate=(
                "<b>%{y} — Fact (FCT)</b><br>"
                "Mean (1–5): %{x:.3f}<br>"
                "Items: %{customdata[0]}<br>"
                "Gap (FCT - CRD): %{customdata[1]:+.3f} (p = %{customdata[2]:.3e})<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dumbbell_df["crd_mean"],
            y=dumbbell_df["aspect_label"],
            mode="markers+text",
            name="Credibility Subscale (CRD Mean)",
            text=[f"{v:.2f}" for v in dumbbell_df["crd_mean"]],
            textposition="top center",
            marker=dict(color=OCEAN_BEACH_PALETTE["warm_tangerine"], size=15, symbol="diamond", line=dict(width=2, color="#ffffff")),
            customdata=np.column_stack([dumbbell_df["crd_items"], dumbbell_df["gap_fct_minus_crd"], dumbbell_df["p_val"]]),
            hovertemplate=(
                "<b>%{y} — Credibility (CRD)</b><br>"
                "Mean (1–5): %{x:.3f}<br>"
                "Items: %{customdata[0]}<br>"
                "Gap (FCT - CRD): %{customdata[1]:+.3f} (p = %{customdata[2]:.3e})<extra></extra>"
            ),
        )
    )

    apply_ocean_beach_layout(
        fig,
        title="Epistemic Divergence Dumbbell Plot: Fact (FCT) vs. Credibility (CRD) Across Overlapping Evolution Aspects",
        subtitle="Comparing Normalized 1–5 Mean Endorsement | Δ = Mean(FCT) − Mean(CRD) with Paired t-test Significance",
        height=440,
    )
    fig.update_xaxes(title_text="Normalized Endorsement Mean (1–5 Likert Scale)", range=[2.2, 4.4])
    fig.update_yaxes(title_text="Evolution Theoretical Aspect")
    return fig


def plot_likert_diverging_bars(likert_df: pd.DataFrame) -> Any:
    """
    48-Item Diverging Stacked Bar Chart displaying full 1–5 Likert response distributions
    (Strongly Disagree to Strongly Agree) styled across the Ocean Beach gradient
    (#003f5c to #ffa600), centered on Neutral (3).
    """
    fig = go.Figure()
    if likert_df.empty:
        return apply_ocean_beach_layout(fig, "48-Item Diverging Likert Distribution", "No items selected", 400)

    y_labels = likert_df["display_label"].tolist()

    p1 = likert_df["pct_1"].to_numpy()
    p2 = likert_df["pct_2"].to_numpy()
    p3 = likert_df["pct_3"].to_numpy()
    p4 = likert_df["pct_4"].to_numpy()

    base_1 = -(p1 + p2 + 0.5 * p3)
    base_2 = -(p2 + 0.5 * p3)
    base_3 = -0.5 * p3
    base_4 = 0.5 * p3
    base_5 = 0.5 * p3 + p4

    bases = {1: base_1, 2: base_2, 3: base_3, 4: base_4, 5: base_5}

    for k in [1, 2, 3, 4, 5]:
        pct_col = likert_df[f"pct_{k}"].to_numpy()
        cnt_col = likert_df[f"count_{k}"].to_numpy()
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=pct_col,
                base=bases[k],
                orientation="h",
                name=LIKERT_LABELS[k],
                marker=dict(color=LIKERT_COLORS[k], line=dict(width=0.5, color="#ffffff")),
                customdata=np.column_stack([
                    cnt_col,
                    pct_col,
                    likert_df["construct_label"],
                    likert_df["mean"],
                    likert_df["sd"],
                ]),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Response: <b>{LIKERT_LABELS[k]}</b><br>"
                    "Share: %{customdata[1]:.1f}% (n = %{customdata[0]})<br>"
                    "Construct: %{customdata[2]} | Item Mean: %{customdata[3]:.2f} (SD: %{customdata[4]:.2f})"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color=OCEAN_BEACH_PALETTE["deep_ocean_blue"])

    chart_height = max(460, len(likert_df) * 24 + 140)
    apply_ocean_beach_layout(
        fig,
        title="48-Item Diverging Likert Response Distribution (Centered on Neutral = 0%)",
        subtitle="Negative Left = Disagree (1–2) | Center = Neutral (3) | Positive Right = Agree (4–5) | Ocean Beach Gradient",
        height=chart_height,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(
        title_text="Diverging Response Percentage (% of Active Cohort)",
        ticksuffix="%",
        range=[-100, 100],
    )
    fig.update_yaxes(autorange="reversed", title_text="Survey Item & Aspect (TGcodes.csv)")
    return fig


def plot_science_internal_consistency(consistency_df: pd.DataFrame) -> Any:
    """
    Horizontal bar & dot chart displaying inter-item correlations (and Spearman-Brown /
    Cronbach's alpha) for the 13 multi-item Science aspects to detect item-pair
    divergence or reverse-wording polarity effects.
    """
    fig = go.Figure()

    bar_colors = [
        OCEAN_BEACH_PALETTE["sunset_crimson"]
        if r < 0
        else (OCEAN_BEACH_PALETTE["warm_tangerine"] if r < 0.15 else OCEAN_BEACH_PALETTE["deep_ocean_blue"])
        for r in consistency_df["mean_inter_item_r"]
    ]

    fig.add_trace(
        go.Bar(
            y=consistency_df["aspect_label"],
            x=consistency_df["mean_inter_item_r"],
            orientation="h",
            name="Mean Inter-Item Correlation (r)",
            marker=dict(color=bar_colors, line=dict(width=1, color="#ffffff")),
            text=[f"r = {v:+.2f}" for v in consistency_df["mean_inter_item_r"]],
            textposition="outside",
            customdata=np.column_stack([
                consistency_df["items"],
                consistency_df["cronbach_alpha"],
                consistency_df["spearman_brown"],
                consistency_df["pairwise_detail"],
                consistency_df["diagnostic_status"],
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Items: %{customdata[0]}<br>"
                "Mean Inter-Item r: %{x:.3f}<br>"
                "Cronbach's α: %{customdata[1]:.3f} | Spearman-Brown: %{customdata[2]:.3f}<br>"
                "Pairwise Breakdown: %{customdata[3]}<br>"
                "Diagnostic: <b>%{customdata[4]}</b><extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            y=consistency_df["aspect_label"],
            x=consistency_df["cronbach_alpha"],
            mode="markers",
            name="Cronbach's Alpha (α)",
            marker=dict(
                color=OCEAN_BEACH_PALETTE["golden_sun"],
                size=11,
                symbol="diamond",
                line=dict(width=1.5, color=OCEAN_BEACH_PALETTE["deep_ocean_blue"]),
            ),
            hovertemplate="<b>%{y}</b><br>Cronbach's α: %{x:.3f}<extra></extra>",
        )
    )

    fig.add_vline(x=0.0, line_width=2, line_dash="dash", line_color=OCEAN_BEACH_PALETTE["deep_ocean_blue"])
    fig.add_vline(x=0.15, line_width=1.5, line_dash="dot", line_color=OCEAN_BEACH_PALETTE["dusk_purple"])

    apply_ocean_beach_layout(
        fig,
        title="Science Aspect Internal Consistency & Polarity Diagnostic (13 Multi-Item Cognitive Aspects)",
        subtitle="Red (< 0) = Reverse-Wording / Polarity Divergence | Orange (0–.15) = Multi-Faceted | Deep Ocean (>.15) = Convergent",
        height=540,
    )
    fig.update_xaxes(title_text="Inter-Item Correlation (r) / Cronbach's Alpha (α)", range=[-0.95, 0.95])
    fig.update_yaxes(title_text="Science Cognitive Aspect (TGcodes.csv)")
    return fig


# ==============================================================================
# 5. TAB 4 CHARTS: DEMOGRAPHIC MODERATION & SUBGROUP PROFILES
# ==============================================================================
def plot_faceted_moderation_regressions(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    moderator: str,
    mod_res: Dict[str, Any],
) -> Any:
    """
    Small-multiple faceted regression plots comparing the Science -> Evolution slope
    across levels of the selected demographic moderator (Religiosity, Gender, or Age).
    """
    slopes = mod_res.get("subgroup_slopes", [])
    if not slopes:
        fig = go.Figure()
        return apply_ocean_beach_layout(fig, f"Faceted Moderation by {moderator}", "Insufficient subgroup data", 420)

    n_panels = len(slopes)
    subplot_titles = [
        f"{moderator}: {s['level']} (n={s['n']}, β={s['beta']:.2f}, r={s['r']:.2f})"
        if not pd.isna(s["beta"])
        else f"{moderator}: {s['level']} (n={s['n']})"
        for s in slopes
    ]

    fig = make_subplots(
        rows=1,
        cols=n_panels,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
    )

    color_map = MODERATOR_COLOR_MAPS.get(moderator, {})
    for idx, s in enumerate(slopes):
        col_idx = idx + 1
        lvl = s["level"]
        sub = df[df[moderator].astype(str) == lvl]
        color = color_map.get(lvl, OCEAN_BEACH_SEQUENCE[idx % len(OCEAN_BEACH_SEQUENCE)])

        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                name=f"{moderator}: {lvl} (n={len(sub)})",
                marker=dict(color=color, size=9, opacity=0.82, line=dict(width=1, color="#ffffff")),
                customdata=sub["ID"],
                hovertemplate=(
                    f"<b>ID: %{{customdata}}</b> ({moderator}: {lvl})<br>"
                    f"{x_label}: %{{x:.2f}}<br>"
                    f"{y_label}: %{{y:.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )

        if len(s["x_grid"]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([s["x_grid"], s["x_grid"][::-1]]),
                    y=np.concatenate([s["ci_high"], s["ci_low"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(249, 93, 106, 0.16)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=s["x_grid"],
                    y=s["y_hat"],
                    mode="lines",
                    name=f"{lvl} OLS (β={s['beta']:.2f}, p={s['p']:.3f})",
                    line=dict(color=OCEAN_BEACH_PALETTE["sunset_crimson"], width=3),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_xaxes(title_text=x_label, row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text=y_label, row=1, col=col_idx)

    subtitle = (
        f"OLS Interaction Term ({mod_res['interaction_term']}): "
        f"β_interaction = {mod_res['beta_interaction']:.3f}{mod_res['stars_interaction']} "
        f"(p = {mod_res['p_interaction']:.3f}) | Model F = {mod_res['f_stat']:.2f}, R² = {mod_res['r2']:.3f}"
        if mod_res.get("valid") and not pd.isna(mod_res.get("beta_interaction", np.nan))
        else "Subgroup OLS Comparison"
    )
    apply_ocean_beach_layout(
        fig,
        title=f"Faceted Moderation Regressions: {x_label} → {y_label} Moderated by {moderator}",
        subtitle=subtitle,
        height=550,
    )
    fig.update_layout(margin=dict(l=65, r=40, t=180, b=60))
    return fig


def plot_subgroup_distributions(
    df: pd.DataFrame,
    moderator: str = "Religiosity",
    code_to_label: Dict[str, str] | None = None,
) -> Any:
    """
    Subgroup Distribution Comparison (Violin + Box + Jitter) across SCI, FCT, and CRD
    on normalized 1–5 mean scales (or active scoring mode) across levels of Religiosity, Gender, or Age.
    """
    constructs = ["SCI", "FCT", "CRD"]
    titles = [
        code_to_label.get(c, c) if code_to_label else c
        for c in constructs
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.07,
    )

    levels = sorted(df[moderator].astype(str).unique())
    color_map = MODERATOR_COLOR_MAPS.get(moderator, {})

    for c_idx, c_code in enumerate(constructs):
        col_num = c_idx + 1
        for l_idx, lvl in enumerate(levels):
            sub = df[df[moderator].astype(str) == lvl]
            color = color_map.get(lvl, OCEAN_BEACH_SEQUENCE[l_idx % len(OCEAN_BEACH_SEQUENCE)])
            fig.add_trace(
                go.Violin(
                    x=[lvl] * len(sub),
                    y=sub[c_code],
                    name=f"{moderator}: {lvl}",
                    legendgroup=lvl,
                    showlegend=(c_idx == 0),
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    pointpos=-0.25,
                    jitter=0.25,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.72,
                    marker=dict(size=6, color=color, opacity=0.85),
                    customdata=sub["ID"],
                    hovertemplate=(
                        f"<b>ID: %{{customdata}}</b> ({moderator}: {lvl})<br>"
                        f"{titles[c_idx]}: %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_num,
            )
        fig.update_xaxes(title_text=moderator, row=1, col=col_num)
        if col_num == 1:
            fig.update_yaxes(title_text="Construct Score (Active Metric)", row=1, col=col_num)

    apply_ocean_beach_layout(
        fig,
        title=f"Subgroup Distribution Comparison (Violin + Box + Jitter): SCI, FCT, and CRD by {moderator}",
        subtitle="Displaying Subgroup Kernel Density, Interquartile Box, Mean Line, and Individual Respondent Points",
        height=560,
    )
    fig.update_layout(margin=dict(l=65, r=40, t=180, b=60))
    return fig


def plot_multi_aspect_radar(
    df: pd.DataFrame,
    code_to_label: Dict[str, str],
    moderator: str = "Religiosity",
) -> Any:
    """
    Multi-Aspect Radar Profile comparing demographic subgroups across all 14 Science
    aspects and 6 Evolution aspects simultaneously (on the 1-5 mean scale), labeled
    with TGcodes.csv meanings.
    """
    all_20_aspects = SCIENCE_ALL_ASPECTS + EVO_ALL_ASPECTS
    theta_labels = [
        f"[SCI] {code_to_label.get(a, a)}" if a in SCIENCE_ALL_ASPECTS else f"[EVO] {code_to_label.get(a, a)}"
        for a in all_20_aspects
    ]
    theta_closed = theta_labels + [theta_labels[0]]

    fig = go.Figure()
    levels = sorted(df[moderator].astype(str).unique())
    color_map = MODERATOR_COLOR_MAPS.get(moderator, {})

    for idx, lvl in enumerate(levels):
        sub = df[df[moderator].astype(str) == lvl]
        if sub.empty:
            continue
        r_vals = [float(sub[f"{a}_mean"].mean()) for a in all_20_aspects]
        r_closed = r_vals + [r_vals[0]]
        color = color_map.get(lvl, OCEAN_BEACH_SEQUENCE[idx % len(OCEAN_BEACH_SEQUENCE)])

        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill="toself",
                name=f"{moderator}: {lvl} (n={len(sub)})",
                line=dict(color=color, width=2.8),
                marker=dict(color=color, size=7),
                opacity=0.78,
                hovertemplate="<b>%{theta}</b><br>Subgroup Mean (1–5): %{r:.2f}<extra></extra>",
            )
        )

    apply_ocean_beach_layout(
        fig,
        title=f"20-Aspect Epistemic & Scientific Radar Profile by {moderator} (14 Science + 6 Evolution Aspects)",
        subtitle="Normalized 1–5 Likert Item Means Across All Theoretical Aspects (TGcodes.csv Labels)",
        height=710,
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True,
                range=[1.5, 4.8],
                tickvals=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                gridcolor="#d6e0e8",
                linecolor=UI_THEME["border_color"],
                tickfont=dict(size=10, color=UI_THEME["text_secondary"]),
            ),
            angularaxis=dict(
                gridcolor="#e9ecef",
                linecolor=UI_THEME["border_color"],
                tickfont=dict(size=11, color=UI_THEME["text_primary"]),
            ),
        ),
        margin=dict(l=65, r=40, t=180, b=60),
    )
    return fig
