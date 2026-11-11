# Psychometric & Epistemic Business Intelligence Dashboard

An interactive Python Business Intelligence (BI) Dashboard built with **Streamlit** and **Plotly** to investigate the relationship between **Views on the Nature of Science (`VNOS`: `SCI`, `SKSI`, `ATS`)** and **Views on the Theory of Evolution (`VOTE`: `EVO`, split into Fact `FCT` and Credibility `CRD`)** across $N = 121$ respondents, moderated by `Age`, `Gender`, and `Religiosity`.

Styled in the [Ocean Beach Palette](https://pikka.app/palettes/ocean-beach-palette) (`#003f5c`, `#2f4b7c`, `#665191`, `#a05195`, `#d45087`, `#f95d6a`, `#ff7c43`, `#ffa600`).

## Project Structure

```text
├── config.py              # Ocean Beach palette, item-to-aspect mappings, and TGcodes.csv label loader
├── data_engine.py         # Data loading (dropping Religion), dynamic TG.csv & TGcodes.csv parsing, scoring, and stats
├── visuals.py             # Reusable Plotly chart generators styled in the Ocean Beach theme
├── app.py                 # Main Streamlit dashboard interface (sidebar controls + 4 analytical tabs)
├── requirements.txt       # Required Python libraries (streamlit, plotly, pandas, numpy, scipy, statsmodels)
└── README.md              # Quick-start instructions to launch the dashboard
```

## Quick-Start Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Read-Only Input Data Files Are Present:**
   Place `2026data_rev.csv`, `TG.csv`, and `TGcodes.csv` in the same directory as `app.py`. The data engine treats all three CSV files strictly as read-only inputs and automatically drops the `Religion` column upon ingestion.

3. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## Key Analytical Capabilities

- **Global Sidebar Controller:** Cohort slicers for `Gender` (`F`, `M`), `Religiosity` (`Religious`, `Non Religious`), and `Age` (`14`–`18`), alongside toggles for Primary Science Predictor (`SCI`, `SKSI`, `ATS`), Primary Evolution Criterion (`EVO`, `FCT`, `CRD`), Scoring Metric (`Mean (1-5)`, `Raw Sum`, `Z-Score`), and Correlation Method (`Pearson (r)`, `Spearman (rho)`).
- **Tab 1 — Macro Relationship & Epistemic Split:** 5-card KPI ribbon, Joint Scatterplot with Marginal Box/Violin Distributions & OLS confidence band, and Dual-Slope Epistemic Overlay Canvas comparing `FCT` (`#003f5c`) vs. `CRD` (`#f95d6a`) regression slopes on a normalized 1–5 scale.
- **Tab 2 — Aspect-to-Aspect Cross-Correlation Matrix:** Toggleable correlation heatmap between the 14 Science aspects and either the 6 Evolution aspects (`Standard View`) or the 10 Epistemic Split sub-aspects (`Epistemic Split View`), paired with a Bivariate Aspect Drill-Down Inspector (jittered scatterplot + 2D contingency density surface).
- **Tab 3 — Item Diagnostics, Polarity & Epistemic Gap:** Epistemic Divergence Dumbbell Plot across `val`, `pro`, `evi`, `bel`, and Overall (`FCT` vs. `CRD`), 48-Item Diverging Stacked Likert Bar Chart, and Science Aspect Internal Consistency Diagnostic (`Cronbach's α`, `Spearman-Brown`, and inter-item $r$).
- **Tab 4 — Demographic Moderation & Subgroup Profiles:** Small-multiple faceted moderation regressions with automated OLS interaction term ($X \times \text{Moderator}$) statistics, Subgroup Violin + Box + Jitter distributions, and a 20-Aspect Polar Radar Profile.
