# TASK: Psychometric & Epistemic BI Dashboard

## 1. Role & Objective
You are an expert Full-Stack Data Engineer and Psychometric Visualization Architect. Your objective is to build a modular, interactive Business Intelligence (BI) Dashboard in Python using **Streamlit** and **Plotly** to investigate the relationship between **Views on Science (`SCI`, `SKSI`, `ATS`)** and **Views on Evolution (`EVO`, split into Fact `FCT` and Credibility `CRD`)** across $N = 121$ respondents, moderated by 3 demographic variables (`Age`, `Gender`, and `Religiosity`).

---

## 2. Strict Guardrails & Operating Constraints
1. **READ-ONLY DATA GUARDRAIL:** You are strictly prohibited from modifying, overwriting, renaming, or deleting the existing input files (`2026data_rev.csv`, `TG.csv`, and `TGcodes.csv`) or any other existing repository files. Treat `2026data_rev.csv`, `TG.csv`, and `TGcodes.csv` strictly as read-only inputs.
2. **NO PLACEHOLDERS OR TRUNCATION:** Write complete, production-ready code across all files. Do not use `# TODO`, `pass`, or truncated code blocks.
3. **UI/UX THEME:** Style the dashboard and all Plotly charts using the **Ocean Beach Palette** (`https://pikka.app/palettes/ocean-beach-palette`):
   - **Core Data & Accent Palette:**
     - Deep Ocean Blue: `#003f5c`
     - Twilight Indigo: `#2f4b7c`
     - Dusk Purple: `#665191`
     - Berry Violet: `#a05195`
     - Coral Rose: `#d45087`
     - Sunset Crimson: `#f95d6a`
     - Warm Tangerine: `#ff7c43`
     - Golden Sun: `#ffa600`
   - **UI Surfaces & Typography:**
     - Background: `#f8f9fa` (Main Coastal Foam) and `#eef2f5` / `#fff9f0` (Warm Sand Cards/Sidebar)
     - Text: `#003f5c` (Primary Headers/Text) and `#2f4b7c` (Secondary/Muted Text)
     - Primary Analytical Accents: Deep Ocean (`#003f5c`, `#2f4b7c`) for `FCT` / cognitive constructs and Sunset Coral/Orange (`#f95d6a`, `#ff7c43`, `#ffa600`) for `CRD` / affective constructs.
4. **DEMOGRAPHIC VARIABLE EXCLUSION:** Completely exclude and drop the `Religion` column from all data processing, filtering, and visualizations. Do not derive `Religion_Group`. Stick strictly to `Religiosity` (`Religious` vs. `Non Religious`), `Gender`, and `Age`.

---

## 3. Input Data Schema & Theoretical Grounding

### 3.1 `2026data_rev.csv` ($N = 121$ rows, 57 columns)
- **Identifier:** `ID` (string, e.g., `LS2`, `MS24`).
- **Survey 1 Items (28 items, 1–5 Likert):** `S1` through `S15`, and `S17` through `S29` (Note: `S16` does not exist in the dataset).
- **Survey 2 Items (20 items, 1–5 Likert):** `E1` through `E20`.
- **Demographics (3 active variables; drop `Religion` on load):**
  - `Age`: integer (`14` to `18`).
  - `Gender`: string (`F`, `M`).
  - `Religiosity`: string (`Religious`, `Non Religious`).
- **Pre-computed Composites:**
  - `SKSI`: Sum of the 27 cognitive Science items (`S1`–`S15`, `S17`–`S28`).
  - `ATS`: Single item `S29` (Attitude Towards Science).
  - `SCI`: Total Science score (`SKSI` + `ATS`, 28 items).
  - `EVO`: Total Evolution score (sum of `E1`–`E20`, 20 items).

### 3.2 `TG.csv` (48 rows, 4 columns: `Item#`, `key`, `prj.3.0 Construct`, `prj.3.0 Aspect`)
Dynamically parse `TG.csv` (and cross-validate against the exact mappings below) to construct subscales and aspect scores:
- **Science Constructs & 14 Aspects:**
  - `SCI` (Cognitive constructs, 13 aspects):
    - `cns`: `['S1', 'S2']`
    - `prv`: `['S3', 'S4']`
    - `tsp`: `['S5', 'S6']`
    - `tlm`: `['S7', 'S8', 'S28']` *(3-item triad)*
    - `soc`: `['S9', 'S10']`
    - `com`: `['S11', 'S12']`
    - `app`: `['S13', 'S14']`
    - `cre`: `['S15', 'S17']`
    - `lin`: `['S18', 'S19']`
    - `cnt`: `['S20', 'S21']`
    - `tes`: `['S22', 'S23']`
    - `cmp`: `['S24', 'S25']`
    - `lmt`: `['S26', 'S27']`
  - `ATS` (Affective construct, 1 aspect):
    - `ats`: `['S29']`
- **Evolution Epistemic Constructs (20 items total):**
  - `FCT` (Fact-oriented items, 10 items): `['E1', 'E3', 'E8', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E20']`
  - `CRD` (Credibility-oriented items, 10 items): `['E2', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E14', 'E17', 'E19']`
- **Evolution Theoretical Aspects (6 populated aspects in `TG.csv`):**
  - `val` (5 items): `['E2', 'E10', 'E12', 'E13', 'E20']` -> Split: `val_FCT` (`E12, E13, E20`), `val_CRD` (`E2, E10`)
  - `pro` (4 items): `['E1', 'E9', 'E18', 'E19']` -> Split: `pro_FCT` (`E1, E18`), `pro_CRD` (`E9, E19`)
  - `evi` (4 items): `['E4', 'E6', 'E8', 'E16']` -> Split: `evi_FCT` (`E8, E16`), `evi_CRD` (`E4, E6`)
  - `bel` (3 items): `['E7', 'E11', 'E14']` -> Split: `bel_FCT` (`E11`), `bel_CRD` (`E7, E14`)
  - `hum` (2 items): `['E3', 'E15']` -> Pure `FCT` (`hum_FCT`)
  - `scc` (2 items): `['E5', 'E17']` -> Pure `CRD` (`scc_CRD`)

### 3.3 `TGcodes.csv` (Construct & Aspect Label Dictionary)
Dynamically parse `TGcodes.csv` (columns: `code`, `Type`, `Meaning`, dropping empty trailing rows/columns) to map shorthand codes to human-readable labels across all UI controls, axis titles, legends, and hover tooltips:
- **Questionnaires:**
  - `VNOS`: Views on the Nature of Science
  - `VOTE`: Views on the Theory of Evolution
- **Constructs:**
  - `SCI`: Science | `ATS`: Attitude Towards Science | `EVO`: Evolution
  - `NSK`: Nature of Scientific Knowledge | `NSI`: Nature of Scientific Inquiry | `NOS`: Nature of Science | `SI`: Scientific Inquiry
  - `FCT`: Fact | `CRD`: Credibility | `BLF`: Belief
  - *(Note: Map `SKSI` to "Scientific Knowledge & Inquiry (`SKSI`)" as the composite of `NSK` and `NSI`)*
- **Science Aspects (14 active in `TG.csv`):**
  - `cns`: Construction | `prv`: Provisionality | `tsp`: Theory support | `tlm`: Theory limits
  - `soc`: Society and Culture | `com`: Community | `app`: Applications | `cre`: Creativity
  - `lin`: Nonlinearity | `cnt`: Continuity | `tes`: Testing | `cmp`: Comparison
  - `lmt`: Limits of Science | `ats`: Attitudes
- **Evolution Aspects (6 active in `TG.csv`):**
  - `pro`: Process of evolution | `val`: Validity of theory | `hum`: Human evolution
  - `evi`: Evidence for evolution | `scc`: Scientific community | `bel`: Beliefs related to evolution

---

## 4. End-to-End Implementation Instructions

### Step 1: Project Architecture
Create a clean, modular codebase with the following directory structure:
```text
├── config.py              # Ocean Beach palette, item-to-aspect mappings, and TGcodes.csv label loader
├── data_engine.py         # Data loading (dropping Religion), dynamic TG.csv & TGcodes.csv parsing, scoring, and stats
├── visuals.py             # Reusable Plotly chart generators styled in the Ocean Beach theme
├── app.py                 # Main Streamlit dashboard interface (sidebar controls + 4 analytical tabs)
├── requirements.txt       # Required Python libraries (streamlit, plotly, pandas, numpy, scipy, statsmodels)
└── README.md              # Quick-start instructions to launch the dashboard

```

### Step 2: Data & Statistical Engine (`data_engine.py`)

1. Load `2026data_rev.csv`, `TG.csv`, and `TGcodes.csv` with `@st.cache_data`.
2. Explicitly drop the `Religion` column from the dataframe upon load and retain only `Age`, `Gender`, and `Religiosity` as demographic moderators.
3. Build a lookup dictionary from `TGcodes.csv` so every construct and aspect can be displayed as `"Meaning (code)"` (e.g., `"Provisionality (prv)"`, `"Credibility (CRD)"`, `"Validity of theory (val)"`).
4. Compute aspect scores under three user-selectable aggregation modes:
* **Item Mean (1–5 scale):** Recommended default so aspects with different item counts (e.g., `tlm` with 3 items, `val` with 5 items) are directly comparable on a 1–5 metric.
* **Raw Sum:** Direct summation of items within each construct/aspect.
* **Z-Score Standardized:** Mean = 0, SD = 1 across the active sample.


5. Implement statistical helper functions using `scipy.stats` and `statsmodels`:
* Pearson ($r$) and Spearman ($\rho$) correlation matrices with exact two-tailed $p$-values and significance stars (`* p<.05`, `** p<.01`, `*** p<.001`).
* OLS linear regression models returning $\beta$, intercept, $R^2$, and $p$-value, plus interaction term $p$-values ($X \times \text{Moderator}$) for demographic moderation analysis across `Gender`, `Religiosity`, and `Age`.
* Paired $t$-test / Wilcoxon signed-rank test comparing normalized `FCT` (Fact) vs. `CRD` (Credibility) scores within respondents.



### Step 3: Interactive Dashboard Layout (`app.py` & `visuals.py`)

#### A. Global Sidebar Controller

* **Cohort Slicers:** Multi-select filters for `Gender` (`F`, `M`), `Religiosity` (`Religious`, `Non Religious`), and an `Age` range slider (`14`–`18`). *(No `Religion` filter).*
* **Primary Science Predictor Selector:** Radio button using `TGcodes.csv` meanings: `SCI` (Science - Total), `SKSI` (Scientific Knowledge & Inquiry), or `ATS` (Attitude Towards Science).
* **Primary Evolution Criterion Selector:** Radio button using `TGcodes.csv` meanings: `EVO` (Evolution - Total), `FCT` (Fact Subscale), or `CRD` (Credibility Subscale).
* **Scoring Metric Toggle:** `Mean (1-5)`, `Raw Sum`, or `Z-Score`.
* **Correlation Method Toggle:** `Pearson (r)` vs. `Spearman (rho)`.

#### B. Tab 1: Macro Relationship & Epistemic Split (Executive Overview)

1. **KPI Metric Ribbon:** 5 cards showing Active $n / 121$, Primary Correlation ($r$ and $p$-value), Science Predictor Mean ($SD$), EVO `FCT` (Fact) Mean ($SD$), and EVO `CRD` (Credibility) Mean ($SD$) with paired difference $\Delta(\text{FCT} - \text{CRD})$ $p$-value.
2. **Joint Scatterplot with Marginal Distributions:** Plot the selected Science Predictor (X) vs. Evolution Criterion (Y), optional color-coding by `Religiosity`, `Gender`, or `Age` (using the Ocean Beach palette), OLS trendline, and marginal box/violin plots.
3. **Dual-Slope Epistemic Overlay Canvas:** Plot the Science Predictor on the X-axis and overlay **both** `FCT` (Deep Ocean `#003f5c`) and `CRD` (Sunset Crimson `#f95d6a`) on the Y-axis (normalized to a 1–5 mean scale) with separate OLS regression lines and 95% confidence bands to visually compare whether science views predict factual endorsement vs. perceived credibility at different rates.

#### C. Tab 2: Aspect-to-Aspect Cross-Correlation Matrix (Granular Explorer)

1. **Interactive Correlation Heatmap:**
* **Rows:** The 14 Science aspects labeled with full `TGcodes.csv` meanings (e.g., `Construction (cns)`, `Provisionality (prv)`, ..., `Attitudes (ats)`).
* **Columns:** Toggleable between:
* **Standard View (6 EVO Aspects):** `Validity of theory (val)`, `Process of evolution (pro)`, `Evidence for evolution (evi)`, `Beliefs related to evolution (bel)`, `Human evolution (hum)`, `Scientific community (scc)` (+ `FCT`, `CRD`, `EVO` summary columns).
* **Epistemic Split View (10 EVO Sub-Aspects):** `val_FCT`, `val_CRD`, `pro_FCT`, `pro_CRD`, `evi_FCT`, `evi_CRD`, `bel_FCT`, `bel_CRD`, `hum_FCT`, `scc_CRD` (with readable tooltips from `TGcodes.csv`).


* Color scale anchored to the Ocean Beach diverging/sequential stops (`#003f5c` $\to$ `#f8f9fa` $\to$ `#f95d6a` / `#ffa600`) and annotated with correlation coefficients and significance stars (`*`, `**`, `***`).


2. **Bivariate Aspect Drill-Down Inspector:** Provide dropdown selectors (formatted with `TGcodes.csv` meanings) for any Science Aspect $\times$ EVO Aspect pair to render a jittered scatterplot with OLS trendline, contingency density heatmap, and student `ID` hover tooltips.

#### D. Tab 3: Item Diagnostics, Polarity & Epistemic Gap (48-Item Deep Dive)

1. **Epistemic Divergence Dumbbell Plot:** Across the 4 overlapping EVO aspects (`val`, `pro`, `evi`, `bel` labeled via `TGcodes.csv`) plus Overall (`FCT` vs. `CRD`), plot a horizontal connected dumbbell chart comparing mean Fact (`FCT`, `#003f5c`) endorsement vs. mean Credibility (`CRD`, `#ff7c43`) endorsement, highlighting the exact epistemic gap per domain.
2. **48-Item Diverging Stacked Bar Chart:** Display full 1–5 Likert response distributions (Strongly Disagree to Strongly Agree, styled across the Ocean Beach gradient from `#003f5c` to `#ffa600`) across all 28 Science and 20 Evolution items, filterable by Construct (`SCI`, `ATS`, `FCT`, `CRD`) and sorted by aspect (`TGcodes.csv` meaning) or mean endorsement.
3. **Science Aspect Internal Consistency Diagnostic:** A horizontal bar/dot chart displaying inter-item correlations (and Spearman-Brown / Cronbach's $\alpha$) for the 13 multi-item Science aspects (labeled via `TGcodes.csv`) to detect item-pair divergence or reverse-wording effects.

#### E. Tab 4: Demographic Moderation & Subgroup Profiles

1. **Faceted Moderation Regressions:** Small-multiple regression plots comparing the Science $\to$ Evolution slope across levels of a selected demographic moderator (`Religiosity`, `Gender`, or `Age`), accompanied by an automated statistical readout of the OLS interaction term ($\beta_{\text{interaction}}, F, p$-value).
2. **Subgroup Distribution Comparison (Violin + Box + Jitter):** Compare `SCI`, `FCT`, and `CRD` distributions across `Religiosity` (`Religious` vs. `Non Religious`), `Gender` (`F` vs. `M`), or `Age` cohorts using Ocean Beach palette accents.
3. **Multi-Aspect Radar Profile:** An overlaid polar/radar chart comparing demographic subgroups (defaulting to `Religiosity`: `Religious` vs. `Non Religious`) across all 14 Science aspects and 6 Evolution aspects simultaneously, labeled with `TGcodes.csv` meanings.

---

## 5. Expected Deliverables & Verification Checklist

1. **Source Code Files:** Generate `config.py`, `data_engine.py`, `visuals.py`, `app.py`, `requirements.txt`, and `README.md`.
2. **Automated Verification:** Before finalizing, run a self-check script to verify:
* `2026data_rev.csv`, `TG.csv`, and `TGcodes.csv` remain completely untouched (verify via `git status` or checksum).
* `Religion` is completely excluded from the processed dataframe and UI, while `TGcodes.csv` meanings populate all construct and aspect labels.
* Composite reconstructions match the dataset: `SKSI + ATS == SCI`, `FCT + CRD == EVO`, and baseline full-sample correlations equal $r(\text{SCI}, \text{EVO}) = 0.6056$, $r(\text{SCI}, \text{FCT}) = 0.4675$, $r(\text{SCI}, \text{CRD}) = 0.5175$, and $r(\text{FCT}, \text{CRD}) = 0.2825$.
* All modules import cleanly with zero syntax or runtime errors.


3. **Packaged Deliverable:** Package all generated application files (`config.py`, `data_engine.py`, `visuals.py`, `app.py`, `requirements.txt`, `README.md`) into a single compressed archive named `bi_dashboard.zip`.
