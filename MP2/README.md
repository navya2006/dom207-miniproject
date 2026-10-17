# DOM207 MP2 — Determinants of Employee Compensation at HCL Technologies

An econometric analysis of the salary packages of 526 employees in the `ITEmp.xlsx`
extract, done for the DOM207 / DOM3007 / DOM6401 mini-project. The client brief asks for
an evaluation of what drives pay, with explicit hypotheses, quadratic and interaction
effects, assumption checks, and supplementary analysis of model specification,
multicollinearity and heteroskedasticity.

The formal write-up is generated as **`MP2_Report.docx`** (title page, Introduction,
Data Processing, Data Visualization, Hypotheses, Analysis, Recommendations, References).
This repository holds the analysis — notebooks in Python, scripts in R — together with
the figures and the output tables that the report draws on. The confidential source
extract and the course brief are deliberately not published (see *A note on the data*).

## Pipeline

The Python analysis is delivered as three Jupyter notebooks in `notebooks/`, with the
narrative written out and every output — including the figures — saved in the file.
Run them in order, **from inside `notebooks/`**:

| Notebook | Produces |
|---|---|
| `01_data_processing.ipynb` | `ITEmp_Cleaned.csv`, `ITEmp_CompleteCase.csv`, `output_tables/cleaning_log.csv` |
| `02_eda_visualisations.ipynb` | `report_charts/01..07*.png`, descriptive and group tables |
| `03_hypothesis_regression.ipynb` | `output_tables/table_*.csv`, `report_charts/08..09*.png` |

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

The notebooks are the source of truth for the Python analysis; open them with
`../.venv/bin/jupyter lab` to step through interactively. (Each one is self-contained
and carries its executed outputs, so they also read as documents.) The finished report,
`MP2_Report.docx`, is committed as-is.

### Independent R replication

The same analysis is implemented in R and reproduces the Python results exactly
(identical coefficients, Breusch-Pagan 32.30, RESET 6.86, Jarque-Bera 948.48):

```bash
Rscript R/01_data_preprocessing.R     # -> ITEmp_Cleaned_R.csv
Rscript R/02_eda_visualizations.R     # -> R_outputs/r_*.png
Rscript R/03_analysis.R               # -> R_outputs/table_*.csv
```

R outputs live in `R_outputs/` and never overwrite the Python artefacts.

## Headline findings

Estimated on log average hourly salary; preferred specification M7, HC3 robust
standard errors, n = 526, adjusted R² = 0.495.

| Effect | Estimate | p |
|---|---|---|
| Education, per year (at mean) | +7.0% | <0.001 |
| Experience, per year (at mean; profile peaks at ~26 yrs) | +0.85% | <0.001 |
| Tenure, per year (at mean) | +1.9% | <0.001 |
| Metro location | +14.7% | 0.004 |
| Women vs comparable men, unmarried | −12.8% | 0.035 |
| Women vs comparable men, married | −29.6% | 0.012 |
| Married men vs unmarried men | +17.5% | 0.011 |
| IT R&D division | +16.3% | 0.006 |
| IT architecture | −25.2% | <0.001 |
| AI-exposed IT work | −26.8% | <0.001 |

Not supported: non-privileged status, work-from-home, family background
(parental education, siblings), and the education × gender interaction.

## What the diagnostics found

- **Multicollinearity** — the maximal VIF is 6.0. Before mean-centring the polynomial
  and interaction terms, education and its square had VIFs above 80; centring removes
  that purely structural collinearity and changes nothing substantive.
- **Heteroskedasticity** — not present (Breusch-Pagan p = 0.354, White/Wooldridge
  p = 0.202). HC3 robust errors are reported anyway and make almost no difference.
- **Normality** — rejected (Jarque-Bera p < 0.001); the Q-Q plot shows fat tails rather
  than skew. With n = 526 the central limit theorem applies and robust errors are used.
- **Specification** — RESET rejects (p = 0.009). Investigated rather than assumed away:
  partial-residual tests find no unmodelled curvature in any continuous regressor, the
  rejection survives dropping influential observations, and a flexible gradient-boosting
  benchmark is *worse* out-of-sample than the linear model (5-fold CV R² 0.362 vs 0.458
  for the preferred design). It is treated as a symptom of fat tails and handled with
  robust inference plus four alternative specifications.
- **Influence** — the largest Cook's distance is 0.096; no observation dominates.

## Data dictionary

| Variable | Meaning |
|---|---|
| `sal` | Average hourly salary (DV; modelled in logarithms) |
| `edu` | Years of formal education |
| `exper` | Years of professional experience |
| `ten` | Years of work in the current company |
| `nonprev` | 1 if a non-privileged employee |
| `female`, `married` | 1 if female / married |
| `dep` | Number of dependents in the immediate family |
| `momed`, `daded`, `sib` | Mother's education, father's education, number of siblings |
| `metro` | 1 if lives in a metro area |
| `norcen`, `south`, `west` | Province dummies (eastern province is the reference) |
| `itreq`, `itprog`, `itqual`, `itarch` | IT work-group indicators |
| `itai`, `itclient`, `itrd`, `ittech` | AI-exposed work, client-only work, R&D, tech support |
| `wrkhm` | 1 if the employee works primarily from home |

## A note on the data

`data/ITEmp.xlsx` is confidential employee data supplied only for this assignment. The
raw file contains 29 invalid cells (text in numeric fields such as `SS0`, `aaa`, `acb`;
and impossible values such as `sal = 1111`, `edu = 1001`, `sib = 333`) which the cleaning
stage sets to missing and then imputes; the audit trail is in
`output_tables/cleaning_log.csv`.

Neither the extract nor the course brief is published: both are listed in `.gitignore`.
The pipeline expects the workbook to be present locally at `data/ITEmp.xlsx`.

## Requirements

Python: see `requirements.txt`. R: `readxl`, `ggplot2`, `dplyr`, `tidyr`, `car`,
`lmtest`, `sandwich`.
