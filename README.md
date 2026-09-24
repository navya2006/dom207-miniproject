# DOM207 MP1 — Foodie India Restaurant Analysis

Data cleaning, exploratory analysis, and hypothesis testing on `Restaurant.xlsx`, done for
the DOM207 mini-project. The goal is to recommend viable expansion propositions for Foodie
India based only on the supplied data — see `MP1Specs` for the full assignment brief.

The formal write-up (title page, Introduction/Preprocessing/Visualization/Analysis/
Discussion/References) is submitted as a separate Word/PDF report; this repo holds the
underlying data and analysis code that report draws on.

## Pipeline

Run in this order, from a clean checkout, no Colab required:

1. **`data_preprocessing_nb.ipynb`** — loads `Restaurant.csv`, fixes typos in the
   categorical fields, coerces the numeric fields, drops rows where `tip > amount`,
   KNN-imputes remaining missing values, caps `tip`/`partysize` at their 99th percentile,
   and writes `Restaurant_Cleaned.csv` (357 rows).
2. **`DOM207_MP1_eda_visualizations.ipynb`** — descriptive stats, normality checks, and the
   charts in `report_charts/`.
3. **`DOM207_MP1_hypothesis_testing_FINAL.ipynb`** — the 7 hypothesis tests (with assumption
   checks, effect sizes, and Bonferroni-corrected post-hocs) and the business recommendations
   drawn from them.

## Data dictionary

| Field | Meaning |
|---|---|
| `amount` | Total bill for a party at a single meal |
| `tip` | Tip given to the **waiter** who served the party |
| `gender` | Gender of the **waiter** who served the party (not the customer) |
| `smoker` | Whether anyone in the party smoked |
| `day` | Day of the meal — only `thursday` / `friday` / `saturday` / `sunday` occur in this data |
| `time` | `lunch` or `dinner` |
| `partysize` | Number of people in the party |

## Requirements

```
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
```
