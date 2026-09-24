#!/usr/bin/env python
# coding: utf-8

# # DOM207 MP1 — Statistical Analysis, Hypothesis Testing & Business Recommendations
# 
# **Dataset:** `Restaurant_Cleaned.csv` (357 rows after cleaning)
# 
# This notebook runs the hypothesis tests for the report's Analysis section, then carries the findings that actually hold up into the Discussion and Recommendations section.
# 
# ### Questions tested
# - **HT1 — Lunch vs Dinner:** does the bill amount differ by meal period?
# - **HT2 — Smoker vs Non-smoker:** does the bill amount differ by smoking status?
# - **HT3 — Party size vs Bill amount:** is there a monotonic relationship?
# - **HT4 — Bill amount vs Tip:** is there a relationship?
# - **HT5 — Day × Time:** are day of week and meal period associated?
# - **HT6 (bonus) — Amount across Days:** does bill amount differ across the 4 days?
# - **Bonus — Tip% by waiter gender:** does tip percentage differ by the waiter's gender?
# 
# ### Test choice
# `amount` and `tip` are both right-skewed (see the Shapiro-Wilk/Levene checks before each test), so a rank-based test (Mann-Whitney U / Kruskal-Wallis / Spearman) is used as the primary result for every comparison below. Welch's t-test / Pearson is still reported alongside for magnitude and a confidence interval, but the rank-based p-value is what the decision is based on. Because of that, the hypotheses are written in terms of distributions rather than means — a rank-based test doesn't actually test whether two means are equal.
# 
# Significance level for all tests: **α = 0.05**.
# 

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import itertools
from IPython.display import display  # so this prints sensibly outside Jupyter too

pd.set_option("display.precision", 4)
ALPHA = 0.05

def fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ## 1. Load data
# 
# Reloaded independently so this notebook runs standalone; derived columns (`tip_pct`, `spend_per_person`) and day ordering match the EDA notebook.

# In[2]:


df = pd.read_csv("Restaurant_Cleaned.csv")

day_order = ["thursday", "friday", "saturday", "sunday"]
df["day"] = pd.Categorical(df["day"].str.lower(), categories=day_order, ordered=True)
df["smoker"] = df["smoker"].astype(bool)
df["tip_pct"] = (df["tip"] / df["amount"]) * 100
df["spend_per_person"] = df["amount"] / df["partysize"]

print("Shape:", df.shape)
display(df.head())

display(pd.DataFrame({
    "Metric": ["Total observations", "Mean bill", "Median bill", "Mean tip"],
    "Value": [len(df), df["amount"].mean(), df["amount"].median(), df["tip"].mean()],
}))


# ## 2. Assumption-checking helper
# 
# Runs Shapiro-Wilk per group and Levene's test across groups, so the choice between a parametric and non-parametric test is justified by the data rather than assumed.

# In[3]:


def check_assumptions(groups: dict, label: str, verbose=True):
    """groups: {name: pd.Series}. Returns (all_normal, equal_var)."""
    if verbose:
        print(f"--- Assumption check: {label} ---")
    normal_flags = []
    for name, vals in groups.items():
        stat, p = stats.shapiro(vals)
        normal = p > ALPHA
        normal_flags.append(normal)
        if verbose:
            print(f"  Shapiro-Wilk [{name}] (n={len(vals)}): W={stat:.4f}, p={fmt_p(p)} -> {'normal' if normal else 'NOT normal'}")
    lev_stat, lev_p = stats.levene(*groups.values())
    equal_var = lev_p > ALPHA
    if verbose:
        print(f"  Levene's test (equal variance): stat={lev_stat:.4f}, p={fmt_p(lev_p)} -> "
              f"{'equal variance' if equal_var else 'UNEQUAL variance'}")
    return all(normal_flags), equal_var

def welch_ci(a, b, confidence=0.95):
    """CI for (mean(a) - mean(b)) using Welch-Satterthwaite df."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = ma - mb
    se = np.sqrt(va / na + vb / nb)
    df_w = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    crit = stats.t.ppf((1 + confidence) / 2, df_w)
    return diff, diff - crit * se, diff + crit * se

def rank_biserial(u_stat, n1, n2):
    """Effect size for Mann-Whitney U; ranges -1..1."""
    return 1 - (2 * u_stat) / (n1 * n2)

def cohend(a, b):
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_sd

def group_desc(groups: dict, val_label="Bill"):
    return pd.DataFrame({
        "Group": list(groups.keys()),
        "N": [len(v) for v in groups.values()],
        f"Mean {val_label}": [v.mean() for v in groups.values()],
        f"Median {val_label}": [v.median() for v in groups.values()],
        "Std Dev": [v.std() for v in groups.values()],
    })

results = []  # summary rows for the final table


# ## HT1 — Lunch vs Dinner (bill amount)
# 
# **H0:** the bill amount distribution is the same for lunch and dinner
# **H1:** the bill amount distribution differs between lunch and dinner
# 
# Business relevance: staffing and pricing decisions by meal period.

# In[4]:


lunch = df.loc[df.time == "lunch", "amount"]
dinner = df.loc[df.time == "dinner", "amount"]

display(group_desc({"Lunch": lunch, "Dinner": dinner}, "Bill"))

all_normal, equal_var = check_assumptions({"lunch": lunch, "dinner": dinner}, "amount by time")

# Parametric (Welch) — reported regardless, for magnitude/CI, but NOT the deciding test since data is non-normal
t_stat, t_p = stats.ttest_ind(lunch, dinner, equal_var=False)
diff, ci_lo, ci_hi = welch_ci(lunch, dinner)
print(f"\n[Reference] Welch t-test: t={t_stat:.4f}, p={fmt_p(t_p)}, "
      f"mean diff (lunch-dinner)={diff:.2f}, 95% CI=({ci_lo:.2f}, {ci_hi:.2f})")

# Non-parametric — primary decision, since Shapiro rejected normality for both groups
u_stat, u_p = stats.mannwhitneyu(lunch, dinner, alternative="two-sided")
effect = rank_biserial(u_stat, len(lunch), len(dinner))
decision = "Reject H0" if u_p < ALPHA else "Fail to reject H0"
print(f"[PRIMARY] Mann-Whitney U: U={u_stat:.1f}, p={fmt_p(u_p)}, rank-biserial r={effect:.3f}")
print(f"Decision (alpha={ALPHA}): {decision}")
agree = "agrees with" if (t_p < ALPHA) == (u_p < ALPHA) else "DISAGREES with"
print(f"Note: Welch t-test {agree} the Mann-Whitney decision.")

results.append(dict(test="HT1: Amount ~ Time", method="Mann-Whitney U (primary)", statistic=u_stat,
                     p_value=u_p, effect_size=effect, effect_label="rank-biserial r", decision=decision,
                     reference=f"Welch t p={fmt_p(t_p)}"))


# ### HT1 interpretation
# Dinner's sample mean is higher than lunch's, but neither test finds the difference statistically significant at α=0.05, and the Welch 95% CI for the mean difference includes zero. That's not enough on its own to justify a dinner-only pricing or expansion push.

# ## HT2 — Smoker vs Non-smoker (bill amount)
# 
# **H0:** the bill amount distribution is the same for smokers and non-smokers
# **H1:** the bill amount distribution differs between smokers and non-smokers

# In[5]:


smoker_amt = df.loc[df.smoker == True, "amount"]
nonsmoker_amt = df.loc[df.smoker == False, "amount"]

display(group_desc({"Smoker": smoker_amt, "Non-Smoker": nonsmoker_amt}, "Bill"))

all_normal, equal_var = check_assumptions({"smoker": smoker_amt, "non-smoker": nonsmoker_amt}, "amount by smoker")

t_stat, t_p = stats.ttest_ind(smoker_amt, nonsmoker_amt, equal_var=False)
diff, ci_lo, ci_hi = welch_ci(smoker_amt, nonsmoker_amt)
print(f"\n[Reference] Welch t-test: t={t_stat:.4f}, p={fmt_p(t_p)}, "
      f"mean diff (smoker-non-smoker)={diff:.2f}, 95% CI=({ci_lo:.2f}, {ci_hi:.2f})")

u_stat, u_p = stats.mannwhitneyu(smoker_amt, nonsmoker_amt, alternative="two-sided")
effect = rank_biserial(u_stat, len(smoker_amt), len(nonsmoker_amt))
decision = "Reject H0" if u_p < ALPHA else "Fail to reject H0"
print(f"[PRIMARY] Mann-Whitney U: U={u_stat:.1f}, p={fmt_p(u_p)}, rank-biserial r={effect:.3f}")
print(f"Decision (alpha={ALPHA}): {decision}")
agree = "agrees with" if (t_p < ALPHA) == (u_p < ALPHA) else "DISAGREES with"
print(f"Note: Welch t-test {agree} the Mann-Whitney decision (p={fmt_p(t_p)} vs p={fmt_p(u_p)} — "
      f"they can diverge like this when variances differ a lot and the data are skewed, which is why the "
      f"assumption checks above matter for picking which one to trust).")

results.append(dict(test="HT2: Amount ~ Smoker", method="Mann-Whitney U (primary)", statistic=u_stat,
                     p_value=u_p, effect_size=effect, effect_label="rank-biserial r", decision=decision,
                     reference=f"Welch t p={fmt_p(t_p)}"))


# ### HT2 interpretation
# Smoker and non-smoker bills have noticeably different variance, and neither group is close to normal — exactly when Welch's t-test and Mann-Whitney can disagree. Mann-Whitney (rank-biserial r close to 0, p ≈ 0.75) is the one to trust here given the skew, and it doesn't show a real bill-amount difference by smoking status. **Business implication:** don't segment pricing or marketing by smoking status on this basis alone.

# ## HT3 — Party size vs Bill amount
# 
# **H0:** ρ(partysize, amount) = 0   **H1:** ρ(partysize, amount) ≠ 0
# 
# Party size is discrete and the relationship need not be linear, so Spearman is primary; Pearson is reported too.

# In[6]:


pearson_r3, pearson_p3 = stats.pearsonr(df["partysize"], df["amount"])
spearman_r3, spearman_p3 = stats.spearmanr(df["partysize"], df["amount"])

display(pd.DataFrame({"Method": ["Pearson", "Spearman (primary)"],
                       "Correlation": [pearson_r3, spearman_r3],
                       "p-value": [fmt_p(pearson_p3), fmt_p(spearman_p3)]}))

decision = "Reject H0" if spearman_p3 < ALPHA else "Fail to reject H0"
print(f"Decision (alpha={ALPHA}, based on Spearman): {decision}")
if abs(spearman_r3 - pearson_r3) > 0.15:
    print("Note: Pearson (linear) is notably weaker than Spearman (monotonic) here -> the relationship is "
          "real but non-linear, not merely 'weak'. Worth showing a scatterplot with a trend line in the report.")

results.append(dict(test="HT3: Partysize vs Amount", method="Spearman (primary)", statistic=spearman_r3,
                     p_value=spearman_p3, effect_size=spearman_r3, effect_label="rho", decision=decision,
                     reference=f"Pearson r={pearson_r3:.3f}"))


# ### HT3 interpretation
# Bigger parties tend to run up bigger bills — real, but an association rather than proof that party size itself causes the higher bill. **Business implication:** factor the expected mix of party sizes into capacity and seating decisions.

# ## HT4 — Bill amount vs Tip
# 
# **H0:** ρ(amount, tip) = 0   **H1:** ρ(amount, tip) ≠ 0

# In[7]:


pearson_r4, pearson_p4 = stats.pearsonr(df["amount"], df["tip"])
spearman_r4, spearman_p4 = stats.spearmanr(df["amount"], df["tip"])

display(pd.DataFrame({"Method": ["Pearson", "Spearman"],
                       "Correlation": [pearson_r4, spearman_r4],
                       "p-value": [fmt_p(pearson_p4), fmt_p(spearman_p4)]}))

decision = "Reject H0" if spearman_p4 < ALPHA else "Fail to reject H0"
print(f"Decision (alpha={ALPHA}, based on Spearman — amount is non-normal, see HT1/HT2 checks): {decision}")

results.append(dict(test="HT4: Amount vs Tip", method="Spearman (primary)", statistic=spearman_r4,
                     p_value=spearman_p4, effect_size=spearman_r4, effect_label="rho", decision=decision,
                     reference=f"Pearson r={pearson_r4:.3f}"))


# ### HT4 interpretation
# Bill amount and tip move together closely, and both Pearson and Spearman pick up a clear positive relationship. Spearman is noticeably higher than Pearson (0.67 vs 0.54), which means the relationship is better described as monotonic than as strictly linear. Either way, bill volume is a reasonable stand-in for tip volume when forecasting service-staff income — it's still correlation, not causation.

# ## HT5 — Day × Time (Chi-square test of independence)
# 
# **H0:** day of week and meal period are independent   **H1:** they are associated
# 
# Business relevance: staffing, table allocation, and inventory planning by shift.

# In[8]:


# 'day' was imputed from 'time' for 12 rows (groupby('time') mode fill), so
# testing day against time on the full sample is partly circular. Run the
# primary test on complete cases only, and keep the full sample as a
# sensitivity check.
complete = df.loc[~df["day_imputed"]]

contingency = pd.crosstab(complete["day"], complete["time"])
print(f"Observed counts (complete cases, day not imputed, n={len(complete)}):")
display(contingency)

chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
print("Expected counts under independence:")
display(expected_df.round(2))

n = contingency.sum().sum()
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt((chi2 / n) / min_dim)

decision = "Reject H0" if chi_p < ALPHA else "Fail to reject H0"
print(f"\n[PRIMARY, complete cases] Chi-square = {chi2:.4f}, dof = {dof}, p = {fmt_p(chi_p)}")
print(f"Cramer's V (effect size) = {cramers_v:.3f}")
print(f"Decision (alpha={ALPHA}): {decision}")

full_ct = pd.crosstab(df["day"], df["time"])
chi2_full, chi_p_full, dof_full, _ = stats.chi2_contingency(full_ct)
cramers_v_full = np.sqrt((chi2_full / len(df)) / min_dim)
print(f"\n[Sensitivity, full sample incl. imputed day, n={len(df)}] "
      f"Chi-square = {chi2_full:.4f}, p = {fmt_p(chi_p_full)}, Cramer's V = {cramers_v_full:.3f}")

results.append(dict(test="HT5: Day vs Time", method="Chi-square (complete cases)", statistic=chi2, p_value=chi_p,
                     effect_size=cramers_v, effect_label="Cramer's V", decision=decision,
                     reference=f"full sample p={fmt_p(chi_p_full)}"))


# ### HT5 interpretation
# Day and meal period are strongly linked (Cramer's V ≈ 0.56 on complete cases, 0.57 on the full sample — the imputed rows barely move it): Thursday is mostly a lunch crowd, while Saturday and Sunday are mostly dinner. This is a scheduling pattern, not evidence that one meal period is more profitable than the other (see HT1). **Business implication:** plan staffing, table allocation, and inventory by day-and-shift instead of treating every lunch or every dinner shift the same.

# ## HT6 (bonus) — Bill amount across all 4 Days
# 
# **H0:** the bill amount distribution is the same across Thursday/Friday/Saturday/Sunday
# **H1:** at least one day's distribution differs
# 
# This adds a direct day-level amount comparison on top of HT5's day × time association.

# In[9]:


# same complete-cases treatment as HT5, for the same reason (day was imputed
# from time for 12 rows)
complete = df.loc[~df["day_imputed"]]
day_groups = {name: sub["amount"].values for name, sub in complete.groupby("day", observed=True)}

all_normal, equal_var = check_assumptions(
    {k: pd.Series(v) for k, v in day_groups.items()}, "amount by day (complete cases)"
)

if all_normal and equal_var:
    stat_val, kw_p = stats.f_oneway(*day_groups.values())
    omnibus_name = "One-way ANOVA"
    eps_sq = None
else:
    stat_val, kw_p = stats.kruskal(*day_groups.values())
    omnibus_name = "Kruskal-Wallis"
    k, n_total = len(day_groups), sum(len(v) for v in day_groups.values())
    # this is epsilon-squared, the standard Kruskal-Wallis effect size
    # (H - k + 1) / (n - k); NOT eta-squared, which is an ANOVA statistic
    eps_sq = (stat_val - k + 1) / (n_total - k)

decision = "Reject H0" if kw_p < ALPHA else "Fail to reject H0"
print(f"\n[PRIMARY, complete cases, n={sum(len(v) for v in day_groups.values())}] "
      f"{omnibus_name}: statistic={stat_val:.4f}, p={fmt_p(kw_p)}")
if eps_sq is not None:
    print(f"epsilon-squared (effect size) = {eps_sq:.3f}")
print(f"Decision (alpha={ALPHA}): {decision}")

results.append(dict(test="HT6 (bonus): Amount ~ Day", method=omnibus_name + " (complete cases)", statistic=stat_val,
                     p_value=kw_p, effect_size=eps_sq, effect_label="epsilon-squared", decision=decision, reference="-"))

if kw_p < ALPHA:
    print("\nPost-hoc pairwise Mann-Whitney (Bonferroni-corrected, complete cases):")
    pairs = list(itertools.combinations(day_groups.keys(), 2))
    bonf_alpha = ALPHA / len(pairs)
    for d1, d2 in pairs:
        u, p = stats.mannwhitneyu(day_groups[d1], day_groups[d2], alternative="two-sided")
        sig = "significant" if p < bonf_alpha else "not significant"
        print(f"  {d1} vs {d2}: U={u:.1f}, p={fmt_p(p)} -> {sig} (adj. alpha={bonf_alpha:.4g})")


# ### HT6 interpretation
# On complete cases, Kruskal-Wallis still finds a significant difference in bill amount across days, and after Bonferroni correction only Thursday vs Sunday holds up — the effect size (epsilon-squared ≈ 0.03) is small either way. Worth treating as a mild, secondary signal next to HT5's much stronger Day × Time scheduling pattern, not as a standalone pricing driver.

# ## Bonus — Tip percentage by waiter gender
# 
# Not required by the assignment, but it's a natural staffing question the fields can answer: does tip percentage differ by the waiter's gender? (`gender` here is the waiter's gender per the data dictionary, not the customer's.)
# 
# **H0:** the tip-percentage distribution is the same for male and female waiters
# **H1:** it differs by waiter gender

# In[10]:


male_tip = df.loc[df.gender == "male", "tip_pct"]
female_tip = df.loc[df.gender == "female", "tip_pct"]

display(group_desc({"Male waiter": male_tip, "Female waiter": female_tip}, "Tip %"))

all_normal, equal_var = check_assumptions({"male": male_tip, "female": female_tip}, "tip_pct by gender")

t_stat, t_p = stats.ttest_ind(male_tip, female_tip, equal_var=equal_var)
u_stat, u_p = stats.mannwhitneyu(male_tip, female_tip, alternative="two-sided")
effect = rank_biserial(u_stat, len(male_tip), len(female_tip))
decision = "Reject H0" if u_p < ALPHA else "Fail to reject H0"
print(f"[Reference] t-test: p={fmt_p(t_p)}   |   [PRIMARY] Mann-Whitney: U={u_stat:.1f}, p={fmt_p(u_p)}, "
      f"rank-biserial r={effect:.3f}")
print(f"Decision (alpha={ALPHA}): {decision}")

results.append(dict(test="Bonus: Tip% ~ Waiter Gender", method="Mann-Whitney U (primary)", statistic=u_stat,
                     p_value=u_p, effect_size=effect, effect_label="rank-biserial r", decision=decision,
                     reference=f"t-test p={fmt_p(t_p)}"))


# ### Bonus interpretation
# No significant difference in tip percentage by waiter gender — staffing/scheduling decisions don't need to account for this.

# ## Overall Statistical Findings (summary table)
# 
# Ready to paste into the report's Analysis section.

# In[11]:


summary = pd.DataFrame(results)
summary["p_value"] = summary["p_value"].map(fmt_p)
summary["statistic"] = summary["statistic"].map(lambda x: f"{x:.4f}")
summary["effect_size"] = summary["effect_size"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
summary


# ## Business Recommendations
# 
# Each recommendation below is tied to a specific finding above, not speculation.
# 
# ### 1. Flexible small-table seating, not a bet on big groups
# HT3 shows party size has a real, positive relationship with total *bill* amount (Spearman ρ ≈ 0.53) — bigger groups run up bigger tabs. But spend *per person* actually goes the other way (Spearman ρ ≈ -0.27, bigger groups spend less per head), and the vast majority of parties in this data are small: 94.7% are size 4 or under, 97.2% are size 6 or under. Betting floor space on rare large-party observations (some of which — a 200-person party on an $18 bill — look like data errors more than real bookings) isn't well supported. The better-supported move is flexible 2- and 4-person tables that can be pushed together for the occasional group of 5-6, rather than permanently dedicating space to large-party seating.
# 
# ### 2. Use bill volume to forecast tip income at new sites
# Bill amount and tip are strongly, positively associated (HT4: Pearson r ≈ 0.54, Spearman ρ ≈ 0.66). For staffing and pay planning at a new location, expected bill volume is a reasonable stand-in for expected tip income when projecting waitstaff take-home pay — it's a correlation to plan around, not a guarantee.
# 
# ### 3. Staff and schedule new sites around the Day × Time pattern, not a blanket lunch/dinner split
# Day of week and meal period are strongly associated (HT5: Cramer's V ≈ 0.56-0.57) — Thursdays skew lunch-heavy, weekends skew dinner-heavy — with a small but real day-to-day difference in average bill, Thursday vs Sunday (HT6). An expansion pilot should plan staffing, table allocation, and inventory around day-and-shift combinations rather than assuming every lunch or every dinner shift looks the same. Treat this as an operational hypothesis worth validating against actual demand at a new site: the data only tells us about transaction counts here, not true footfall or turned-away customers.
# 
# ### 4. Don't base expansion targeting on meal period or smoking status
# The lunch-vs-dinner and smoker-vs-non-smoker bill differences are not statistically significant at α = 0.05 (HT1, HT2), under both the primary rank-based test and the reference parametric test. The current sample doesn't support building a new location's concept, pricing, or marketing around either split.
# 
# ### Overall
# The best-supported expansion levers here are flexible small-table seating and day/shift-aware staffing — both show real, statistically supported effects, and both are about how a new site should be laid out and run rather than who it should target. Meal period and smoking status don't show a meaningful difference and shouldn't drive expansion decisions on their own; the large-party and big-spender framing sounds appealing but isn't what the data actually shows once per-person spend and sample size are accounted for.
# 

# ## Limitations
# 
# - Findings are based only on the cleaned dataset supplied for this assignment (357 of 365 original rows, after dropping/imputing invalid records — see the preprocessing notebook).
# - Statistical significance describes association, not causation.
# - Several fields (`tip`, `partysize`, `day`) were partly imputed (KNN / stratified mode) during cleaning; this analysis inherits whatever uncertainty that introduced. HT5/HT6 re-run their primary test on complete cases for `day` specifically, since day was imputed from time and testing day against time on the full sample would otherwise be somewhat circular.
# - Rows where `tip > amount` (8 rows) were dropped as likely data-entry errors — several were extreme (e.g. a $300+ tip on a ~$22 bill) but not all of them necessarily were, since the data dictionary doesn't rule out a tip exceeding the bill. `tip` and `partysize` were also capped at their 99th percentile rather than removed outright. Both choices are judgment calls, not certainties, and the report should say so rather than presenting them as the only correct cleaning approach.
# - The Day × Time association and the Thursday/Sunday amount difference describe *this* sample and should be validated against a larger or more recent time window before being used for major operational or expansion decisions.
# - All tests use α = 0.05; "fail to reject H0" means the sample doesn't provide sufficient evidence of a difference — it does not prove the groups are identical.
# 
