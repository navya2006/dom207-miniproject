# =============================================================================
# DOM207-DOM3007-DOM6401 : MP2
# Stage 3 (R) - Hypotheses, regression modelling and diagnostics.
#
# Reads  : ITEmp_Cleaned_R.csv
# Writes : R_outputs/regression_summary_R.txt
#          R_outputs/table_model_comparison_R.csv
#          R_outputs/table_vif_R.csv
#          R_outputs/table_diagnostics_R.csv
#          R_outputs/table_hypotheses_R.csv
#          R_outputs/table_main_models_R.txt
#
# Mirrors notebooks/03_hypothesis_regression.ipynb: centred regressors, quadratic
# and interaction effects, VIF, Breusch-Pagan, RESET, robust (HC3) inference and
# a battery of alternative specifications.
#
# Run:  Rscript R/03_analysis.R      (from the MP2/ directory)
# =============================================================================

suppressPackageStartupMessages({
  library(car)
  library(lmtest)
  library(sandwich)
})

root <- normalizePath(".")
out_dir <- file.path(root, "R_outputs")
dir.create(out_dir, showWarnings = FALSE)

# round only the numeric columns of a data frame
round_df <- function(d, k = 3) {
  num <- sapply(d, is.numeric)
  d[num] <- lapply(d[num], round, digits = k)
  d
}

df <- read.csv(file.path(root, "ITEmp_Cleaned_R.csv"))

# ---- mean-centre the continuous regressors used in squares / interactions ---
CENTER <- c("edu", "exper", "ten", "dep", "momed", "daded", "sib")
for (v in CENTER) df[[paste0("c_", v)]] <- df[[v]] - mean(df[[v]])
for (v in c("exper", "ten", "edu")) df[[paste0("c_", v, "_sq")]] <- df[[paste0("c_", v)]]^2

HC    <- "c_edu + c_exper + c_ten"
DEMO  <- "female + married + nonprev + c_dep"
GEO   <- "metro + norcen + south + west"
JOB   <- "itreq + itprog + itqual + itarch + itrd + ittech + itai + itclient + wrkhm"
FAM   <- "c_momed + c_daded + c_sib"
QUAD  <- "c_exper_sq + c_ten_sq + c_edu_sq"
INTER <- "c_edu:female + c_exper:c_edu + married:female + metro:c_edu"

formulas <- list(
  "M1 Human capital"     = as.formula(paste("log_sal ~", HC)),
  "M2 + Demographics"    = as.formula(paste("log_sal ~", HC, "+", DEMO)),
  "M3 + Geography"       = as.formula(paste("log_sal ~", HC, "+", DEMO, "+", GEO)),
  "M4 + Job function"    = as.formula(paste("log_sal ~", HC, "+", DEMO, "+", GEO, "+", JOB)),
  "M5 Full main effects" = as.formula(paste("log_sal ~", HC, "+", DEMO, "+", GEO, "+", JOB, "+", FAM)),
  "M6 + Quadratics"      = as.formula(paste("log_sal ~", HC, "+", DEMO, "+", GEO, "+", JOB, "+", FAM, "+", QUAD)),
  "M7 + Interactions"    = as.formula(paste("log_sal ~", HC, "+", DEMO, "+", GEO, "+", JOB, "+", FAM, "+", QUAD, "+", INTER))
)
PREFERRED <- "M7 + Interactions"

models <- lapply(formulas, function(f) lm(f, data = df))

# ---- model comparison -------------------------------------------------------
comp <- do.call(rbind, lapply(names(models), function(nm) {
  m <- models[[nm]]
  s <- summary(m)
  data.frame(model = nm, k_params = length(coef(m)), n = nobs(m),
             r_squared = s$r.squared, adj_r_squared = s$adj.r.squared,
             aic = AIC(m), bic = BIC(m))
}))
write.csv(round_df(comp, 4), file.path(out_dir, "table_model_comparison_R.csv"), row.names = FALSE)

# ---- main coefficient table (M1-M7) ----------------------------------------
main_tab <- stargazer <- NULL
labels <- c()
for (nm in names(models)) {
  m <- models[[nm]]
  labels <- union(labels, names(coef(m)))
}
mat <- matrix("", nrow = length(labels) * 2, ncol = length(models))
colnames(mat) <- names(models)
rownames(mat) <- as.vector(rbind(labels, rep("", length(labels))))
for (j in seq_along(models)) {
  m <- models[[j]]; s <- summary(m)$coefficients
  for (i in seq_along(labels)) {
    v <- labels[i]
    if (v %in% rownames(s)) {
      p <- s[v, 4]
      st <- ifelse(p < .01, "***", ifelse(p < .05, "**", ifelse(p < .10, "*", "")))
      mat[2 * i - 1, j] <- sprintf("%.4f%s", s[v, 1], st)
      mat[2 * i, j]     <- sprintf("(%.4f)", s[v, 2])
    }
  }
}
write.csv(mat, file.path(out_dir, "table_main_models_R.csv"), row.names = TRUE)
capture.output(print(mat, quote = FALSE),
               file = file.path(out_dir, "table_main_models_R.txt"))

# ---- preferred model: VIF, heteroskedasticity, RESET, robust SE -------------
pref <- models[[PREFERRED]]

vif_vals <- suppressMessages(vif(pref))
vif_df <- data.frame(variable = names(vif_vals), VIF = as.numeric(vif_vals))
vif_df$flag <- ifelse(vif_df$VIF > 10, "severe (>10)",
                      ifelse(vif_df$VIF > 5, "moderate (>5)", "ok"))
vif_df <- vif_df[order(-vif_df$VIF), ]
write.csv(round_df(vif_df, 3), file.path(out_dir, "table_vif_R.csv"), row.names = FALSE)

bp   <- bptest(pref)
rr   <- resettest(pref, power = 2, type = "fitted")
# Jarque-Bera statistic computed directly: n/6 * (skew^2 + (kurtosis-3)^2/4)
res_pref <- residuals(pref)
n_pref <- length(res_pref)
m2 <- mean((res_pref - mean(res_pref))^2)
skew <- mean((res_pref - mean(res_pref))^3) / m2^1.5
kurt <- mean((res_pref - mean(res_pref))^4) / m2^2
jb_stat <- n_pref / 6 * (skew^2 + (kurt - 3)^2 / 4)
jb_p <- 1 - pchisq(jb_stat, df = 2)
cooks <- cooks.distance(pref)

diag_df <- data.frame(
  test = c("Breusch-Pagan (heteroskedasticity)",
           "Ramsey RESET (functional form)",
           "Jarque-Bera (residual normality)",
           "Max Cook's distance"),
  statistic = c(unname(bp$statistic), unname(rr$statistic), jb_stat, max(cooks)),
  p_value = c(bp$p.value, rr$p.value, jb_p, NA),
  conclusion = c(ifelse(bp$p.value > .05, "homoskedastic", "heteroskedastic"),
                 ifelse(rr$p.value > .05, "correctly specified", "misspecified"),
                 ifelse(jb_p > .05, "residuals normal", "residuals non-normal"),
                 ifelse(max(cooks) < 1, "no dominating observation",
                        "an observation is highly influential"))
)
write.csv(diag_df, file.path(out_dir, "table_diagnostics_R.csv"), row.names = FALSE)

hc3 <- coeftest(pref, vcov = vcovHC(pref, type = "HC3"))

# ---- hypotheses -------------------------------------------------------------
hyp_one <- function(hid, statement, term, expected) {
  b <- coef(pref)[term]; se <- hc3[term, 2]; p <- hc3[term, 4]
  sign_ok <- if (expected == "+") b > 0 else b < 0
  v <- if (p < .05 && sign_ok) "Supported" else
       if (p < .05 && !sign_ok) "Significant, opposite sign" else
       if (p < .10 && sign_ok) "Weakly supported" else "Not supported"
  data.frame(H = hid, hypothesis = statement, term = term, estimate = b,
             robust_se = se, p_value = p, verdict = v)
}
hyp <- rbind(
  hyp_one("H1",  "Education raises pay", "c_edu", "+"),
  hyp_one("H2",  "Experience raises pay with diminishing returns", "c_exper", "+"),
  hyp_one("H2b", "The experience profile is concave", "c_exper_sq", "-"),
  hyp_one("H3",  "Firm tenure raises pay", "c_ten", "+"),
  hyp_one("H4",  "Women earn less than comparable men", "female", "-"),
  hyp_one("H5",  "Married employees earn a premium", "married", "+"),
  hyp_one("H6",  "Metro employees earn a premium", "metro", "+"),
  hyp_one("H8",  "Non-privileged employees earn less", "nonprev", "-"),
  hyp_one("H9",  "Working from home lowers pay", "wrkhm", "-"),
  hyp_one("H9b", "AI-exposed work lowers pay", "itai", "-"),
  hyp_one("H10", "Mother's education matters", "c_momed", "+"),
  hyp_one("H11", "Education premium differs by gender", "c_edu:female", "-"),
  hyp_one("H12", "Education and experience are complements", "c_edu:c_exper", "+"),
  hyp_one("H13", "Marriage premium differs by gender", "female:married", "-")
)
write.csv(round_df(hyp, 4), file.path(out_dir, "table_hypotheses_R.csv"), row.names = FALSE)

# ---- alternative specifications --------------------------------------------
lo <- quantile(df$sal, .01); hi <- quantile(df$sal, .99)
alt <- list(
  "M8  HC3 robust SE (same point estimates)" = pref,
  "M9  DV in levels" = lm(update(formulas[[PREFERRED]], sal ~ .), data = df),
  "M11 1% trimmed salary" = lm(formulas[[PREFERRED]], data = df[df$sal >= lo & df$sal <= hi, ])
)

# ---- report text ------------------------------------------------------------
sink(file.path(out_dir, "regression_summary_R.txt"))
cat("DOM207 MP2 - R regression summary\n")
cat(strrep("=", 70), "\n")
print(summary(pref))
cat("\n--- HC3 robust coefficient test (preferred model) ---\n")
print(hc3)
cat("\n--- Model comparison ---\n"); print(comp, row.names = FALSE)
cat("\n--- VIF ---\n"); print(vif_df, row.names = FALSE)
cat("\n--- Diagnostics ---\n"); print(diag_df, row.names = FALSE)
cat("\n--- Hypotheses ---\n"); print(hyp, row.names = FALSE)
cat("\n--- Alternative specifications (adjusted R-squared) ---\n")
for (nm in names(alt)) cat(sprintf("%-45s adj-R2 = %.4f\n", nm, summary(alt[[nm]])$adj.r.squared))
sink()

cat("Preferred model M7: R2 =", round(summary(pref)$r.squared, 3),
    " adj-R2 =", round(summary(pref)$adj.r.squared, 3),
    " n =", nobs(pref), "\n")
cat("\nHypotheses:\n"); print(hyp[, c("H", "term", "estimate", "p_value", "verdict")],
                              row.names = FALSE)
cat("\nDiagnostics:\n"); print(diag_df, row.names = FALSE)
cat("\nR stage 3 complete. Outputs in R_outputs/\n")
