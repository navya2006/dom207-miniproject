# =============================================================================
# DOM207-DOM3007-DOM6401 : MP2
# Stage 2 (R) - Exploratory analysis and visualisations.
#
# Reads  : ITEmp_Cleaned_R.csv
# Writes : R_outputs/r_*.png  and  R_outputs/descriptive_stats.csv
#
# Run:  Rscript R/02_eda_visualizations.R      (from the MP2/ directory)
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(tidyr)
  library(dplyr)
})

root <- normalizePath(".")
out_dir <- file.path(root, "R_outputs")
dir.create(out_dir, showWarnings = FALSE)

df <- read.csv(file.path(root, "ITEmp_Cleaned_R.csv"))
DV <- "sal"
BINARY_VARS <- c("nonprev", "female", "married", "metro", "norcen", "south",
                 "west", "itreq", "itprog", "itqual", "itarch", "itai",
                 "itclient", "itrd", "ittech", "wrkhm")
CONTINUOUS_VARS <- c("sal", "edu", "exper", "ten", "dep", "momed", "daded", "sib")

theme_set(theme_minimal(base_size = 12))

# ---- descriptive statistics -------------------------------------------------
desc <- data.frame(
  variable = c(CONTINUOUS_VARS, BINARY_VARS),
  mean     = sapply(c(CONTINUOUS_VARS, BINARY_VARS), function(v) mean(df[[v]])),
  sd       = sapply(c(CONTINUOUS_VARS, BINARY_VARS), function(v) sd(df[[v]])),
  min      = sapply(c(CONTINUOUS_VARS, BINARY_VARS), function(v) min(df[[v]])),
  median   = sapply(c(CONTINUOUS_VARS, BINARY_VARS), function(v) median(df[[v]])),
  max      = sapply(c(CONTINUOUS_VARS, BINARY_VARS), function(v) max(df[[v]]))
)
desc$skew <- sapply(desc$variable, function(v) {
  x <- df[[v]]; m <- mean(x); s <- sd(x); mean((x - m)^3) / s^3
})
desc[, -1] <- lapply(desc[, -1], function(x) round(x, 3))
write.csv(desc, file.path(out_dir, "descriptive_stats.csv"), row.names = FALSE)

# ---- 1. salary distribution -------------------------------------------------
p <- ggplot(df, aes(x = sal)) +
  geom_histogram(bins = 30, fill = "#2b6cb0", colour = "white") +
  labs(title = "Figure 1a. Average hourly salary (raw)", x = "USD per hour", y = "Count")
ggsave(file.path(out_dir, "r_01_salary_raw.png"), p, width = 7, height = 4.5, dpi = 150)

p <- ggplot(df, aes(x = log_sal)) +
  geom_histogram(bins = 30, fill = "#2f855a", colour = "white") +
  labs(title = "Figure 1b. log(average hourly salary)", x = "log USD per hour", y = "Count")
ggsave(file.path(out_dir, "r_02_salary_log.png"), p, width = 7, height = 4.5, dpi = 150)

# ---- 2. correlation heatmap -------------------------------------------------
cols <- c("log_sal", "edu", "exper", "ten", "dep", "momed", "daded", "sib",
          "female", "married", "nonprev", "metro", "wrkhm", "itai", "itrd", "ittech")
cm <- round(cor(df[, cols]), 2)
cm_long <- as.data.frame(as.table(cm))
p <- ggplot(cm_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq), size = 2.6) +
  scale_fill_gradient2(low = "#2c5282", mid = "white", high = "#9b2c2c",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Figure 2. Correlation matrix", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave(file.path(out_dir, "r_03_correlation.png"), p, width = 8.5, height = 7, dpi = 150)

# ---- 3. salary by function --------------------------------------------------
df$workgroup <- with(df, ifelse(itreq == 1, "Requirements",
                         ifelse(itprog == 1, "Programming",
                         ifelse(itqual == 1, "Quality ass.",
                         ifelse(itarch == 1, "Architecture", "Other / reference")))))
df$workgroup <- factor(df$workgroup,
                       levels = c("Requirements", "Programming", "Quality ass.",
                                  "Architecture", "Other / reference"))
p <- ggplot(df, aes(workgroup, sal, fill = workgroup)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Figure 3. Salary by IT work-group", x = "", y = "USD per hour") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave(file.path(out_dir, "r_04_salary_by_function.png"), p, width = 7.5, height = 5, dpi = 150)

# ---- 4. salary by region ----------------------------------------------------
df$province <- with(df, ifelse(norcen == 1, "North-central",
                        ifelse(south == 1, "South",
                        ifelse(west == 1, "West", "East (reference)"))))
df$province <- factor(df$province, levels = c("East (reference)", "North-central",
                                              "South", "West"))
p <- ggplot(df, aes(province, sal, fill = province)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Figure 4. Salary by province", x = "", y = "USD per hour")
ggsave(file.path(out_dir, "r_05_salary_by_province.png"), p, width = 7.5, height = 5, dpi = 150)

# ---- 5. human-capital profiles (linear + quadratic) -------------------------
hc <- rbind(
  data.frame(x = df$edu,   y = df$sal, var = "Years of education"),
  data.frame(x = df$exper, y = df$sal, var = "Years of experience"),
  data.frame(x = df$ten,   y = df$sal, var = "Years of tenure")
)
p <- ggplot(hc, aes(x, y)) +
  geom_point(alpha = 0.35, size = 0.9) +
  geom_smooth(method = "lm", formula = y ~ x, colour = "#c53030", se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), colour = "#2b6cb0",
              linetype = "dashed", se = FALSE) +
  facet_wrap(~var, scales = "free_x") +
  labs(title = "Figure 5. Salary vs human capital (solid linear, dashed quadratic)",
       x = "", y = "USD per hour")
ggsave(file.path(out_dir, "r_06_human_capital.png"), p, width = 11, height = 4.5, dpi = 150)

# ---- 6. group comparisons ---------------------------------------------------
grp <- rbind(
  data.frame(group = "Gender",       level = ifelse(df$female == 1, "Female", "Male"),          sal = df$sal),
  data.frame(group = "Marital",      level = ifelse(df$married == 1, "Married", "Unmarried"),  sal = df$sal),
  data.frame(group = "Work mode",    level = ifelse(df$wrkhm == 1, "Home", "On-site"),         sal = df$sal),
  data.frame(group = "AI exposure",  level = ifelse(df$itai == 1, "AI work", "No AI work"),    sal = df$sal)
)
p <- ggplot(grp, aes(level, sal, fill = level)) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~group, scales = "free_x") +
  labs(title = "Figure 6. Salary across demographic and work-mode groups",
       x = "", y = "USD per hour")
ggsave(file.path(out_dir, "r_07_group_comparisons.png"), p, width = 11, height = 4.5, dpi = 150)

# ---- 7. experience vs tenure ------------------------------------------------
p <- ggplot(df, aes(exper, ten)) +
  geom_point(alpha = 0.4, size = 1, colour = "#2b6cb0") +
  labs(title = "Figure 7. Experience vs tenure", x = "Years of experience",
       y = "Years of tenure")
ggsave(file.path(out_dir, "r_08_experience_tenure.png"), p, width = 6.5, height = 4.5, dpi = 150)

cat("R stage 2 complete. Charts in R_outputs/\n")
