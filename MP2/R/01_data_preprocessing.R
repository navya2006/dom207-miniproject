# =============================================================================
# DOM207-DOM3007-DOM6401 : MP2
# Stage 1 (R) - Data processing for the HCL Technologies IT employee file.
#
# Reads  : data/ITEmp.xlsx  (sheet "Salary")
# Writes : ITEmp_Cleaned_R.csv
#          R_outputs/cleaning_log.csv
#
# This is an independent R implementation of notebooks/01_data_processing.ipynb.
# The two pipelines apply identical cleaning rules and should produce the same
# analytic sample (526 rows).  The Python pipeline is the primary one; this R
# version exists so the work can be reproduced in either language.
#
# Run:  Rscript R/01_data_preprocessing.R      (from the MP2/ directory)
# =============================================================================

suppressPackageStartupMessages(library(readxl))

root <- normalizePath(".")
raw_path <- file.path(root, "data", "ITEmp.xlsx")
out_dir <- file.path(root, "R_outputs")
dir.create(out_dir, showWarnings = FALSE)

DV <- "sal"
BINARY_VARS <- c("nonprev", "female", "married", "metro", "norcen", "south",
                 "west", "itreq", "itprog", "itqual", "itarch", "itai",
                 "itclient", "itrd", "ittech", "wrkhm")
CONTINUOUS_VARS <- c("sal", "edu", "exper", "ten", "dep", "momed", "daded", "sib")
ALL_VARS <- c(DV, CONTINUOUS_VARS[-1], BINARY_VARS)

VALID_RANGE <- list(
  sal   = c(0, 100), edu   = c(1, 25),  exper = c(0, 60), ten = c(0, 60),
  dep   = c(0, 10),  momed = c(0, 25),  daded = c(0, 25), sib = c(0, 10)
)

raw <- as.data.frame(read_excel(raw_path, sheet = "Salary"))
cat("Raw data:", nrow(raw), "rows x", ncol(raw), "columns\n")

# ---- 1. coercion: anything non-numeric becomes NA -------------------------
clean <- raw
for (v in ALL_VARS) clean[[v]] <- suppressWarnings(as.numeric(raw[[v]]))

# ---- 2. logical range checks ---------------------------------------------
log_rows <- list()
for (v in ALL_VARS) {
  raw_missing  <- sum(is.na(raw[[v]]))
  non_numeric  <- sum(is.na(clean[[v]])) - raw_missing
  before       <- sum(is.na(clean[[v]]))
  if (v %in% names(VALID_RANGE)) {
    lo <- VALID_RANGE[[v]][1]; hi <- VALID_RANGE[[v]][2]
    clean[[v]][!is.na(clean[[v]]) & (clean[[v]] < lo | clean[[v]] > hi)] <- NA
  }
  if (v %in% BINARY_VARS) {
    clean[[v]][!is.na(clean[[v]]) & !(clean[[v]] %in% c(0, 1))] <- NA
  }
  out_of_range <- sum(is.na(clean[[v]])) - before
  log_rows[[v]] <- data.frame(variable = v,
                              role = ifelse(v == DV, "dependent",
                                            ifelse(v %in% BINARY_VARS,
                                                   "binary", "continuous")),
                              raw_missing = raw_missing,
                              non_numeric_errors = max(non_numeric, 0),
                              out_of_range_errors = max(out_of_range, 0),
                              total_set_to_missing = sum(is.na(clean[[v]])))
}
cleaning_log <- do.call(rbind, log_rows)
write.csv(cleaning_log, file.path(out_dir, "cleaning_log.csv"), row.names = FALSE)

# ---- 3. imputation: median for continuous, mode for binary ----------------
mode_of <- function(x) {
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}
for (v in CONTINUOUS_VARS) clean[[v]][is.na(clean[[v]])] <- median(clean[[v]], na.rm = TRUE)
for (v in BINARY_VARS)     clean[[v]][is.na(clean[[v]])] <- mode_of(clean[[v]])

# ---- 4. derived variables -------------------------------------------------
clean$log_sal   <- log(clean$sal)
clean$exper_sq  <- clean$exper^2
clean$ten_sq    <- clean$ten^2
clean$edu_sq    <- clean$edu^2
clean <- clean[order(clean$id), ]
rownames(clean) <- NULL

write.csv(clean, file.path(root, "ITEmp_Cleaned_R.csv"), row.names = FALSE)

cat("Cleaning log written to R_outputs/cleaning_log.csv\n")
cat("Non-numeric errors :", sum(cleaning_log$non_numeric_errors), "\n")
cat("Out-of-range errors:", sum(cleaning_log$out_of_range_errors), "\n")
cat("Rows cleaned       :", nrow(clean), "\n")
cat("Written            : ITEmp_Cleaned_R.csv\n")
