import pandas as pd
import numpy as np
import os


# 1. LOAD DATASET
FILE_PATH = "Restaurant.xlsx"
df = pd.read_excel(FILE_PATH)
print("1. DATASET LOADED")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())



# 2. UNDERSTAND THE DATASET

print("2. DATASET OVERVIEW")
print("\nNumber of observations:", df.shape[0])
print("Number of variables:", df.shape[1])

print("\nData types:")
print(df.dtypes)

print("\nFirst 10 rows:")
print(df.head(10))



# 3. IDENTIFY SPREADSHEET ARTIFACT COLUMNS
print("3. INVESTIGATION OF UNNAMED COLUMNS")

unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]

print("\nUnnamed columns:")
print(unnamed_cols)

for col in unnamed_cols:
    print(f"\n{col}:")
    print("Unique values:", df[col].unique())
    print("Non-null values:", df[col].notna().sum())


# 4. UNIQUE VALUES
print("4. UNIQUE VALUES")
for col in df.columns:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))


# 5. MISSING VALUES
print("5. MISSING VALUE ANALYSIS")

missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().sum() / len(df) * 100).round(2)
})

print(missing)

missing.to_csv("missing_value_analysis.csv")


# 6. DUPLICATE RECORDS

print("6. DUPLICATE RECORD ANALYSIS")
duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)

if duplicate_count > 0:
    print("\nDuplicate records:")
    print(df[df.duplicated(keep=False)].sort_values(
        by=df.columns.tolist()
    ))



# 7. INVESTIGATE CATEGORICAL VARIABLES
print("7. CATEGORICAL VARIABLES")

categorical_columns = [
    "Gender",
    "Smoker",
    "Day",
    "Time"
]

for col in categorical_columns:
    print("\n" + "-" * 50)
    print(col)
    print("-" * 50)
    print(df[col].value_counts(dropna=False))



# 8. CONVERT NUMERICAL VARIABLES
print("8. NUMERICAL VARIABLE CONVERSION")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["Tip"] = pd.to_numeric(df["Tip"], errors="coerce")
df["Partysize"] = pd.to_numeric(df["Partysize"], errors="coerce")

print("\nData types after conversion:")
print(df[["Amount", "Tip", "Partysize"]].dtypes)



# 9. INVESTIGATE INVALID NUMERICAL VALUES
print("9. INVALID NUMERICAL VALUES")

print("\nAmount values <= 0:")
print(df[df["Amount"] <= 0][["Amount", "Tip", "Partysize"]])

print("\nTip values < 0:")
print(df[df["Tip"] < 0][["Amount", "Tip", "Partysize"]])

print("\nUnusually large Tip values:")
print(df[df["Tip"] > 50][["Amount", "Tip", "Partysize"]])

print("\nUnusual Partysize values:")
print(
    df[
        (df["Partysize"] <= 0) |
        (df["Partysize"] > 20)
    ][["Amount", "Tip", "Partysize"]]
)


# Gender cleaning
gender_mapping = {
    "M": "Male",
    "Mal": "Male",
    "Mle": "Male",

    "F": "Female",
    "Fe": "Female",
    "Fem": "Female",
    "Femle": "Female",
    "Fmle": "Female"
}

df["Gender"] = df["Gender"].replace(gender_mapping)


# Smoker
smoker_mapping = {
    "N": "No",
    "Y": "Yes",
    "s": "Yes"
}

df["Smoker"] = df["Smoker"].replace(smoker_mapping)


# Day
day_mapping = {
    "Saturday": "Sat",
    "Friday": "Fri",
    "Thurs": "Thur",
    "Th": "Thur",
    "Trhurs": "Thur",

    "S": "Sun",
    "SS": "Sun",
    "SSS": "Sun",
    "San": "Sun",
    "Sn": "Sun",
    "sun": "Sun",

    "T": "Thur",
    "Ft": "Fri"
}

df["Day"] = df["Day"].replace(day_mapping)


# Time
time_mapping = {
    "L": "Lunch",
    "LD": "Lunch",
    "Lan": "Lunch",
    "Lu": "Lunch",

    "D": "Dinner",
    "DD": "Dinner",
    "DDD": "Dinner",
    "Di": "Dinner",
    "Din": "Dinner",
    "Diner": "Dinner"
}

df["Time"] = df["Time"].replace(time_mapping)



# 11. RECHECK CATEGORICAL VALUES
print("11. CATEGORICAL VALUES AFTER CLEANING")
for col in categorical_columns:
    print("\n", col)
    print(df[col].value_counts(dropna=False))


# ------------------------------------------------------------
# 12. HANDLE CLEARLY INVALID NUMERICAL VALUES
# ------------------------------------------------------------

# These values are clearly invalid for the variables:
#
# Amount:
#   <= 0 is invalid for a restaurant bill.
#
# Tip:
#   negative values are invalid.
#
# Partysize:
#   <= 0 is invalid.
#   Extremely large values are treated as invalid observations.
#
# We convert these values to NaN rather than inventing replacement
# values.

df.loc[df["Amount"] <= 0, "Amount"] = np.nan
df.loc[df["Tip"] < 0, "Tip"] = np.nan

df.loc[
    (df["Partysize"] <= 0) |
    (df["Partysize"] > 20),
    "Partysize"
] = np.nan


# ------------------------------------------------------------
# 13. DERIVED VARIABLE - TIP PERCENTAGE
# ------------------------------------------------------------

df["Tip_Percentage"] = (
    df["Tip"] / df["Amount"]
) * 100


# ------------------------------------------------------------
# 14. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("14. DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_variables = [
    "Amount",
    "Tip",
    "Partysize"
]

descriptive_stats = df[numeric_variables].describe().T

descriptive_stats = descriptive_stats[
    [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ]
]

descriptive_stats = descriptive_stats.rename(
    columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Std_Dev",
        "min": "Minimum",
        "25%": "Q1",
        "50%": "Median",
        "75%": "Q3",
        "max": "Maximum"
    }
)

descriptive_stats = descriptive_stats.round(2)

print(descriptive_stats)

descriptive_stats.to_csv(
    "descriptive_statistics.csv"
)


# ------------------------------------------------------------
# 15. TIP PERCENTAGE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("15. TIP PERCENTAGE")
print("=" * 70)

print(
    df["Tip_Percentage"].describe().round(2)
)


# ------------------------------------------------------------
# 16. FINAL MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("16. FINAL MISSING VALUE ANALYSIS")
print("=" * 70)

final_missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage":
        (df.isnull().sum() / len(df) * 100).round(2)
})

print(final_missing)

final_missing.to_csv(
    "final_missing_value_analysis.csv"
)


# ------------------------------------------------------------
# 17. FINAL DUPLICATE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("17. FINAL DUPLICATE CHECK")
print("=" * 70)

print(
    "Duplicate records after cleaning:",
    df.duplicated().sum()
)


# ------------------------------------------------------------
# 18. DATA DICTIONARY
# ------------------------------------------------------------

data_dictionary = pd.DataFrame({
    "Variable": [
        "Amount",
        "Tip",
        "Gender",
        "Smoker",
        "Day",
        "Time",
        "Partysize",
        "Tip_Percentage"
    ],

    "Type": [
        "Numeric",
        "Numeric",
        "Categorical",
        "Categorical",
        "Categorical",
        "Categorical",
        "Numeric",
        "Numeric"
    ],

    "Description": [
        "Total bill amount for a party at a single meal",
        "Tip amount given to the waiter",
        "Gender of the waiter serving the party",
        "Whether anyone in the party smoked",
        "Day of the meal",
        "Time of the meal",
        "Number of people in the party",
        "Tip as a percentage of the total bill"
    ],

    "Role": [
        "Original variable",
        "Original variable",
        "Original variable",
        "Original variable",
        "Original variable",
        "Original variable",
        "Original variable",
        "Derived variable"
    ]
})

print("\n" + "=" * 70)
print("18. DATA DICTIONARY")
print("=" * 70)

print(data_dictionary.to_string(index=False))

data_dictionary.to_csv(
    "data_dictionary.csv",
    index=False
)


# ------------------------------------------------------------
# 19. SAVE CLEANED DATASET
# ------------------------------------------------------------

# Remove spreadsheet artifact columns.
cleaned_df = df.drop(columns=unnamed_cols)

cleaned_df.to_csv(
    "Restaurant_cleaned.csv",
    index=False
)

cleaned_df.to_excel(
    "Restaurant_cleaned.xlsx",
    index=False
)


# ------------------------------------------------------------
# 20. BASIC EDA FINDINGS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("20. BASIC EDA FINDINGS")
print("=" * 70)

print("\nAverage bill amount:",
      round(df["Amount"].mean(), 2))

print("Median bill amount:",
      round(df["Amount"].median(), 2))

print("\nAverage tip:",
      round(df["Tip"].mean(), 2))

print("Median tip:",
      round(df["Tip"].median(), 2))

print("\nAverage party size:",
      round(df["Partysize"].mean(), 2))

print("Median party size:",
      round(df["Partysize"].median(), 2))

print("\nAverage tip percentage:",
      round(df["Tip_Percentage"].mean(), 2))

print("\nMost common day:")
print(df["Day"].mode()[0])

print("\nMost common meal time:")
print(df["Time"].mode()[0])

print("\nGender distribution:")
print(df["Gender"].value_counts())

print("\nSmoker distribution:")
print(df["Smoker"].value_counts())

print("\nDay distribution:")
print(df["Day"].value_counts())

print("\nTime distribution:")
print(df["Time"].value_counts())


# ------------------------------------------------------------
# 21. CLEANING SUMMARY
# ------------------------------------------------------------

cleaning_summary = pd.DataFrame({
    "Cleaning_Action": [
        "Investigated Unnamed columns",
        "Converted Amount to numeric",
        "Converted Tip to numeric",
        "Converted Partysize to numeric",
        "Standardized Gender categories",
        "Standardized Smoker categories",
        "Standardized Day categories",
        "Standardized Time categories",
        "Invalid Amount values converted to missing",
        "Invalid Tip values converted to missing",
        "Invalid Partysize values converted to missing",
        "Created Tip_Percentage",
        "Removed spreadsheet artifact columns from final dataset"
    ]
})

cleaning_summary.to_csv(
    "cleaning_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 22. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nOriginal dataset shape:", (365, 11))
print("Final cleaned dataset shape:", cleaned_df.shape)

print("\nFiles generated:")
print("1. Restaurant_cleaned.csv")
print("2. Restaurant_cleaned.xlsx")
print("3. data_dictionary.csv")
print("4. missing_value_analysis.csv")
print("5. final_missing_value_analysis.csv")
print("6. descriptive_statistics.csv")
print("7. cleaning_summary.csv")

print("\nDone.")