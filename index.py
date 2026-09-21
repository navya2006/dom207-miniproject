import pandas as pd
import numpy as np

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_excel("Restaurant.xlsx")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ============================================
# 2. BASIC DATA CHECK
# ============================================

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ============================================
# 3. REMOVE EMPTY SPREADSHEET COLUMNS
# ============================================

unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]

print("\nSpreadsheet artifact columns:")
print(unnamed_cols)

df = df.drop(columns=unnamed_cols)

# ============================================
# 4. CONVERT NUMERICAL VARIABLES
# ============================================

df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["Tip"] = pd.to_numeric(df["Tip"], errors="coerce")
df["Partysize"] = pd.to_numeric(df["Partysize"], errors="coerce")

# ============================================
# 5. CLEAN CATEGORICAL VARIABLES
# ============================================

# Gender
df["Gender"] = df["Gender"].replace({
    "M": "Male",
    "Mal": "Male",
    "Mle": "Male",
    "F": "Female",
    "Fe": "Female",
    "Fem": "Female",
    "Femle": "Female",
    "Fmle": "Female"
})

# Smoker
df["Smoker"] = df["Smoker"].replace({
    "Y": "Yes",
    "N": "No",
    "s": "Yes"
})

# Day
df["Day"] = df["Day"].replace({
    "Saturday": "Sat",
    "Friday": "Fri",
    "Thurs": "Thur",
    "Th": "Thur",
    "Trhurs": "Thur",
    "T": "Thur",
    "Ft": "Fri",
    "S": "Sun",
    "SS": "Sun",
    "SSS": "Sun",
    "San": "Sun",
    "Sn": "Sun",
    "sun": "Sun"
})

# Time
df["Time"] = df["Time"].replace({
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
})

# ============================================
# 6. HANDLE INVALID NUMERICAL VALUES
# ============================================

df.loc[df["Amount"] <= 0, "Amount"] = np.nan
df.loc[df["Tip"] < 0, "Tip"] = np.nan

df.loc[
    (df["Partysize"] <= 0) | (df["Partysize"] > 20),
    "Partysize"
] = np.nan

# ============================================
# 7. CREATE TIP PERCENTAGE
# ============================================

df["Tip_Percentage"] = (df["Tip"] / df["Amount"]) * 100

# ============================================
# 8. DESCRIPTIVE STATISTICS
# ============================================

stats = df[
    ["Amount", "Tip", "Partysize"]
].describe().T

stats = stats[
    ["mean", "50%", "std", "min", "25%", "75%", "max"]
]

stats.columns = [
    "Mean",
    "Median",
    "Std_Dev",
    "Minimum",
    "Q1",
    "Q3",
    "Maximum"
]

print("\nDescriptive Statistics:")
print(stats.round(2))

# ============================================
# 9. BASIC CATEGORICAL ANALYSIS
# ============================================

print("\nGender:")
print(df["Gender"].value_counts())

print("\nSmoker:")
print(df["Smoker"].value_counts())

print("\nDay:")
print(df["Day"].value_counts())

print("\nTime:")
print(df["Time"].value_counts())

# ============================================
# 10. FINAL DATA CHECK
# ============================================

print("\nFinal missing values:")
print(df.isnull().sum())

print("\nFinal shape:", df.shape)

# ============================================
# 11. SAVE OUTPUT
# ============================================

df.to_excel("Restaurant_cleaned.xlsx", index=False)

stats.to_csv("descriptive_statistics.csv")

print("\nPreprocessing complete!")
print("Created:")
print("- Restaurant_cleaned.xlsx")
print("- descriptive_statistics.csv")