import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Restaurant_cleaned.csv")

os.makedirs("figures", exist_ok=True)
os.makedirs("eda_results", exist_ok=True)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# Categorical analysis

categorical_cols = ["Gender", "Smoker", "Day", "Time"]

print("\nUnique categorical values:")

for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].unique())

print("\nCategory frequencies:")

for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))


# Descriptive statistics

numeric_cols = [
    "Amount",
    "Tip",
    "Partysize",
    "Tip_Percentage"
]

print("\nDescriptive statistics:")

descriptive_stats = df[numeric_cols].describe().T

descriptive_stats = descriptive_stats[
    ["count", "mean", "50%", "std", "min", "25%", "75%", "max"]
]

descriptive_stats = descriptive_stats.rename(
    columns={
        "50%": "Median",
        "25%": "Q1",
        "75%": "Q3"
    }
)

print(descriptive_stats)

descriptive_stats.to_csv(
    "eda_results/eda_descriptive_statistics.csv"
)


# IQR analysis

print("\nIQR analysis:")

iqr_results = []

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    potential_outliers = (
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ).sum()

    iqr_results.append({
        "Variable": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Potential_Outliers": potential_outliers
    })

    print(f"\n{col}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Potential outliers: {potential_outliers}")

iqr_df = pd.DataFrame(iqr_results)

iqr_df.to_csv(
    "eda_results/iqr_analysis.csv",
    index=False
)


# Grouped statistics by day

print("\nAmount statistics by day:")

day_stats = (
    df.groupby("Day", dropna=False)["Amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(day_stats)

day_stats.to_csv(
    "eda_results/amount_statistics_by_day.csv"
)


# Grouped statistics by meal time

print("\nAmount statistics by meal time:")

time_stats = (
    df.groupby("Time", dropna=False)["Amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(time_stats)

time_stats.to_csv(
    "eda_results/amount_statistics_by_time.csv"
)


# Grouped statistics by party size

print("\nAmount statistics by party size:")

party_stats = (
    df.groupby("Partysize", dropna=False)["Amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(party_stats)

party_stats.to_csv(
    "eda_results/amount_statistics_by_party_size.csv"
)


# Grouped statistics by gender

print("\nAmount statistics by gender:")

gender_stats = (
    df.groupby("Gender", dropna=False)["Amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(gender_stats)

gender_stats.to_csv(
    "eda_results/amount_statistics_by_gender.csv"
)


# Grouped statistics by smoker status

print("\nAmount statistics by smoker status:")

smoker_stats = (
    df.groupby("Smoker", dropna=False)["Amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(smoker_stats)

smoker_stats.to_csv(
    "eda_results/amount_statistics_by_smoker.csv"
)


# Lunch and Dinner subset

meal_df = df[
    df["Time"].isin(["Lunch", "Dinner"])
].copy()

print("\nMeal-time comparison:")

print("Total observations:", len(df))
print("Lunch/Dinner observations:", len(meal_df))
print(
    "Other Time categories excluded from meal comparison:",
    len(df) - len(meal_df)
)


# Chart 1: Bill amount distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df["Amount"].dropna(),
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Restaurant Bill Amounts")
plt.xlabel("Bill Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "figures/01_bill_amount_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Chart 2: Tip distribution
# Logarithmic x-axis keeps extreme observations visible

tip_values = df["Tip"].dropna()

plt.figure(figsize=(8, 5))

plt.hist(
    tip_values,
    bins=30,
    edgecolor="black"
)

plt.xscale("log")

plt.title("Distribution of Tips")
plt.xlabel("Tip Amount (log scale)")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "figures/02_tip_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Chart 3: Bill amount by day

day_order = ["Thur", "Fri", "Sat", "Sun"]

day_plot_data = []
valid_day_labels = []

for day in day_order:

    values = df.loc[
        df["Day"] == day,
        "Amount"
    ].dropna()

    if len(values) > 0:
        day_plot_data.append(values)
        valid_day_labels.append(day)

plt.figure(figsize=(8, 5))

plt.boxplot(
    day_plot_data,
    tick_labels=valid_day_labels
)

plt.title("Bill Amount by Day")
plt.xlabel("Day")
plt.ylabel("Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/03_bill_amount_by_day.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Chart 4: Bill amount by meal time

lunch_amount = meal_df.loc[
    meal_df["Time"] == "Lunch",
    "Amount"
].dropna()

dinner_amount = meal_df.loc[
    meal_df["Time"] == "Dinner",
    "Amount"
].dropna()

plt.figure(figsize=(8, 5))

plt.boxplot(
    [lunch_amount, dinner_amount],
    tick_labels=["Lunch", "Dinner"]
)

plt.title("Bill Amount by Meal Time")
plt.xlabel("Meal Time")
plt.ylabel("Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/04_bill_amount_by_meal_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Chart 5: Party size vs bill amount

party_amount_df = df[
    ["Partysize", "Amount"]
].dropna()

plt.figure(figsize=(8, 5))

plt.scatter(
    party_amount_df["Partysize"],
    party_amount_df["Amount"],
    alpha=0.6
)

plt.title("Party Size vs Bill Amount")
plt.xlabel("Party Size")
plt.ylabel("Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/05_party_size_vs_bill_amount.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Chart 6: Bill amount vs tip
# Logarithmic y-axis keeps extreme tips visible

amount_tip_df = df[
    ["Amount", "Tip"]
].dropna()

plt.figure(figsize=(8, 5))

plt.scatter(
    amount_tip_df["Amount"],
    amount_tip_df["Tip"],
    alpha=0.6
)

plt.yscale("log")

plt.title("Bill Amount vs Tip")
plt.xlabel("Bill Amount")
plt.ylabel("Tip Amount (log scale)")

plt.tight_layout()

plt.savefig(
    "figures/06_bill_amount_vs_tip.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Chart 7: Tip percentage by meal time
# Logarithmic y-axis keeps extreme percentages visible

lunch_tip_percentage = meal_df.loc[
    meal_df["Time"] == "Lunch",
    "Tip_Percentage"
].dropna()

dinner_tip_percentage = meal_df.loc[
    meal_df["Time"] == "Dinner",
    "Tip_Percentage"
].dropna()

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        lunch_tip_percentage,
        dinner_tip_percentage
    ],
    tick_labels=["Lunch", "Dinner"]
)

plt.yscale("log")

plt.title("Tip Percentage by Meal Time")
plt.xlabel("Meal Time")
plt.ylabel("Tip Percentage (log scale)")

plt.tight_layout()

plt.savefig(
    "figures/07_tip_percentage_by_meal_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Correlation analysis

print("\nCorrelation matrix:")

correlation_matrix = df[
    [
        "Amount",
        "Tip",
        "Partysize",
        "Tip_Percentage"
    ]
].corr()

print(correlation_matrix)

correlation_matrix.to_csv(
    "eda_results/correlation_matrix.csv"
)


# Median values

print("\nMedian values:")

for col in numeric_cols:

    median_value = df[col].median()

    print(
        f"{col}: {median_value:.2f}"
    )


print("\nEDA complete.")

print("\nGenerated figures:")

print("01_bill_amount_distribution.png")
print("02_tip_distribution.png")
print("03_bill_amount_by_day.png")
print("04_bill_amount_by_meal_time.png")
print("05_party_size_vs_bill_amount.png")
print("06_bill_amount_vs_tip.png")
print("07_tip_percentage_by_meal_time.png")

print("\nGenerated result files:")

print("eda_descriptive_statistics.csv")
print("iqr_analysis.csv")
print("amount_statistics_by_day.csv")
print("amount_statistics_by_time.csv")
print("amount_statistics_by_party_size.csv")
print("amount_statistics_by_gender.csv")
print("amount_statistics_by_smoker.csv")
print("correlation_matrix.csv")