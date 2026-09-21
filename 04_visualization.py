import os
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset

df = pd.read_csv("Restaurant_cleaned.csv")

os.makedirs("figures", exist_ok=True)


# --------------------------------------------------
# 1. BILL AMOUNT DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["Amount"].dropna(),
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Restaurant Bill Amounts", fontsize=14)
plt.xlabel("Bill Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "figures/final_01_bill_amount_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 2. BILL AMOUNT BY DAY
# --------------------------------------------------

day_order = ["Thur", "Fri", "Sat", "Sun"]

day_data = []
day_labels = []

for day in day_order:

    values = df.loc[
        df["Day"] == day,
        "Amount"
    ].dropna()

    if len(values) > 0:
        day_data.append(values)
        day_labels.append(day)


plt.figure(figsize=(9, 6))

plt.boxplot(
    day_data,
    tick_labels=day_labels
)

plt.title("Bill Amount by Day", fontsize=14)
plt.xlabel("Day")
plt.ylabel("Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/final_02_bill_amount_by_day.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 3. BILL AMOUNT BY MEAL TIME
# --------------------------------------------------

meal_df = df[
    df["Time"].isin(["Lunch", "Dinner"])
].copy()

lunch_amount = meal_df.loc[
    meal_df["Time"] == "Lunch",
    "Amount"
].dropna()

dinner_amount = meal_df.loc[
    meal_df["Time"] == "Dinner",
    "Amount"
].dropna()


plt.figure(figsize=(9, 6))

plt.boxplot(
    [
        lunch_amount,
        dinner_amount
    ],
    tick_labels=[
        "Lunch",
        "Dinner"
    ]
)

plt.title("Bill Amount by Meal Time", fontsize=14)
plt.xlabel("Meal Time")
plt.ylabel("Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/final_03_bill_amount_by_meal_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 4. PARTY SIZE VS BILL AMOUNT
# --------------------------------------------------

party_df = df[
    ["Partysize", "Amount"]
].dropna()


plt.figure(figsize=(9, 6))

plt.scatter(
    party_df["Partysize"],
    party_df["Amount"],
    alpha=0.6
)

plt.title("Party Size vs Bill Amount", fontsize=14)
plt.xlabel("Party Size")
plt.ylabel("Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/final_04_party_size_vs_bill_amount.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 5. BILL AMOUNT VS TIP
# --------------------------------------------------

amount_tip_df = df[
    ["Amount", "Tip"]
].dropna()


plt.figure(figsize=(9, 6))

plt.scatter(
    amount_tip_df["Amount"],
    amount_tip_df["Tip"],
    alpha=0.6
)

plt.yscale("log")

plt.title(
    "Bill Amount vs Tip",
    fontsize=14
)

plt.xlabel("Bill Amount")
plt.ylabel("Tip Amount (log scale)")

plt.tight_layout()

plt.savefig(
    "figures/final_05_bill_amount_vs_tip.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 6. TIP PERCENTAGE BY MEAL TIME
# --------------------------------------------------

lunch_tip = meal_df.loc[
    meal_df["Time"] == "Lunch",
    "Tip_Percentage"
].dropna()

dinner_tip = meal_df.loc[
    meal_df["Time"] == "Dinner",
    "Tip_Percentage"
].dropna()


plt.figure(figsize=(9, 6))

plt.boxplot(
    [
        lunch_tip,
        dinner_tip
    ],
    tick_labels=[
        "Lunch",
        "Dinner"
    ]
)

plt.yscale("log")

plt.title(
    "Tip Percentage by Meal Time",
    fontsize=14
)

plt.xlabel("Meal Time")
plt.ylabel("Tip Percentage (log scale)")

plt.tight_layout()

plt.savefig(
    "figures/final_06_tip_percentage_by_meal_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 7. AVERAGE BILL AMOUNT BY PARTY SIZE
# --------------------------------------------------

party_average = (
    df.groupby("Partysize")["Amount"]
    .mean()
    .dropna()
)


plt.figure(figsize=(9, 6))

plt.bar(
    party_average.index.astype(str),
    party_average.values,
    edgecolor="black"
)

plt.title(
    "Average Bill Amount by Party Size",
    fontsize=14
)

plt.xlabel("Party Size")
plt.ylabel("Average Bill Amount")

plt.tight_layout()

plt.savefig(
    "figures/final_07_average_bill_by_party_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# COMPLETION MESSAGE
# --------------------------------------------------

print("\nFinal visualization generation complete.")

print("\nGenerated final visualization files:")

print("final_01_bill_amount_distribution.png")
print("final_02_bill_amount_by_day.png")
print("final_03_bill_amount_by_meal_time.png")
print("final_04_party_size_vs_bill_amount.png")
print("final_05_bill_amount_vs_tip.png")
print("final_06_tip_percentage_by_meal_time.png")
print("final_07_average_bill_by_party_size.png")