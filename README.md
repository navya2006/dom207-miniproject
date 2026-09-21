# Foodie India - Restaurant Data Analysis

## DOM207 Mini Project

This project analyzes restaurant transaction data to understand customer spending, tipping behavior, party size, meal-time patterns, and day-wise transaction patterns.

The project follows a structured data-analysis pipeline consisting of:

1. Data ingestion
2. Data preprocessing and cleaning
3. Exploratory Data Analysis (EDA)
4. Data visualization
5. Statistical analysis
6. Business interpretation and recommendations

---

# Team Members

| Person | Name | Role |
|---|---|---|
| Person 1 | Navya | Data Engineer / Preprocessing Lead |
| Person 2 | Amitoj | Data Analyst / EDA & Visualization Lead |
| Person 3 | Dev | Statistical & Business Analysis Lead |

All team members are responsible for reviewing the complete project and preparing for the final presentation and viva.

---

# Project Objectives

The main objectives of this project are to:

- Understand the structure and quality of the restaurant transaction dataset.
- Clean and prepare the data for analysis.
- Examine restaurant bill amounts and tipping behavior.
- Analyze spending patterns across different days.
- Compare Lunch and Dinner transactions.
- Study the relationship between party size and bill amount.
- Examine the relationship between bill amount and tip.
- Analyze tip percentage across meal times.
- Perform statistical tests on relevant relationships.
- Translate statistically supported findings into business insights.
- Develop evidence-based recommendations without making unsupported assumptions.

---

# Dataset

The dataset contains **365 restaurant transactions** and the following variables:

| Variable | Description |
|---|---|
| Amount | Restaurant bill amount |
| Tip | Tip amount |
| Gender | Gender category |
| Smoker | Whether the customer is a smoker |
| Day | Day of the transaction |
| Time | Meal time |
| Partysize | Number of people in the party |
| Tip_Percentage | Derived percentage of tip relative to bill amount |

The derived variable is calculated as:

```text
Tip_Percentage = (Tip / Amount) × 100