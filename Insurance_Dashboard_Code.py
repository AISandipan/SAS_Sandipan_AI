
# ======================================
# Insurance Risk & Claims Dashboard Code
# Python Version
# ======================================

import pandas as pd

# Load data
df = pd.read_csv("insurance_data.csv")

# -------------------
# KPI Calculations
# -------------------

total_policies = df["PolicyID"].nunique()
total_claim_amount = df["ClaimAmount"].sum()
avg_claim_freq = df["ClaimFrequency"].mean()
avg_claim_amount = df["ClaimAmount"].mean()

male_count = df[df["Gender"]=="Male"]["CustomerID"].nunique()
female_count = df[df["Gender"]=="Female"]["CustomerID"].nunique()

print("Total Policies:", total_policies)
print("Total Claim Amount:", total_claim_amount)
print("Avg Claim Frequency:", round(avg_claim_freq,2))
print("Avg Claim Amount:", round(avg_claim_amount,2))

# -------------------
# Visual Datasets
# -------------------

claim_by_car_use = df.groupby("CarUse")["ClaimAmount"].sum()

claim_by_make = (
    df.groupby("CarMake")["ClaimAmount"]
    .sum()
    .sort_values(ascending=False)
)

claim_by_zone = df.groupby("CoverageZone")["ClaimAmount"].sum()

claim_by_age = df.groupby("AgeGroup")["ClaimAmount"].sum()

claim_by_year = (
    df.groupby("CarYear")["ClaimAmount"]
    .sum()
    .reset_index()
)

claim_by_kids = df.groupby("KidsDriving")["ClaimAmount"].sum()

claim_by_education = df.groupby("Education")["ClaimAmount"].sum()

matrix = pd.pivot_table(
    df,
    values="ClaimAmount",
    index="Education",
    columns="MaritalStatus",
    aggfunc="sum",
    fill_value=0
)

print(matrix)
