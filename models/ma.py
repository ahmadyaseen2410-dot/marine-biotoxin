import pandas as pd

# Load the dataset
df = pd.read_csv("ToxiGuard_Domoic_Acid_Corrected.csv")

# Rename the column
df.rename(columns={"synthetic_high_risk": "observed"}, inplace=True)

# Save the updated dataset
df.to_csv("Domoic_Acid.csv", index=False)

print("Column renamed successfully!")
print(df.columns.tolist())