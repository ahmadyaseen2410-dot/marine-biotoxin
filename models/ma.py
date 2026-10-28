import pandas as pd

df = pd.read_csv("Domoic_Acid.csv")
print(sorted(df["species"].dropna().astype(str).unique().tolist()))