import pandas as pd
df = pd.read_excel("/mnt/d/Codes/Regression/double_model/results/without_Fe/Alumina_inference_results.xlsx")

count = df["SiO2%"].isna().sum()
print(count)
nan_positions = df[df["SiO2%"].isna()].index.tolist()
print(nan_positions)
print(df.loc[nan_positions[0]])

# df["SiO2%"] = df["SiO2%"].astype(str).str.strip()
