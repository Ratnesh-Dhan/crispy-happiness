import pandas as pd
import joblib
import os

# -------- CONFIG --------
data_path = "/mnt/d/DATASETS/Noamundi_Data.xlsx"
model_file = "./models/Fe_preds/ExtraTreeRegressor_Fe_.pkl"     # whatever file you saved
output_file = "./results/without_Fe/Fe_inference_results.xlsx"

# -------- LOAD MODEL --------
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model not found: {model_file}")

model = joblib.load(model_file)

# -------- LOAD DATA --------
df = pd.read_excel(data_path)

# Ensure same feature ordering
X = df[["T1", "T2", "T3", "T4", "AvgTemp"]]

# -------- PREDICT --------
predictions = model.predict(X)

df["Predicted_Fe%"] = predictions

# -------- SAVE --------
df.to_excel(output_file, index=False)

print(f"Inference completed. Saved to: {output_file}")
