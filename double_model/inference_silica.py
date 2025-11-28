import pandas as pd
import joblib
import os

# -------- CONFIG --------
data_path = "./results/without_Fe/Alumina_inference_results.xlsx"
model_file = "./models/Final_Silica/xgb_SiO2.pkl"     # whatever file you saved
output_file = "./results/Final_Silica/Silica_inference_results.xlsx"

# -------- LOAD MODEL --------
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model not found: {model_file}")

model = joblib.load(model_file)

# -------- LOAD DATA --------
df = pd.read_excel(data_path)

# Ensure same feature ordering
X = df[["T1", "T2", "T3", "T4", "AvgTemp", "Predicted_Fe%", "Predicted_Alumina%"]]

# -------- PREDICT --------
predictions = model.predict(X)

df["Predicted_SiO2%"] = predictions

# -------- SAVE --------
df.to_excel(output_file, index=False)

print(f"Inference completed. Saved to: {output_file}")
