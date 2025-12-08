import pandas as pd
import joblib
import os

# -------- CONFIG --------
base_path = "/mnt/d/Codes/Regression/re-try"
data_path = "/mnt/d/DATASETS/noamundi_split_dataset/final_test_dataset.xlsx"
# silica_model = os.path.join(base_path, "models/Final_Silica/xgb_SiO2.pkl")     # whatever file you saved
# alumina_model = os.path.join(base_path, "models/xgb_Al2O3.pkl")
# silica_model = "/mnt/d/Codes/Regression/re-try/models/Final_Silica/xgb_SiO2.pkl"
# alumina_model = "/mnt/d/Codes/Regression/re-try/models/Final_Alumina/xgb_Al2O3.pkl"
# iron_model = os.path.join(base_path, "models/Fe_preds/xgb_Fe.pkl")
# # iron_model = "/mnt/d/Codes/Regression/re-try/models/Fe_preds/xgb_Fe.pkl"

iron_model = "/mnt/d/Codes/Regression/re-try/train/models/Fe_preds/xgb_Fe.pkl"
alumina_model="/mnt/d/Codes/Regression/re-try/train/models/Final_Alumina/xgb_Al2O3.pkl"
silica_model="/mnt/d/Codes/Regression/re-try/train/models/Final_Silica/xgb_SiO2.pkl"
output_file = "/mnt/d/Codes/Regression/re-try/results/FINAL/inference_results.xlsx"
print(iron_model)
os.makedirs("/mnt/d/Codes/Regression/re-try/results/FINAL", exist_ok=True)

# -------- LOAD DATA --------
df = pd.read_excel(data_path)

# -------- IRON MODEL --------
model_fe = joblib.load(iron_model)
X = df[["T1", "T2", "T3", "T4", "AvgTemp"]]
fe_prediction = model_fe.predict(X)
df["Fe%"] = fe_prediction
print("done")
# ------- ALUMINA MODEL ------
model_alumina = joblib.load(alumina_model)
X = df[["T1", "T2", "T3", "T4", "AvgTemp", "Fe%"]]
alumina_prediction = model_alumina.predict(X)
df["Al2O3%"] = alumina_prediction
print("done")

# -------- SILICA MODEL --------
model_silica = joblib.load(silica_model)
X = df[["T1", "T2", "T3", "T4", "AvgTemp", "Fe%", "Al2O3%"]]
predictions = model_silica.predict(X)
df["SiO2%"] = predictions
print("done")

# -------- SAVE --------
df.to_excel(output_file, index=False)

print(f"Inference completed. Saved to: {output_file}")
