import pandas as pd
import joblib
import os

# -------- CONFIG --------
base_path = "/mnt/d/Codes/Regression/doulbe_model"
data_path = "/mnt/d/DATASETS/Test_Data_Noamundi.xlsx"
# silica_model = os.path.join(base_path, "models/Final_Silica/xgb_SiO2.pkl")     # whatever file you saved
# alumina_model = os.path.join(base_path, "models/xgb_Al2O3.pkl")
silica_model = "/mnt/d/Codes/Regression/double_model/models/Final_Silica/xgb_SiO2.pkl"
alumina_model = "/mnt/d/Codes/Regression/double_model/models/xgb_Al2O3.pkl"
# iron_model = os.path.join(base_path, "models/Fe_preds/ExtraTreeRegressor_Fe_.pkl")
iron_model = "/mnt/d/Codes/Regression/double_model/models/Fe_preds/ExtraTreeRegressor_Fe_.pkl"
output_file = "/mnt/d/Codes/Regression/double_model/results/FINAL/inference_results.xlsx"
print(iron_model)
os.makedirs("/mnt/d/Codes/Regression/double_model/results/FINAL", exist_ok=True)

# -------- LOAD DATA --------
df = pd.read_excel(data_path)

# -------- IRON MODEL --------
model_fe = joblib.load(iron_model)
X = df[["T1", "T2", "T3", "T4", "AvgTemp"]]
fe_prediction = model_fe.predict(X)
df["Predicted_Fe%"] = fe_prediction
print("done")
# ------- ALUMINA MODEL ------
model_alumina = joblib.load(alumina_model)
X = df[["T1", "T2", "T3", "T4", "AvgTemp", "Predicted_Fe%"]]
alumina_prediction = model_alumina.predict(X)
df["Predicted_Alumina%"] = alumina_prediction
print("done")

# -------- SILICA MODEL --------
model_silica = joblib.load(silica_model)
X = df[["T1", "T2", "T3", "T4", "AvgTemp", "Predicted_Fe%", "Predicted_Alumina%"]]
predictions = model_silica.predict(X)
df["Predicted_SiO2%"] = predictions
print("done")

# -------- SAVE --------
df.to_excel(output_file, index=False)

print(f"Inference completed. Saved to: {output_file}")
