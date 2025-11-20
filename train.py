from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load CSV
dataset_path = "/mnt/d/DATASETS"
base_path = "/mnt/d/Codes/Regression"
df = pd.read_excel(os.path.join(dataset_path, "Noamundi_Data.xlsx"))

# Inputs and outputs
X = df[["T1", "T2", "T3", "T4", "AvgTemp"]]
y = df[["Fe%", "SiO2%", "Al2O3%"]]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "multi_output_rf_model.pkl")

# Plot the results
y_test_np = y_test.values
y_pred_np = y_pred

targets = ["Fe%", "SiO2%", "Al2O3%"]

# 1. Predicted vs Actual
for i, t in enumerate(targets):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test_np[:, i], y_pred_np[:, i])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{t} – Actual vs Predicted")
    plt.grid(True)
    plt.savefig(os.path.join(base_path,'result', f"{t}_actual_vs_predicted.png"))
    # plt.show()

# 2. Residuals plot
for i, t in enumerate(targets):
    residuals = y_test_np[:, i] - y_pred_np[:, i]
    plt.figure(figsize=(6,5))
    plt.scatter(y_pred_np[:, i], residuals)
    plt.axhline(0, color='red', linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"{t} – Residuals Plot")
    plt.grid(True)
    plt.savefig(os.path.join(base_path, 'result', f"{t}Residuals_plot.png"))
    # plt.show()

# 3. Error distribution
for i, t in enumerate(targets):
    residuals = y_test_np[:, i] - y_pred_np[:, i]
    plt.figure(figsize=(6,5))
    plt.hist(residuals, bins=20)
    plt.title(f"{t} – Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(base_path, 'result', f"{t}Error_distribution.png"))
    # plt.show()

# 4. Feature importance
importances = model.feature_importances_
plt.figure(figsize=(7,5))
plt.bar(["T1", "T2", "T3", "T4", "AvgTemp"], importances)
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.grid(True)
plt.savefig(os.path.join(base_path, 'result', "Feature_importance.png"))
# plt.show()
