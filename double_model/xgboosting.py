from unittest import result
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os, joblib, sys, numpy as np

# df = pd.read_excel("/mnt/d/DATASETS/Noamundi_Data.xlsx")
df = pd.read_excel("/mnt/d/Codes/Regression/double_model/results/without_Fe/Fe_inference_results.xlsx")
model_path = "./models"
results_path = "./results"

X = df[["T1","T2","T3","T4","AvgTemp","Predicted_Fe%"]]
targets = ["SiO2%","Al2O3%"]

X_train, X_test, y_train, y_test = train_test_split(
    X, df[targets], test_size=0.2, random_state=42
)

models = {}
preds = {}

for col in targets:
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train[col])
    models[col] = model
    preds[col] = model.predict(X_test)

errors = {}

for col in targets:
    actual = y_test[col].values
    pred   = preds[col]

    err = np.abs(pred - actual) / actual
    errors[col] = err * 100

# save to txt
with open(os.path.join(results_path, "model_errors.txt"), "w") as f:
    for col in targets:
        f.write(f"===== {col} Errors =====\n")
        for e in errors[col]:
            f.write(f"{e}\n")
        f.write("\n")

print(f"Saved {results_path}/model_errors.txt")

out_lines = []
out_lines.append("SiO2_actual, SiO2_pred, SiO2_error, Al2O3_actual, Al2O3_pred, Al2O3_error")

si_actual = y_test["SiO2%"].values
si_pred   = preds["SiO2%"]
si_err    = np.abs(si_pred - si_actual) / si_actual

al_actual = y_test["Al2O3%"].values
al_pred   = preds["Al2O3%"]
al_err    = np.abs(al_pred - al_actual) / al_actual

for i in range(len(si_actual)):
    line = f"{si_actual[i]}, {si_pred[i]}, {si_err[i]}, {al_actual[i]}, {al_pred[i]}, {al_err[i]}"
    out_lines.append(line)

# create dataframe
df_err = pd.DataFrame({
    "SiO2_actual": si_actual,
    "SiO2_pred": si_pred,
    "SiO2_error": si_err,
    "Al2O3_actual": al_actual,
    "Al2O3_pred": al_pred,
    "Al2O3_error": al_err
})

# save to excel
df_err.to_excel(os.path.join(results_path, "silica_alumina_errors.xlsx"), index=False)

print(f"Saved {results_path}/silica_alumina_errors.xlsx")

with open(os.path.join(results_path, "silica_alumina_errors.txt"), "w") as f:
    f.write("\n".join(out_lines))

print(f"{results_path}/Saved silica_alumina_errors.txt")

# sys.exit(0)

for col, model in models.items():
    path = os.path.join(model_path, f"xgb_{col.replace('%', '')}.pkl")
    joblib.dump(model, path)
    print("Saved: ",path)

# ---- METRICS ----
y_pred = np.column_stack([preds[col] for col in targets])

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred, multioutput="uniform_average"))

for col in targets:
    mae = mean_absolute_error(y_test[col], preds[col])
    r2  = r2_score(y_test[col], preds[col])

    print(f"{col} MAE:", mae)
    print(f"{col} R2 :", r2)
# ---- GRAPHS ----
for i, col in enumerate(targets):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test[col], preds[col], s=10)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{col} — Actual vs Predicted")
    plt.plot([y_test[col].min(), y_test[col].max()],
             [y_test[col].min(), y_test[col].max()], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{col}_ActualVSpredicted.png"))
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(preds[col], label="Pred", linewidth=1)
    plt.plot(y_test[col].values, label="Actual", linewidth=1)
    plt.title(f"{col} — Prediction Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{col}_PredictionCurve.png"))
    # plt.show()
