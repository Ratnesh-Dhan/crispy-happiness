import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import os, joblib, numpy as np

df = pd.read_excel("/mnt/d/DATASETS/Noamundi_Data.xlsx")
model_path = "./models/Fe_preds"
os.makedirs(model_path, exist_ok=True)
results_path = './results/without_Fe'
os.makedirs(results_path, exist_ok=True)

X = df[["T1", "T2", "T3", "T4", "AvgTemp"]]
target = ["Fe%"]

X_train, X_test, y_train, y_test = train_test_split(
    X, df[target], test_size=0.2, random_state=42
)

models = {}
preds = {}

## others
# from lightgbm import LGBMRegressor
# model = LGBMRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     num_leaves=31
# )

# from sklearn.ensemble import ExtraTreesRegressor
# model = ExtraTreesRegressor(
#     n_estimators=600,
#     max_depth=20
# )


for col in target:
#     model = XGBRegressor(
#         n_estimators=400,
#         learning_rate=0.05,
#         max_depth=5, 
#         subsample=0.9,
#         colsample_bytree=0.9,
#         objective="reg:squarederror"
#     )
    model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=20
    )
    model.fit(X_train, y_train[col])
    models[col] = model
    preds[col] = model.predict(X_test)

errors = {}

for col in target:
    actual = y_test[col].values
    pred   = preds[col]

    err = np.abs(pred - actual) / actual
    errors[col] = err * 100

# save to txt
with open(os.path.join(results_path, "model_errors.txt"), "w") as f:
    for col in target:
        f.write(f"===== {col} Errors in % =====\n")
        f.write("======= abs(pred - actual) / actual =======\n")
        for e in errors[col]:
            f.write(f"{e}\n")
        f.write("\n")

print(f"Saved {results_path}/model_errors.txt")

for col, model in models.items():
    path = os.path.join(model_path, f"xgb_{col.replace('%', '')}.pkl")
    joblib.dump(model, path)
    print("Saved: ",path)

# ---- METRICS ----
y_pred = np.column_stack([preds[col] for col in target])

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred, multioutput="uniform_average"))

for col in target:
    mae = mean_absolute_error(y_test[col], preds[col])
    r2  = r2_score(y_test[col], preds[col])

    print(f"{col} MAE:", mae)
    print(f"{col} R2 :", r2)
# ---- GRAPHS ----
for i, col in enumerate(target):
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
    