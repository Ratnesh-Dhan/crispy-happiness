import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_excel("your_file.xlsx")

X = df[["T1","T2","T3","T4","AvgTemp"]]
y = df[["Fe%","SiO2%","Al2O3%"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {}
preds = {}

for col in y.columns:
    print(f"Training XGB for {col}...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train[col])
    models[col] = model
    preds[col] = model.predict(X_test)

# Combine predictions
import numpy as np
y_pred = np.column_stack([preds[col] for col in y.columns])

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred, multioutput="uniform_average"))
