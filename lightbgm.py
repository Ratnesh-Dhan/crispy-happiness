import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import joblib, os

dataset_path = "/mnt/d/DATASETS"
base_path = "/mnt/d/Codes/Regression"
df = pd.read_excel(os.path.join(dataset_path, "Noamundi_Data.xlsx"))

X = df[["T1","T2","T3","T4","AvgTemp"]]
y = df[["Fe%","SiO2%","Al2O3%"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_base = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9
)

model = MultiOutputRegressor(model_base)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "./models/lgbm_model.pkl")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred, multioutput="uniform_average"))
