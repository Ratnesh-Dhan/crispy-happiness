import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_excel("/mnt/data/Noamundi_Data.xlsx")

# FEATURES (5 columns)
X = df[["T1","T2","T3","T4","AvgTemp"]]

# TARGET
y = df["Fe%"]

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- MLP MODEL ----
model = Sequential([
    Dense(64, activation="relu", input_shape=(5,)),
    Dense(64, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=150, batch_size=16,
                    validation_split=0.2, verbose=0)

# PREDICT
pred = model.predict(X_test).flatten()

# ---- METRICS ----
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("R2 :", r2)

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, pred, s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Fe%")
plt.ylabel("Predicted Fe%")
plt.title("Actual vs Predicted Fe% (MLP)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(y_test.values, label="Actual", linewidth=1)
plt.plot(pred, label="Pred", linewidth=1)
plt.title("Fe% Prediction Curve")
plt.legend()
plt.tight_layout()
plt.show()

# If per-sample errors needed
# err = np.abs(pred - y_test.values) / y_test.values
