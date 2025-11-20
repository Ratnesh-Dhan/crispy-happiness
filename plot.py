import matplotlib.pyplot as plt

# Assuming y_test, y_pred already exist
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
    plt.show()

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
    plt.show()

# 3. Error distribution
for i, t in enumerate(targets):
    residuals = y_test_np[:, i] - y_pred_np[:, i]
    plt.figure(figsize=(6,5))
    plt.hist(residuals, bins=20)
    plt.title(f"{t} – Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 4. Feature importance
importances = model.feature_importances_
plt.figure(figsize=(7,5))
plt.bar(["T1", "T2", "T3", "T4", "AvgTemp"], importances)
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.grid(True)
plt.show()
