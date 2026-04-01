import numpy as np
import matplotlib.pyplot as plt

# ================== MODELS ==================
models = ["CNN", "LSTM", "CNN-LSTM", "BiLSTM", "MLP"]

# ================== YOUR ACTUAL RESULTS ==================

r2 = [
    0.7712997794151306,
    0.7794101238250732,
    0.7817565202713013,
    0.7787361145019531,
    0.9667447405325381
]

mae = [
    0.03325958922505379,
    0.0298842191696167,
    0.029593966901302338,
    0.029160797595977783,
    0.015297326568593771
]

rmse = [
    0.0553814275373072,
    0.054390569504948166,
    0.054100524551543246,
    0.0544736055426154,
    0.02109061802882191
]

# ================== PLOT ==================
x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(12,6))

plt.bar(x - width, r2, width, label="R² Score")
plt.bar(x, mae, width, label="MAE")
plt.bar(x + width, rmse, width, label="RMSE")

plt.xticks(x, models)

plt.xlabel("Models")
plt.ylabel("Metric Values")
plt.title("Model Performance Comparison")

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()