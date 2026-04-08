import matplotlib.pyplot as plt
import numpy as np

# ================== DATA ==================

stations = ["Anand Vihar", "Dwarka Sec-8", "ITO", "Rohini"]

# R2 Scores
cnn_r2 = [0.7328, 0.7422, 0.7243, 0.7789]
lstm_r2 = [0.7940, 0.7885, 0.7793, 0.8112]
cnn_lstm_r2 = [0.7736, 0.7697, 0.7622, 0.7931]
mlp_r2 = [0.9192, 0.9369, 0.9408, 0.9408]

# MAE
cnn_mae = [31.06, 31.02, 31.07, 29.98]
lstm_mae = [25.69, 26.96, 26.31, 25.43]
cnn_lstm_mae = [26.82, 27.85, 27.46, 29.03]
mlp_mae = [20.13, 18.62, 17.76, 18.69]

# RMSE
cnn_rmse = [51.11, 50.93, 51.55, 51.11]
lstm_rmse = [44.88, 46.13, 46.12, 47.23]
cnn_lstm_rmse = [47.06, 48.13, 47.87, 49.45]
mlp_rmse = [28.07, 25.17, 23.85, 26.43]

# X-axis positions
x = np.arange(len(stations))
width = 0.2

# ================== PLOTTING FUNCTION ==================
def plot_comparison(metric_name, cnn, lstm, cnn_lstm, mlp):
    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - 1.5*width, cnn, width, label="CNN")
    ax.bar(x - 0.5*width, lstm, width, label="LSTM")
    ax.bar(x + 0.5*width, cnn_lstm, width, label="CNN-LSTM")
    ax.bar(x + 1.5*width, mlp, width, label="MLP")

    # Labels
    ax.set_xlabel("Monitoring Stations", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)

    # Title
    ax.set_title(f"{metric_name} Comparison Across Models and Stations", fontsize=14)

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=15)

    # Dynamic headroom (fixes R² overlap)
    max_val = max(max(cnn), max(lstm), max(cnn_lstm), max(mlp))
    if "R²" in metric_name:
        ax.set_ylim(0, max_val * 1.5)
    else:
        ax.set_ylim(0, max_val * 1.3)

    # Slightly shrink plot for legend space
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Legend (inside, no overlap)
    ax.legend(title="Models", loc="upper right", frameon=True)

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Prevent title cut
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

# ================== PLOTS ==================

# 1) R² Score
plot_comparison("R² Score", cnn_r2, lstm_r2, cnn_lstm_r2, mlp_r2)

# 2) MAE
plot_comparison("Mean Absolute Error (MAE)", cnn_mae, lstm_mae, cnn_lstm_mae, mlp_mae)

# 3) RMSE
plot_comparison("Root Mean Squared Error (RMSE)", cnn_rmse, lstm_rmse, cnn_lstm_rmse, mlp_rmse)