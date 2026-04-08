import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ================== USER INPUT ==================
FILE_PATH = "Rohini_DPCC.csv"
STATION_NAME = "Rohini"

# ================== LOAD DATA ==================
df = pd.read_csv(FILE_PATH)
df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])

df = df.sort_values('datetimeLocal')
df['year'] = df['datetimeLocal'].dt.year

# ================== CYCLIC ENCODING ==================
def cyc(df, col, max_val):
    df[col+'_sin'] = np.sin(2*np.pi*df[col]/max_val)
    df[col+'_cos'] = np.cos(2*np.pi*df[col]/max_val)
    return df

df = cyc(df,'hour',24)
df = cyc(df,'month',12)
df = cyc(df,'day_of_week',7)

# ================== FEATURES ==================
features = [
    'PM10','NO2','CO','SO2',
    'T2M','WS2M','RH2M','PRECTOTCORR',
    'crop_burning','winter_inversion','diwali',
    'hour_sin','hour_cos',
    'month_sin','month_cos',
    'day_of_week_sin','day_of_week_cos'
]

target = 'PM2.5'

# ================== SPLIT ==================
train = df[df['year'] < 2025].copy()
test = df[df['year'] == 2025].copy()

# ================== SCALING ==================
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

train[features] = scalerX.fit_transform(train[features])
test[features] = scalerX.transform(test[features])

train[target] = scalerY.fit_transform(train[[target]])
test[target] = scalerY.transform(test[[target]])

# ================== MODEL ==================
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ================== TRAIN ==================
history = model.fit(
    train[features], train[target],
    epochs=100,
    batch_size=64,
    validation_data=(test[features], test[target]),
    verbose=1
)

# ================== PREDICTION ==================
pred = model.predict(test[features])

# ================== INVERSE ==================
y_true_actual = scalerY.inverse_transform(test[[target]])
y_pred_actual = scalerY.inverse_transform(pred)

# ================== METRICS ==================
print(f"\nResults for MLP: Station - {STATION_NAME}:")
print("R2:", r2_score(y_true_actual, y_pred_actual))
print("MAE:", mean_absolute_error(y_true_actual, y_pred_actual))
print("RMSE:", np.sqrt(mean_squared_error(y_true_actual, y_pred_actual)))

# =========================================================
# 1) PM2.5 vs Time (2025)
# =========================================================

plt.figure(figsize=(14,5))

plt.plot(test['datetimeLocal'], y_true_actual, label="Actual PM2.5 (2025)", alpha=0.6)
plt.plot(test['datetimeLocal'], y_pred_actual, label="Predicted PM2.5 (2025)", linewidth=2)

plt.xlabel("Time (2025)")
plt.ylabel("PM2.5 (µg/m³)")
plt.title(f"PM2.5 vs Time (2025) - Actual vs Predicted ({STATION_NAME})")

plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 2) TIME STEPS GRAPH
# =========================================================

plt.figure(figsize=(12,5))
plt.plot(y_true_actual[:500], label="Actual")
plt.plot(y_pred_actual[:500], label="Predicted")

plt.xlabel("Time Steps")
plt.ylabel("PM2.5 (µg/m³)")
plt.title(f"PM2.5 vs Time Steps ({STATION_NAME})")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 3) LOSS CURVE
# =========================================================

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Loss Curve ({STATION_NAME})")

plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 4) RESIDUAL PLOT
# =========================================================

residuals = y_true_actual.flatten() - y_pred_actual.flatten()

plt.figure()
plt.scatter(y_pred_actual, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted PM2.5")
plt.ylabel("Residuals")
plt.title(f"Residual Plot ({STATION_NAME})")

plt.grid(True)
plt.show()

# =========================================================
# 5) ERROR DISTRIBUTION
# =========================================================

plt.figure()
plt.hist(residuals, bins=50)

plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title(f"Error Distribution ({STATION_NAME})")

plt.grid(True)
plt.show()