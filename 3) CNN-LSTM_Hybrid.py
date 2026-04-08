import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout

# ================== USER INPUT ==================
FILE_PATH = "Dwarka-Sec-8_DPCC.csv"
STATION_NAME = "Dwarka Sector-8"

# ================== LOAD DATA ==================
df = pd.read_csv(FILE_PATH)
df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])

df = df.sort_values('datetimeLocal')
df['year'] = df['datetimeLocal'].dt.year

# ================== CYCLIC FEATURES ==================
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

# ================== SEQUENCE ==================
def seq(df, step=24):
    X, y, time = [], [], []

    data_X = df[features].values
    data_y = df[target].values
    data_time = df['datetimeLocal'].values

    for i in range(len(df) - step):
        X.append(data_X[i:i+step])
        y.append(data_y[i+step])
        time.append(data_time[i+step])

    return np.array(X), np.array(y), np.array(time)

gc.collect()

X_train, y_train, _ = seq(train)
X_test, y_test, time_test = seq(test)

# ================== CNN-LSTM MODEL ==================
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(2),

    Conv1D(32, 3, activation='relu'),

    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),

    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ================== TRAIN ==================
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ================== PREDICTION ==================
pred = model.predict(X_test)

# ================== INVERSE ==================
y_test_actual = scalerY.inverse_transform(y_test.reshape(-1,1))
pred_actual = scalerY.inverse_transform(pred)

# ================== METRICS ==================
print(f"\nResults for CNN-LSTM Hybrid: Station - {STATION_NAME}:")
print("R2:", r2_score(y_test_actual, pred_actual))
print("MAE:", mean_absolute_error(y_test_actual, pred_actual))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, pred_actual)))

# =========================================================
# 1) PM2.5 vs Time (2025)
# =========================================================

test_actual_full = scalerY.inverse_transform(test[[target]])

plt.figure(figsize=(14,5))

plt.plot(test['datetimeLocal'], test_actual_full, label="Actual PM2.5 (2025)", alpha=0.6)
plt.plot(time_test, pred_actual, label="Predicted PM2.5 (2025)", linewidth=2)

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
plt.plot(y_test_actual[:500], label="Actual")
plt.plot(pred_actual[:500], label="Predicted")

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

plt.tight_layout()  # ✅ FIX

plt.show()

# =========================================================
# 4) RESIDUAL PLOT
# =========================================================

residuals = y_test_actual.flatten() - pred_actual.flatten()

plt.figure()
plt.scatter(pred_actual, residuals, alpha=0.3)
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