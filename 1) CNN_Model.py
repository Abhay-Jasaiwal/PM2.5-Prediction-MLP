import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

# ================== DATA FILES ==================
files = [
    "ITO_CPCB.csv",
    "Anand-Vihar_DPCC.csv",
    "Dwarka-Sec-8_DPCC.csv",
    "Jahangirpuri_DPCC.csv",
    "Mandir-Marg_DPCC.csv",
    "Okhla-Phase-2_DPCC.csv",
    "Punjabi-Bagh_DPCC.csv",
    "RK-Puram_DPCC.csv",
    "Rohini_DPCC.csv",
    "Vivek-Vihar_DPCC.csv"
]

# ================== LOAD DATA ==================
dfs = []

for i, file in enumerate(files):
    df_temp = pd.read_csv(file)
    df_temp['station'] = i
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])
df = df.sort_values(['station','datetimeLocal'])
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
    'day_of_week_sin','day_of_week_cos',
    'station'
]

target = 'PM2.5'

# ================== SCALING ==================
scalerX = MinMaxScaler()
scalery = MinMaxScaler()

df[features] = scalerX.fit_transform(df[features])
df[[target]] = scalery.fit_transform(df[[target]])

# keep station as int
df['station'] = df['station'].astype(int)

df[features] = df[features].astype(np.float32)
df[target] = df[target].astype(np.float32)

# ================== SPLIT ==================
train = df[df['year'] < 2025]
test = df[df['year'] == 2025]

# ================== SEQUENCE ==================
def seq(df, step=24):
    X, y = [], []

    for st in df['station'].unique():
        d = df[df['station'] == st].reset_index(drop=True)

        data_X = d[features].values
        data_y = d[target].values

        for i in range(len(d) - step):
            X.append(data_X[i:i+step])
            y.append(data_y[i+step])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

gc.collect()

X_train, y_train = seq(train)
X_test, y_test = seq(test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ================== CNN MODEL ==================
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ================== TRAIN ==================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ================== PREDICTION ==================
pred = model.predict(X_test)

# ================== METRICS ==================
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\nCNN RESULTS:")
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# ================== INVERSE TRANSFORM ==================
y_test_actual = scalery.inverse_transform(y_test.reshape(-1,1))
pred_actual = scalery.inverse_transform(pred)

# ================== 1) ACTUAL VS PREDICTED ==================
plt.figure(figsize=(12,5))
plt.plot(y_test_actual[:500], label="Actual PM2.5 (µg/m³)")
plt.plot(pred_actual[:500], label="Predicted PM2.5 (µg/m³)")

plt.xlabel("Time Steps")
plt.ylabel("PM2.5 Concentration (µg/m³)")
plt.title("Actual vs Predicted PM2.5 (CNN Model)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== 2) LOSS CURVE ==================
plt.figure()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss (CNN Model)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== 3) RESIDUAL PLOT ==================
residuals = y_test_actual.flatten() - pred_actual.flatten()

plt.figure()

plt.scatter(pred_actual, residuals, alpha=0.3, color='blue', label="Residuals")
plt.axhline(0, color='red', linestyle='--', linewidth=2, label="Zero Error Line")

plt.xlabel("Predicted PM2.5 (µg/m³)")
plt.ylabel("Residuals (µg/m³)")
plt.title("Residual Plot (CNN Model)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== 4) ERROR DISTRIBUTION ==================
plt.figure()

plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')

plt.xlabel("Prediction Error (µg/m³)")
plt.ylabel("Frequency")
plt.title("Error Distribution (CNN Model)")

plt.grid(True)
plt.tight_layout()
plt.show()

