import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
    temp = pd.read_csv(file)
    temp['station'] = i
    dfs.append(temp)

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

# ================== SPLIT ==================
train = df[df['year'] < 2025].copy()
test = df[df['year'] == 2025].copy()

# ================== SCALING ==================
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

train.loc[:, features] = scalerX.fit_transform(train[features])
test.loc[:, features] = scalerX.transform(test[features])

train.loc[:, target] = scalerY.fit_transform(train[[target]])
test.loc[:, target] = scalerY.transform(test[[target]])

# keep station integer
train['station'] = train['station'].astype(int)
test['station'] = test['station'].astype(int)

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
    epochs=50,
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
r2 = r2_score(y_true_actual, y_pred_actual)
mae = mean_absolute_error(y_true_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))

print("\nMLP RESULTS:")
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# ================== 1) ACTUAL VS PREDICTED ==================
plt.figure(figsize=(12,5))
plt.plot(y_true_actual[:500], label="Actual PM2.5 (µg/m³)")
plt.plot(y_pred_actual[:500], label="Predicted PM2.5 (µg/m³)")

plt.xlabel("Time Steps")
plt.ylabel("PM2.5 Concentration (µg/m³)")
plt.title("Actual vs Predicted PM2.5 (MLP Model)")

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
plt.title("Training vs Validation Loss (MLP Model)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== 3) RESIDUAL PLOT ==================
residuals = y_true_actual.flatten() - y_pred_actual.flatten()

plt.figure()

plt.scatter(y_pred_actual, residuals, alpha=0.3, color='blue', label="Residuals")
plt.axhline(0, color='red', linestyle='--', linewidth=2, label="Zero Error Line")

plt.xlabel("Predicted PM2.5 (µg/m³)")
plt.ylabel("Residuals (µg/m³)")
plt.title("Residual Plot (MLP Model)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== 4) ERROR DISTRIBUTION ==================
plt.figure()

plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')

plt.xlabel("Prediction Error (µg/m³)")
plt.ylabel("Frequency")
plt.title("Error Distribution (MLP Model)")

plt.grid(True)
plt.tight_layout()
plt.show()