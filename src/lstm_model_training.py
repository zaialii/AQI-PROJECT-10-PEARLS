## Trained / modified by Muhammad Ali Khan for 10Pearls
import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Configs & Paths
# -------------------------
LSTM_MODEL_DIR = "lstm_model"
os.makedirs(LSTM_MODEL_DIR, exist_ok=True)


MODEL_PATH = os.path.join(LSTM_MODEL_DIR, "lstm_aqi_model.keras")
SCALER_X_PATH = os.path.join(LSTM_MODEL_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(LSTM_MODEL_DIR, "scaler_y.pkl")
METRICS_PATH = os.path.join(LSTM_MODEL_DIR, "metrics.json")
LOG_PATH = os.path.join(LSTM_MODEL_DIR, "update_log.txt")

# ----------------------
# Load and Prepare Data
# ----------------------
df = pd.read_csv("processed_data/daily_karachi_preprocessed.csv")
df["ds"] = pd.to_datetime(df["date"])
df["y"] = df["AQI"]

features = [
     'AQI', 'PM10', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity',
       'Precipitation', 'month', 'log_PM2.5', 'log_CO', 'season_Spring',
       'season_Summer', 'season_Winter', 'weekday_1', 'weekday_2', 'weekday_3',
       'weekday_4', 'weekday_5', 'weekday_6', 'AQI_t+1', 'AQI_t+2', 'AQI_t+3',
       'AQI_lag_1', 'AQI_lag_2', 'AQI_roll_mean_3', 'AQI_roll_std_3',
       'AQI_diff'] + [col for col in df.columns if col.startswith("season_") or col.startswith("weekday_")]

# ----------------------
# Scale the Data
# ----------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_all = scaler_X.fit_transform(df[features])
y_all = scaler_y.fit_transform(df[["y"]])

# ----------------------
# Create Sequences
# ----------------------
def create_sequences(X, y, seq_length=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 7
X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)

split_idx = len(X_seq) - 30
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# ----------------------
# Build & Train New Model
# ----------------------
karachi_lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.1, callbacks=[es], verbose=0)

# ----------------------
# Evaluate New Model
# ----------------------
y_pred_scaled = karachi_lstm_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n📊 Evaluation on Test Set:")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

# ----------------------
# Compare with Old Model (if exists)
# ----------------------
update_model = True
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        old_metrics = json.load(f)
    if rmse >= old_metrics['RMSE']:
        update_model = False
        print("❌ New model did NOT outperform the previous one. Not saving.")
    else:
        print("✅ New model is better. Saving.")

# ----------------------
# Save Model if Better
# ----------------------

if update_model:
    karachi_lstm_model.save(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("💾 Model, scalers, and metrics updated.")
else:
    print("❌ Model not saved. Performance not improved.")

# ✅ LOGGING BLOCK
log_entry = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "status": "UPDATED" if update_model else "NOT UPDATED",
    "model_name": "lstm_aqi_model.keras",
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 4),
    "note": "EarlyStopping(patience=20)"
}

with open(LOG_PATH, "a") as log:
    log.write(json.dumps(log_entry) + "\n")

# Show last update
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as log:
        lines = log.readlines()
        print("\n🕒 Last Update Log:")
        print(lines[-1].strip())
