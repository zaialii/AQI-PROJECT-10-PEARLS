## Prediction script modified by Muhammad Ali Khan
import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model

# -------------------
# Configs & Paths
# -------------------
LSTM_MODEL_DIR = "lstm_model"
MODEL_PATH = os.path.join(LSTM_MODEL_DIR, "lstm_aqi_model.keras")
SCALER_X_PATH = os.path.join(LSTM_MODEL_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(LSTM_MODEL_DIR, "scaler_y.pkl")
DATA_PATH = "processed_data/daily_karachi_preprocessed.csv"

SEQ_LEN = 7

# -------------------
# Load model & scalers
# -------------------
model = load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# -------------------
# Load & prepare data
# -------------------
df = pd.read_csv(DATA_PATH)
df["ds"] = pd.to_datetime(df["date"])
df["y"] = df["AQI"]

features = [
    'AQI', 'PM10', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity',
    'Precipitation', 'month', 'log_PM2.5', 'log_CO', 'season_Spring',
    'season_Summer', 'season_Winter', 'weekday_1', 'weekday_2', 'weekday_3',
    'weekday_4', 'weekday_5', 'weekday_6', 'AQI_t+1', 'AQI_t+2', 'AQI_t+3',
    'AQI_lag_1', 'AQI_lag_2', 'AQI_roll_mean_3', 'AQI_roll_std_3',
    'AQI_diff'
] + [col for col in df.columns if col.startswith("season_") or col.startswith("weekday_")]

# Ensure latest valid sequence
df_latest = df[features].copy()
X_all = scaler_X.transform(df_latest)
last_seq = X_all[-SEQ_LEN:]

# -------------------
# Forecast Next 3 Days
# -------------------
predictions = []
input_seq = last_seq.copy()

for i in range(3):
    input_batch = np.expand_dims(input_seq, axis=0)  # shape: (1, 7, num_features)
    pred_scaled = model.predict(input_batch)
    pred = scaler_y.inverse_transform(pred_scaled)[0][0]
    predictions.append(pred)

    # Create dummy next step with predicted AQI
    next_input = input_seq[-1].copy()
    # Replace AQI (assume it's the first column) with predicted value
    next_input[0] = pred_scaled[0][0]

    # Update input_seq with new step
    input_seq = np.concatenate([input_seq[1:], [next_input]])

# -------------------
# Prepare output
# -------------------
last_date = df["ds"].max()
future_dates = [last_date + timedelta(days=i+1) for i in range(3)]
results = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Predicted_AQI": np.round(predictions, 2)
})

print("\n📈 Next 3 Days AQI Prediction:")
print(results.to_string(index=False))

# Save results
os.makedirs("predictions", exist_ok=True)
results.to_csv("predictions/karachi_next3days_pred.csv", index=False)


def predict_next_3_days():
    # Load model & scalers
    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Load & prepare data
    df = pd.read_csv(DATA_PATH)
    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = df["AQI"]

    features = [
        'AQI', 'PM10', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity',
        'Precipitation', 'month', 'log_PM2.5', 'log_CO', 'season_Spring',
        'season_Summer', 'season_Winter', 'weekday_1', 'weekday_2', 'weekday_3',
        'weekday_4', 'weekday_5', 'weekday_6', 'AQI_t+1', 'AQI_t+2', 'AQI_t+3',
        'AQI_lag_1', 'AQI_lag_2', 'AQI_roll_mean_3', 'AQI_roll_std_3',
        'AQI_diff'
    ] + [col for col in df.columns if col.startswith("season_") or col.startswith("weekday_")]

    # Ensure latest valid sequence
    df_latest = df[features].copy()
    X_all = scaler_X.transform(df_latest)
    last_seq = X_all[-SEQ_LEN:]

    # Forecast Next 3 Days
    predictions = []
    input_seq = last_seq.copy()

    for i in range(3):
        input_batch = np.expand_dims(input_seq, axis=0)
        pred_scaled = model.predict(input_batch)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)

        # Create dummy next step
        next_input = input_seq[-1].copy()
        next_input[0] = pred_scaled[0][0]
        input_seq = np.concatenate([input_seq[1:], [next_input]])

    # Prepare output
    last_date = df["ds"].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(3)]
    results = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicted_AQI": np.round(predictions, 2)
    })

    return results

if __name__ == "__main__":
    results = predict_next_3_days()
    print("\n📈 Next 3 Days AQI Prediction:")
    print(results.to_string(index=False))
