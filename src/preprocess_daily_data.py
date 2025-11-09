import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/karachi_daily_aqi_weather.csv"
PROCESSED_DATA_PATH = "processed_data/karachi_preprocessed_MAK.csv"

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def iqr_cap(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[column].clip(lower, upper)

def preprocess_data():
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    print("📥 Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Data loaded. Shape: {df.shape}")

    print("🧹 Removing duplicates...")
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"✅ Duplicates removed: {before - df.shape[0]} rows")

    print("🔧 Filling missing values with forward fill...")
    null_before = df.isnull().sum().sum()
    df = df.ffill()
    null_after = df.isnull().sum().sum()
    print(f"✅ Nulls before: {null_before}, after: {null_after}")

    print("🛠️ Parsing date and extracting features...")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["season"] = df["month"].apply(get_season)

    if "Next_Day_AQI" in df.columns:
        df.drop(columns=["Next_Day_AQI"], inplace=True)
        print("🗑️ Dropped column: Next_Day_AQI")

    print("🔁 Applying log transform...")
    for col in ["PM2.5", "CO"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
            print(f"✅ Log transformed: {col}")

    print("📏 Applying IQR capping...")
    for col in ["PM10", "SO2", "NO2", "O3", "Temperature", "Humidity", "Precipitation"]:
        if col in df.columns:
            df[col] = iqr_cap(df, col)
            print(f"✅ IQR capped: {col}")

    print("🎨 One-hot encoding season and weekday...")
    df = pd.get_dummies(df, columns=["season", "weekday"], drop_first=True)

    print("⏭️ Creating future AQI targets...")
    df["AQI_t+1"] = df["AQI"].shift(-1)
    df["AQI_t+2"] = df["AQI"].shift(-2)
    df["AQI_t+3"] = df["AQI"].shift(-3)

    print("➕ Adding lag, rolling, and diff features...")
    df["AQI_lag_1"] = df["AQI"].shift(1)
    df["AQI_lag_2"] = df["AQI"].shift(2)
    df["AQI_roll_mean_3"] = df["AQI"].rolling(3).mean().shift(1)
    df["AQI_roll_std_3"] = df["AQI"].rolling(3).std().shift(1)
    df["AQI_diff"] = df["AQI"].diff().shift(1)
    
    if "AQI" in df.columns:
        df["AQI_rolling_mean"] = df["AQI"].rolling(3).mean().fillna(df["AQI"])

    print("🧹 Final cleanup...")
    df = df.ffill()
    df.dropna(inplace=True)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"✅ Final shape after cleanup: {df.shape}")

    print("💾 Saving processed data...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"🎉 Preprocessing complete. Saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()
