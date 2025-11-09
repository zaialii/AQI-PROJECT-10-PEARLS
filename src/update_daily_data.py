import requests
import pandas as pd
from datetime import date, timedelta
import os

LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"
DATA_PATH = "data/karachi_daily_aqi_weather.csv"

def fetch_today_data(today):
    air_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={today}&end_date={today}"
        "&hourly=us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
        f"&timezone={TIMEZONE}"
    )
    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={today}&end_date={today}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation"
        f"&timezone={TIMEZONE}"
    )

    try:
        air_response = requests.get(air_url).json()
        weather_response = requests.get(weather_url).json()

        if "hourly" not in air_response or "hourly" not in weather_response:
            raise ValueError("Missing 'hourly' data in API response")

        air_data = air_response["hourly"]
        weather_data = weather_response["hourly"]

        if not air_data or not weather_data:
            raise ValueError("Empty hourly data")

    except Exception as e:
        print(f"❌ Failed to fetch data for {today}: {e}")
        return None

    df_air = pd.DataFrame(air_data)
    df_weather = pd.DataFrame(weather_data)
    df = pd.merge(df_air, df_weather, on="time")
    df["time"] = pd.to_datetime(df["time"])
    daily = df.set_index("time").resample("D").mean()

    daily["date"] = today
    daily = daily.rename(columns={
        'us_aqi': 'AQI',
        'pm2_5': 'PM2.5',
        'pm10': 'PM10',
        'nitrogen_dioxide': 'NO2',
        'sulphur_dioxide': 'SO2',
        'carbon_monoxide': 'CO',
        'ozone': 'O3',
        'temperature_2m': 'Temperature',
        'relative_humidity_2m': 'Humidity',
        'precipitation': 'Precipitation'
    })

    return daily[['date', 'AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Precipitation']]

def main():
    today = date.today().isoformat()

    # Load existing data
    if os.path.exists(DATA_PATH):
        df_old = pd.read_csv(DATA_PATH)
        if today in df_old['date'].values:
            print(f"✅ Data for {today} already exists.")
            return
    else:
        df_old = pd.DataFrame()

    # Fetch today's data
    df_new = fetch_today_data(today)
    if df_new is None:
        print("❌ Skipping update due to fetch error.")
        return
    print(f"✅ Fetched data for {today}")

    # Append and re-save
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('date')
    df_combined["Next_Day_AQI"] = df_combined["AQI"].shift(-1)
    df_combined.to_csv(DATA_PATH, index=False)

    print(f"✅ Updated file: {DATA_PATH}")


    # Append and re-save
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('date')
    df_combined["Next_Day_AQI"] = df_combined["AQI"].shift(-1)
    df_combined.to_csv(DATA_PATH, index=False)

    print(f"✅ Updated file: {DATA_PATH}")

if __name__ == "__main__":
    main()

