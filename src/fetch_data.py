import requests
import pandas as pd
from datetime import date, timedelta
import os
from tqdm import tqdm

LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"

def fetch_day_data(day):
    # AQI and Pollutants
    try:
        air_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={day}&end_date={day}"
            "&hourly=us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
            f"&timezone={TIMEZONE}"
        )
        air_response = requests.get(air_url)
        if air_response.status_code != 200 or air_response.text.strip() == "":
            raise ValueError(f"Empty or failed air API response for {day}")

        air_data = air_response.json()["hourly"]

        # same for weather
        weather_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={day}&end_date={day}"
            "&hourly=temperature_2m,relative_humidity_2m,precipitation"
            f"&timezone={TIMEZONE}"
        )
        weather_response = requests.get(weather_url)
        if weather_response.status_code != 200 or weather_response.text.strip() == "":
            raise ValueError(f"Empty or failed weather API response for {day}")

        weather_data = weather_response.json()["hourly"]

        # continue normally...
        df_air = pd.DataFrame(air_data)
        df_weather = pd.DataFrame(weather_data)
        df = pd.merge(df_air, df_weather, on='time')
        df["time"] = pd.to_datetime(df["time"])
        return df

    except Exception as e:
        print(f"Failed on {day}: {e}")
        return None  # return None so you can skip that day

def process_daily(day, df):
    daily = df.set_index("time").resample("D").mean()
    daily["date"] = day
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
    start = date(2023, 1, 1)
    today = date.today()
    all_days = []

    for single in tqdm(pd.date_range(start, today), desc="Fetching days"):
        day = single.date().isoformat()
        
        raw = fetch_day_data(day)  # internally handled

        if raw is None or raw.empty:
            continue  # Skip failed/empty day

        try:
            day_data = process_daily(day, raw)
            if day_data is not None and not day_data.empty:
                all_days.append(day_data)
        except Exception as e:
            print(f"Failed to process {day}: {e}")
            continue

    valid_days = [df for df in all_days if df is not None and not df.empty]

    df_all = pd.concat(valid_days, ignore_index=True)
    df_all = df_all.sort_values("date")
    df_all["Next_Day_AQI"] = df_all["AQI"].shift(-1)
    df_all.to_csv("data/karachi_daily_aqi_weather.csv", index=False)
    print("✅ Data saved to data/karachi_daily_aqi_weather.csv")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
