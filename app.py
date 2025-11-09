import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
from src.predict import predict_next_3_days
from src.create_lime import generate_lime
import plotly.io as pio


st.set_page_config(layout="wide", page_title="10-Pearls Karachi AQI Predictor", page_icon="🌆")

st.markdown("""
<style>
.stApp {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/017/648/540/small/karachi-skyline-illustration-vector.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.main > div {
    background-color: rgba(255, 255, 255, 0.6);
    padding: 3rem;
    border-radius: 15px;
    box-shadow: 4px 4px 20px rgba(0,0,0,0.1);
}
html, body, [class*="css"] {
    color: #002B5B !important;
    font-family: 'Helvetica Neue', sans-serif;
}
[data-testid="stMetricLabel"] > div {
    color: #002B5B !important;
    font-weight: 900 !important;
    font-size: 2.1rem !important;
}
[data-testid="stMetricValue"] > div {
    color: #005f73 !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#002B5B; margin-bottom:0.1rem;'>🌆 10-Pearls Karachi AQI Predictor</h1>
    <h3 style='color:#005f73; margin-top:0rem;'>by Muhammad Ali Khan</h3>
    <hr style='border: 2px solid #005f73; width: 60%; margin:auto;'>
</div>
""", unsafe_allow_html=True)

pio.templates.default = "plotly_white"


@st.cache_data
def load_data():
    df = pd.read_csv("processed_data/daily_karachi_preprocessed.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()


st.markdown("""
<style>
div[data-baseweb="tab-list"] { justify-content:center !important; }
button[data-baseweb="tab"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #002B5B !important;
}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "📈 AQI Trends", "💨 Pollutants & LIME", "🧠 Insights", "🕒 Logs"])


with tabs[0]:
    latest_row = df.sort_values("date").iloc[-1]
    forecast_df = predict_next_3_days()

    st.markdown(f"<h3 style='text-align:center; color:#002B5B;'>Date: {latest_row['date'].strftime('%d %B %Y')}</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_row['AQI'],
            title={'text': "Current AQI", 'font': {'size': 28, 'color': '#002B5B'}},
            gauge={
                'axis': {'range': [0, 500], 'tickwidth': 2, 'tickcolor': "#002B5B"},
                'bar': {'color': "#00B4D8"},
                'steps': [
                    {'range': [0, 50], 'color': "#90E0EF"},
                    {'range': [51, 100], 'color': "#48CAE4"},
                    {'range': [101, 150], 'color': "#00B4D8"},
                    {'range': [151, 200], 'color': "#0096C7"},
                    {'range': [201, 300], 'color': "#0077B6"},
                    {'range': [301, 500], 'color': "#023E8A"},
                ],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=1,b=1))
        st.plotly_chart(fig, use_container_width=True)

    def aqi_message(aqi):
        if aqi <= 50: return "🌿 Fresh air! Perfect outdoor day."
        if aqi <= 100: return "😊 Air quality is good; sensitive people may notice."
        if aqi <= 150: return "😐 Moderate pollution. Limit outdoor activity if sensitive."
        if aqi <= 200: return "😷 Unhealthy for sensitive groups."
        if aqi <= 300: return "🚫 Unhealthy! Stay indoors."
        return "☠️ Hazardous! Avoid going outside."

    st.markdown(f"<div style='background-color:#CAF0F8; padding:1rem; border-radius:12px; text-align:center; font-size:1.5rem;'>{aqi_message(latest_row['AQI'])}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h2 style='text-align:center; color:#005f73;'>Today's Pollutants & Weather</h2>", unsafe_allow_html=True)
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "Temperature", "Humidity", "Precipitation"]
        metrics_per_row = 3
        for i in range(0, len(pollutants), metrics_per_row):
            cols = st.columns(metrics_per_row)
            for j in range(metrics_per_row):
                if i+j < len(pollutants):
                    col = cols[j]
                    label = pollutants[i+j]
                    value = latest_row[label]
                    unit = (" µg/m³" if label in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
                            else "%" if label=="Humidity"
                            else "°C" if label=="Temperature"
                            else "mm")
                    col.markdown(f"<div style='text-align:center; font-size:1.4rem; font-weight:600; color:#002B5B;'>{label}: {value:.1f}{unit}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #005f73;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; color:#002B5B;'>Next 3-Day AQI Forecast</h1>", unsafe_allow_html=True)

    forecast_cols = st.columns(3)
    def aqi_category(aqi):
        if aqi <= 50: return "🌿 Very Good"
        if aqi <= 100: return "😊 Good"
        if aqi <= 150: return "😐 Moderate"
        if aqi <= 200: return "😷 Unhealthy for Sensitive"
        if aqi <= 300: return "🚫 Unhealthy"
        return "☠️ Hazardous"

    for i, col in enumerate(forecast_cols):
        date = pd.to_datetime(forecast_df.loc[i,"Date"]).strftime("%d %b %Y")
        aqi_val = forecast_df.loc[i,"Predicted_AQI"]
        category = aqi_category(aqi_val)
        col.markdown(f"""
        <div style='background-color:#ADE8F4; padding:1rem; border-radius:12px; text-align:center; box-shadow:2px 2px 10px rgba(0,0,0,0.1);'>
            <div style='font-size:1.8rem; font-weight:600;'>{date}</div>
            <div style='font-size:2.8rem; font-weight:bold; color:#0077B6;'>{aqi_val:.0f}</div>
            <div style='font-size:1.5rem; font-weight:bold;'>{category}</div>
        </div>
        """, unsafe_allow_html=True)


with tabs[1]:
    st.markdown("<h1 style='text-align:center; color:#002B5B;'>📈 AQI Trends Over Time</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    time_filter = None
    if col1.button("7 Days"): time_filter=7
    if col2.button("Last Month"): time_filter=30
    if col3.button("90 Days"): time_filter=90
    if col4.button("6 Months"): time_filter=180
    if col5.button("Last Year"): time_filter=365
    if col6.button("All Data"): time_filter=None

    latest_date = df["date"].max()
    filtered_df = df[df["date"] >= latest_date - pd.Timedelta(days=time_filter)] if time_filter else df.copy()

    # AQI line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["AQI"], mode="lines+markers",
                             line=dict(color="#0077B6", width=3), marker=dict(size=6), name="AQI"))
    fig.update_layout(template="plotly_white", height=700,
                      title=dict(text="AQI Over Time", font=dict(size=22,color="#002B5B"), x=0.5),
                      xaxis=dict(title="Date", tickfont=dict(color="#002B5B")),
                      yaxis=dict(title="AQI", range=[0,500], tickfont=dict(color="#002B5B")))
    st.plotly_chart(fig, use_container_width=True)


with tabs[2]:
    st.markdown("<h1 style='text-align:center; color:#002B5B;'>💨 Pollutants & LIME Features</h1>", unsafe_allow_html=True)

    # Radar & Pie charts
    pollutants = ['PM2.5','PM10','NO2','O3','CO','SO2']
    who_limits = pd.Series({'PM2.5':15,'PM10':45,'NO2':25,'O3':100,'CO':4000,'SO2':40})
    karachi_avg = df[pollutants].mean()
    karachi_ratio = karachi_avg/who_limits

    # Radar
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=list(karachi_ratio)+[karachi_ratio.iloc[0]],
                                        theta=pollutants+[pollutants[0]], fill='toself',
                                        name='Karachi Avg / WHO', line_color='#0077B6'))
    radar_fig.add_trace(go.Scatterpolar(r=[1]*len(pollutants)+[1],
                                        theta=pollutants+[pollutants[0]], fill='toself',
                                        name='WHO Safe', line_color='#00B4D8'))
    radar_fig.update_layout(template="plotly_white", height=700, title=dict(text="Risk Ratio: Karachi vs WHO", font=dict(size=20,color="#002B5B")))

    # Pie
    pie_fig = px.pie(values=karachi_avg, names=pollutants, title="Karachi Pollutants Avg",
                     color_discrete_map={'PM2.5':'#023E8A','PM10':'#0077B6','NO2':'#0096C7','O3':'#00B4D8','CO':'#48CAE4','SO2':'#90E0EF'})
    pie_fig.update_layout(title_font_color="#002B5B", height=700)

    col1, col2 = st.columns(2)
    col1.plotly_chart(pie_fig, use_container_width=True)
    col2.plotly_chart(radar_fig, use_container_width=True)

    # LIME
    lime_res = generate_lime()
    lime_df = pd.read_csv(lime_res['csv_path'])
    fig_lime = px.bar(lime_df, x='Contribution', y='Feature', orientation='h', color='Contribution', color_continuous_scale=px.colors.diverging.RdBu,
                      title="LIME Feature Contributions")
    fig_lime.update_layout(template="plotly_white", height=700, title_font=dict(size=20,color="#002B5B"))
    st.plotly_chart(fig_lime, use_container_width=True)


with tabs[3]:
    st.markdown("<h1 style='text-align:center; color:#002B5B;'>🧠 General Insights</h1>", unsafe_allow_html=True)
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5,6])
    weekend_aqi = df.groupby('is_weekend')['AQI'].mean()
    st.write(f"Weekday AQI: {weekend_aqi[False]:.2f}, Weekend AQI: {weekend_aqi[True]:.2f}")


with tabs[4]:
    st.markdown("<h2 style='text-align:center; color:#002B5B;'>🕒 Pipeline Logs</h2>", unsafe_allow_html=True)
    log_path = "lstm_model/update_log.txt"
    if os.path.exists(log_path):
        with open(log_path,"r") as f:
            logs = [json.loads(line) for line in f.readlines()]
        logs_df = pd.DataFrame(logs)
        st.dataframe(logs_df)
    else:
        st.info("No logs found.")


st.markdown("""
<div style='background-color:#00B4D8; padding:15px; border-radius:12px; text-align:center; color:white;'>
    Made with by Muhammad Ali Khan — Data Scientist | AI Engineer <br/>
    <a href="https://www.linkedin.com/in/muhammad-ali-khan-9b6900253/" target="_blank" style='color:white;'>LinkedIn</a> |
    <a href="https://github.com/zaialii" target="_blank" style='color:white;'>GitHub</a> |
</div>
""", unsafe_allow_html=True)
