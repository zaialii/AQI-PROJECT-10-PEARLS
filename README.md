# 🌿 Pearls’ Karachi AQI Forecast

*Predict Tomorrow’s Air Quality, Today.*

**Technologies & Tools Used:**  

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python) 
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/-Keras-red?style=flat-square&logo=keras)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly)
![GitHub Actions](https://img.shields.io/badge/-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions)

---

## 🚀 Project Summary

**Pearls’ Karachi AQI Forecast** is an end-to-end solution for monitoring and predicting air quality in Karachi. By combining **real-time data collection, machine learning models, and interactive visualizations**, this project provides accurate 3-day AQI forecasts, pollutant insights, and WHO guideline comparisons. LIME is used to make predictions interpretable, building trust in AI-driven forecasts.

### Core Benefits

- **Automated Data Handling:** Fetches, cleans, and structures AQI & weather data daily.  
- **3-Day Forecasting:** LSTM model predicts the next 3 days of AQI.  
- **Transparency:** LIME explanations show which features drive predictions.  
- **Interactive Dashboard:** Explore trends, pollutants, and forecasts in a visual interface.  
- **WHO Standards Integration:** Compare local air quality against global safety limits.  
- **CI/CD Pipelines:** Automatically updates models and predictions daily.
- 
---

## ✨ Key Features

- **Current AQI & Pollutant Levels:** Daily updated visualizations.  
- **3-Day AQI Prediction:** Forecast generated using **LSTM deep learning**, refreshed every day.  
- **Seasonal & Trend Analysis:** Explore month-wise, weekday, and yearly patterns.  
- **Pollutant Risk Visualization:** Radar and pie charts versus WHO thresholds.  
- **LIME Interpretability:** Understand feature contributions for each prediction.  
- **Transparent Logs:** Track data updates, model performance, and pipeline status.  
- **Responsive Design:** Clean, mobile-friendly interface with Plotly visuals.  
- **Automated Updates:** GitHub Actions handles CI/CD for data & model updates.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Plotly, custom HTML/CSS)  
- **Machine Learning:** Python, TensorFlow (LSTM), scikit-learn, LIME, pandas, NumPy  
- **Deployment:** Render.com (free hosting)  
- **Data Sources:** Open-Meteo AQI & Weather APIs  
- **DevOps:** GitHub Actions for automated daily updates

---

## 📁 Project Structure

```
├── app.py                         # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── src/                            # ML & data pipeline scripts
│ ├── update_daily_data.py
│ ├── preprocess_daily_data.py
│ ├── lstm_model_training.py
│ ├── predict.py
│ ├── fetch_data.py
│ └── create_lime.py
├── data/                              # Raw AQI & weather data
├── processed_data/                      # Cleaned & feature-engineered data
├── predictions/                        # Predicted AQI for next 3 days
├── lstm_model/                      # Saved LSTM model & scalers
├── lime_explanations/               # LIME outputs for interpretability
├── notebooks/                      # Jupyter notebooks for EDA & visualization
└── .github/workflows/                 # GitHub Actions CI/CD workflow
```

## ⚡ Workflow Pipeline

1. **Data Fetch:** `src/update_daily_data.py` pulls daily AQI & weather data.  
2. **Processing:** `src/preprocess_daily_data.py` cleans, fills missing values, caps outliers, and creates features.  
3. **Model Training:** `src/lstm_model_training.py` trains the LSTM and saves best model metrics.  
4. **Prediction:** `src/predict.py` predicts 3-day AQI and saves results.  
5. **Explainability:** `src/create_lime.py` generates LIME reports & charts for predictions.  
6. **Dashboard:** `app.py` provides an interactive interface with visualizations.  
7. **CI/CD:** `.github/workflows/aqi_pipeline.yml` automates daily updates and predictions.

---

## 🖥️ Run Locally

```bash
git clone https://github.com/zaialii/AQI-PROJECT-10-PEARLS.git
cd 10-pearls-AQI-prediction-Karachi
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Deploy to Render

1.  Visit [https://render.com](https://render.com)
2.  Click **"New Web Service"**
3.  Connect your GitHub repo
4.  Set configuration:
    -   **Build Command:** `pip install -r requirements.txt`
    -   **Start Command:** `streamlit run app.py`
    -   **Instance Type:** Free (Starter)
5.  Deploy! (CI/CD pipeline will keep it up-to-date.)

> **Important:** Configure environment variables in Render.com if your application requires API keys or sensitive information.

---

## 🧬 Example: CI/CD Workflow

```
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 3 * * *'  # Every day at 3 AM UTC (8 AM PKT)
```


> **Note:** Modify the cron schedule according to your needs. Ensure that the GitHub Actions workflow has the necessary permissions to push changes to your repository.

---

## 🌍 Data Sources

-   [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
-   [Open-Meteo Weather Archive API](https://open-meteo.com/en/docs#archive)

> **Note:** You may need to sign up for API keys if the usage exceeds the free tier limits. Add the API keys as environment variables in your deployment environment.

---

## 🎯 Upcoming Enhancements

-   🌐 Real-time AQI API integration (e.g. AirVisual, WAQI)
-   📱 PWA support for mobile alerts/notifications
-   🧠 Enhanced Model explainability (SHAP/LIME insights)

---

## 👩‍💻 Author

Made by [Muhammad Ali Khan] 
_Data Scientist • AI Engineer_  

---

## 📄 License

Distributed under the MIT License — use, modify, and contribute freely.

---
