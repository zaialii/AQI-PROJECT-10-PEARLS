Karachi AQI Forecast — Muhammad Ali Khan

Breathe Easier with Tomorrow’s Air Quality Insights

A comprehensive tool to monitor and predict Karachi’s Air Quality Index (AQI) using real-time data, machine learning, and interactive visualizations. This project provides a 3-day AQI forecast, trend analysis, and pollutant insights, all personalized and interpretable with LIME.

🌟 Key Features

Daily AQI & Pollutants: View Karachi's current AQI and pollutant breakdown.
3-Day Forecast: LSTM model predicts the next 3 days' AQI, updated daily.
Trend Analysis: Explore seasonal, monthly, and weekly trends with interactive plots.
Pollutant Insights: Radar and pie charts show pollutant risk vs WHO standards.
WHO Comparison: Compare Karachi’s air quality against global safety limits.
LIME Explainability: Understand how each feature impacts the AQI predictions.
Personalized Model & Data:
Preprocessed data saved as karachi_preprocessed_MAK.csv.
LSTM model saved as lstm_aqi_model_MAK.keras.
Predictions saved as karachi_next3days_pred.csv.

🛠️ Tech Stack

Frontend: Streamlit, Plotly, custom HTML/CSS
Backend / ML: Python, TensorFlow (LSTM), scikit-learn, pandas, NumPy, LIME
Data Sources: Open-Meteo Air Quality & Weather APIs
Deployment / CI/CD: GitHub Actions (daily updates), Render.com
Visualization: Plotly, Matplotlib, Seaborn

📂 Project Structure

10-pearls-AQI-prediction-Karachi/
│
├── app.py                           ← Main Streamlit dashboard
├── README.md                        ← This documentation
├── requirements.txt                 ← Python dependencies
├── run_guide.txt                     ← Step-by-step local run instructions
├── changelog_MAK.txt                 ← Summary of file-level changes
│
├── src/                             ← Main source code
│   ├── create_lime.py
│   ├── fetch_data.py
│   ├── lstm_model_training.py
│   ├── predict.py
│   ├── preprocess_daily_data.py
│   └── update_daily_data.py
│
├── notebook/                        ← Jupyter notebooks for EDA and modeling
│
├── lstm_model/                      ← Saved LSTM model and scalers
├── processed_data/                  ← Preprocessed CSV data (`_MAK` version)
├── data/                            ← Raw AQI & weather data
├── predictions/                     ← Generated AQI predictions (`_MAK` version)
├── lime_explanations/               ← LIME explainability outputs
├── UI/                              ← Images used in dashboard
└── .github/workflows/aqi_pipeline.yml ← CI/CD GitHub Actions workflow

⚡ End-to-End Pipeline

Data Fetch (src/update_daily_data.py)
Fetches daily AQI & weather for Karachi via Open-Meteo API.
Data Preprocessing (src/preprocess_daily_data.py)
Cleans and fills missing values.
Caps outliers, creates new features including a 3-day rolling mean.
Saves processed output as processed_data/karachi_preprocessed_MAK.csv.
Model Training (src/lstm_model_training.py)
Trains an LSTM model (80 epochs, batch size 32).
Saves best model as lstm_model/lstm_aqi_model_MAK.keras.
Prediction (src/predict.py)
Loads model & scalers.
Predicts the next 3 days’ AQI.
Saves predictions as predictions/karachi_next3days_pred.csv.
Explainability (src/create_lime.py)
Generates LIME explanations for AQI predictions.
Outputs: HTML, Plotly JSON, Excel with feature contributions.
Dashboard (app.py)
Interactive multi-tab dashboard with predictions, trends, and LIME explanations.
CI/CD Pipeline (.github/workflows/aqi_pipeline.yml)
Automates daily updates, model retraining, and GitHub commits.



