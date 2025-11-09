## LIME generation modified by Muhammad Ali Khan
import os
import numpy as np
import pandas as pd
import joblib
from lime import lime_tabular
from tensorflow.keras.models import load_model
import plotly.express as px  # Import here for plotly figure

def generate_lime():
    LSTM_MODEL_DIR = "lstm_model"
    MODEL_PATH = os.path.join(LSTM_MODEL_DIR, "lstm_aqi_model.keras")
    SCALER_X_PATH = os.path.join(LSTM_MODEL_DIR, "scaler_X.pkl")
    DATA_PATH = "processed_data/daily_karachi_preprocessed.csv"
    SEQ_LEN = 7
    SAVE_DIR = "lime_explanations"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)

    df = pd.read_csv(DATA_PATH)
    features = [
        'AQI', 'PM10', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity',
        'Precipitation', 'month', 'log_PM2.5', 'log_CO', 'season_Spring',
        'season_Summer', 'season_Winter', 'weekday_1', 'weekday_2', 'weekday_3',
        'weekday_4', 'weekday_5', 'weekday_6', 'AQI_t+1', 'AQI_t+2', 'AQI_t+3',
        'AQI_lag_1', 'AQI_lag_2', 'AQI_roll_mean_3', 'AQI_roll_std_3',
        'AQI_diff'
    ] + [col for col in df.columns if col.startswith("season_") or col.startswith("weekday_")]

    X_all = scaler_X.transform(df[features])
    X_train = X_all[:-SEQ_LEN]
    X_test = X_all[-SEQ_LEN:]

    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=features,
        verbose=False,
        mode='regression'
    )
    sample_last = X_test[-1].reshape(1, -1)

    def predict_fn(x):
        seq = np.repeat(x[:, np.newaxis, :], SEQ_LEN, axis=1)
        return model.predict(seq).reshape(-1,)

    exp = explainer.explain_instance(sample_last[0], predict_fn, num_features=20)

    # Get intercept and local prediction
    intercept = exp.intercept[0]
    pred_local = exp.local_pred  # <-- correct here


    # Save explanation files
    csv_path = os.path.join(SAVE_DIR, "lime_explanation.csv")
    html_path = os.path.join(SAVE_DIR, "lime_explanation.html")
    exp.save_to_file(html_path)

    lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"])
    lime_df.to_csv(csv_path, index=False)

    # Create and save plotly figure PNG
    fig = px.bar(
        lime_df,
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Contribution",
        color_continuous_scale=px.colors.diverging.RdBu,
        title="LIME Explanation - Karachi AQI Forecast by Muhammad Ali Khan Feature Contributions"
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(font=dict(size=20, color="black"), x=0.5),
        xaxis=dict(title="Contribution to Prediction", tickfont=dict(color="black")),
        yaxis=dict(title="Feature", tickfont=dict(color="black")),
        font=dict(color="black")
    )
    png_path = os.path.join(SAVE_DIR, "lime_explanation.png")
    fig.write_image(png_path, scale=3)

    print(f"Saved Plotly PNG to {png_path}")

    return {
        "intercept": intercept,
        "pred_local": pred_local,
        "features_df": lime_df,
        "csv_path": csv_path,
        "html_path": html_path,
        "png_path": png_path
    }


if __name__ == "__main__":
    result = generate_lime()
    print(f"Intercept: {result['intercept']}")
    print(f"Local Prediction: {result['pred_local']}")
    print(result['features_df'])
    print(f"CSV saved to: {result['csv_path']}")
    print(f"HTML saved to: {result['html_path']}")
    print(f"PNG saved to: {result['png_path']}")
