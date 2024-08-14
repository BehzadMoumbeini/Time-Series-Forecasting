# utils.py

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def load_data(file_path):


    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

def preprocess_data(data):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
    
    return scaled_df, scaler

def scale_new_data(data, scaler):

    scaled_data = scaler.transform(data)
    scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
    
    return scaled_df

def plot_forecast(original_data, forecast_data, forecast_period):

    plt.figure(figsize=(12, 6))
    plt.plot(original_data.index, original_data, label='Original Data')
    plt.plot(forecast_data.index, forecast_data, label='Forecasted Data', color='orange')
    plt.axvline(original_data.index[-forecast_period], color='red', linestyle='--')
    plt.title("Time Series Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    
    # Save the plot to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url

def save_model(model, file_path):

    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    return joblib.load(file_path)

