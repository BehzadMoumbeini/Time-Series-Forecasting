import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib


class TimeSeriesModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None
    
    def train(self, data):
        self.model = ExponentialSmoothing(
            data, trend='add', seasonal='add', seasonal_periods=12
        ).fit()
    
    def forecast(self, data, periods):
        return self.model.forecast(periods)
    
    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
    
    def plot_stock_data(self, data, symbol):
        plt.figure(figsize=(10, 6))
        plt.plot(data, label=f'{symbol} Stock Prices')
        plt.title(f'{symbol} Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        plot_url = f'static/{symbol}_stock_plot.png'
        plt.savefig(plot_url)
        plt.close()
        
        return plot_url

