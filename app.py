# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import TimeSeriesModel
from utils import load_data, preprocess_data

app = Flask(__name__)

# Load the model
model = TimeSeriesModel(model_path='../model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    periods = data.get('periods', 30)
    symbol = data.get('symbol', 'AAPL')
    

    stock_data = load_data(symbol)
    stock_data = preprocess_data(stock_data)
    
    # Forecast
    forecast = model.forecast(stock_data, periods)
    
    return jsonify({'symbol': symbol, 'forecast': forecast.tolist()})

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.get_json(force=True)
    symbol = data.get('symbol', 'AAPL')
    
    # Load and preprocess data
    stock_data = load_data(symbol)
    stock_data = preprocess_data(stock_data)
    
    # Retrain the model
    model.train(stock_data)
    model.save_model('../model.pkl')
    
    return jsonify({'status': 'Model retrained successfully'})

@app.route('/visualize', methods=['GET'])
def visualize():
    symbol = request.args.get('symbol', 'AAPL')
    
    stock_data = load_data(symbol)
    
    # make plot
    plot_url = model.plot_stock_data(stock_data, symbol)
    
    return jsonify({'symbol': symbol, 'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)

