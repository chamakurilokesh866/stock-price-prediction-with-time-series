import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Cache trained model and data
_model = None
_scaler = None
_df = None
_predictions = None
_training_len = None

def load_and_train():
    global _model, _scaler, _df, _predictions, _training_len

    print("Loading dataset...")
    df = pd.read_csv("MRF.NS.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Adj Close' in df.columns:
        df = df.drop(['Adj Close'], axis=1)
    df = df.sort_values('Date').reset_index(drop=True)
    _df = df

    data = df[['Close']].values
    training_len = int(len(data) * 0.80)
    _training_len = training_len

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    _scaler = scaler

    # Build training sequences (60-day windows)
    SEQ = 60
    X_train, y_train = [], []
    for i in range(SEQ, training_len):
        X_train.append(scaled[i - SEQ:i, 0])
        y_train.append(scaled[i, 0])
    X_train = np.array(X_train).reshape(-1, SEQ, 1)
    y_train = np.array(y_train)

    print("Building LSTM model...")
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=(SEQ, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Training model (3 epochs)...")
    model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1)
    _model = model

    # Predict on test set
    test_data = scaled[training_len - SEQ:]
    X_test = []
    for i in range(SEQ, len(test_data)):
        X_test.append(test_data[i - SEQ:i, 0])
    X_test = np.array(X_test).reshape(-1, SEQ, 1)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds).flatten()
    _predictions = preds
    print("Model ready!")

# ----- Routes -----

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/status')
def status():
    return jsonify({'ready': _model is not None})

@app.route('/api/train', methods=['POST'])
def train():
    load_and_train()
    return jsonify({'success': True, 'message': 'Model trained successfully'})

@app.route('/api/data')
def get_data():
    if _df is None or _predictions is None:
        return jsonify({'error': 'Model not trained yet. Call /api/train first.'}), 400

    df = _df
    tl = _training_len
    dates_all = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    closes = df['Close'].tolist()

    train_dates = dates_all[:tl]
    train_closes = closes[:tl]

    test_dates = dates_all[tl:]
    test_closes = closes[tl:]
    predictions = [round(float(p), 2) for p in _predictions[:len(test_dates)]]

    # OHLCV stats
    latest = df.iloc[-1]
    stats = {
        'last_date': latest['Date'].strftime('%Y-%m-%d'),
        'last_close': round(float(latest['Close']), 2),
        'last_open': round(float(latest['Open']), 2),
        'last_high': round(float(latest['High']), 2),
        'last_low': round(float(latest['Low']), 2),
        'last_volume': int(latest['Volume']),
        'total_records': len(df),
        'train_size': tl,
        'test_size': len(test_dates),
        'pred_count': len(predictions),
    }

    # last prediction vs actual
    if len(predictions) > 0:
        last_pred = predictions[-1]
        last_actual = test_closes[-1] if test_closes else None
        if last_actual:
            err = round(abs(last_pred - last_actual) / last_actual * 100, 2)
            stats['last_mape'] = err
        stats['last_prediction'] = last_pred

    return jsonify({
        'stats': stats,
        'train': {'dates': train_dates, 'close': train_closes},
        'test': {'dates': test_dates, 'close': test_closes},
        'predictions': predictions,
    })

@app.route('/api/history')
def history():
    """Return full OHLCV history for candlestick table."""
    if _df is None:
        return jsonify({'error': 'Not trained yet'}), 400
    df = _df.tail(30).copy()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'date': r['Date'].strftime('%Y-%m-%d'),
            'open': round(float(r['Open']), 2),
            'high': round(float(r['High']), 2),
            'low': round(float(r['Low']), 2),
            'close': round(float(r['Close']), 2),
            'volume': int(r['Volume']),
        })
    return jsonify({'rows': rows})

if __name__ == '__main__':
    print("Starting MRF Stock Prediction Server...")
    print("Visit: http://localhost:5000")
    app.run(debug=False, port=5000)
