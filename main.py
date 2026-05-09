import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def run_analysis():
    print("Loading dataset...")
    df = pd.read_csv("MRF.NS.csv")
    print(f"Dataset shape: {df.shape}")

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(['Adj Close'], axis=1)
    
    print("Checking for missing values:")
    print(df.isnull().sum())

    # Visualizing
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'])
    plt.title('MRF Stock Close Price History')
    plt.xlabel('Date')
    plt.ylabel("Close Price")
    plt.savefig('stock_price_history.png')
    print("Saved stock_price_history.png")

    # Preparing data for LSTM
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    print("Building LSTM model...")
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(keras.layers.LSTM(units=64, return_sequences=False))
    model.add(keras.layers.Dense(units=25))
    model.add(keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Training model (this might take a while)...")
    model.fit(x_train, y_train, batch_size=1, epochs=1) # Reduced epochs for demonstration

    # Test data
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    print("Making predictions...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot results
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(10, 5))
    plt.title('Model Prediction Results')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train['Date'], train['Close'])
    plt.plot(valid['Date'], valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('prediction_results.png')
    print("Saved prediction_results.png")

if __name__ == "__main__":
    run_analysis()
