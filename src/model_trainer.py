import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

def create_dataset(dataset, look_back=24):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_custom_model(route_id, data_file='traffic_data.csv', model_dir='models'):
    print(f"Training on-the-fly model for Route: {route_id}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    sub_df = df[df['route_id'] == route_id].sort_values('timestamp')
    if sub_df.empty:
        raise ValueError(f"No generation data found for {route_id}")
        
    times = sub_df['travel_time'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_times = scaler.fit_transform(times)
    
    scaler_path = os.path.join(model_dir, f'scaler_{route_id}.pkl')
    joblib.dump(scaler, scaler_path)
    
    train_size = int(len(scaled_times) * 0.8)
    train, test = scaled_times[0:train_size,:], scaled_times[train_size:len(scaled_times),:]
    
    look_back = 24 
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Fast compilation for dynamic UI demo
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Extremely fast training cycle for dashboard (2 epochs)
    model.fit(trainX, trainY, epochs=2, batch_size=64, verbose=0, validation_data=(testX, testY))
    
    model_path = os.path.join(model_dir, f'model_{route_id}.keras')
    model.save(model_path)
    print(f"Saved custom model to {model_path}")
    return True
