import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import logging
from datetime import timedelta
import os

# Set up logging
logging.basicConfig(filename='stock_predictions.log', level=logging.INFO)

# Load stock data for multiple tickers
def load_stock_data(tickers):
    all_data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        all_data[ticker] = df
    return all_data

# Preprocess data: Handle missing values
def preprocess_data(df):
    df = df.dropna()
    return df

# Prepare data for LSTM
def prepare_data(df, selected_features, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[selected_features])
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# Define and compile the LSTM model for multiple outputs
def create_lstm_model(input_shape, output_shape):
    model = Sequential([  
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Custom accuracy metric based on closeness of prediction
def custom_accuracy(y_true, y_pred, tolerance=0.05):
    correct = np.sum(np.abs(y_true - y_pred) / y_true < tolerance)
    return correct / len(y_true)

# Predict future stock prices (n steps ahead)
def lstm_predict(tickers, steps=3, selected_features=None):
    predictions_all_tickers = {}
    stock_data = load_stock_data(tickers)
    
    for ticker, df in stock_data.items():
        df = preprocess_data(df)
        X, y, scaler = prepare_data(df, selected_features)
        
        if ticker == tickers[0]:
            model = create_lstm_model((X.shape[1], X.shape[2]), len(selected_features))
            
            # Run model training silently in the background
            with st.spinner(f'Training model for {ticker}...'):
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)  # Verbose set to 0 for silent training
            
            # Make predictions
            y_pred = model.predict(X)
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_true_rescaled = scaler.inverse_transform(y)
            
            # Calculate MSE and accuracy, but log them instead of showing on the frontend
            mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
            accuracy = custom_accuracy(y_true_rescaled, y_pred_rescaled)
            logging.info(f"Model trained for {ticker} - Final Loss (MSE): {mse:.4f}, Accuracy: {accuracy:.2%}")
        
        # Predict the next `steps` ahead
        last_sequence = X[-1]
        predictions = []
        dates = pd.to_datetime(df.index[-1])

        for _ in range(steps):
            prediction = model.predict(last_sequence.reshape(1, X.shape[1], X.shape[2]), verbose=0)
            predictions.append(prediction[0])
            last_sequence = np.vstack((last_sequence[1:], prediction))
            dates += timedelta(days=1)

        # Inverse transform predictions to original scale
        predictions = scaler.inverse_transform(np.array(predictions))
        logging.info(f"Predictions for {ticker} (next {steps} steps): {predictions}")
        
        # Store predictions in DataFrame
        future_dates = pd.date_range(df.index[-1], periods=steps + 1, freq='D')[1:]
        predictions_df = pd.DataFrame({'Date': future_dates})
        
        for i, feature in enumerate(selected_features):
            predictions_df[f'Predicted {feature}'] = predictions[:, i]
        
        predictions_all_tickers[ticker] = predictions_df

        # Automatically download the predicted data
        file_path = os.path.join(os.path.expanduser('~'), 'Documents', f'{ticker}_predictions.csv')
        predictions_df.to_csv(file_path, index=False)
        st.success(f'Predictions for {ticker} saved at {file_path}')
    
    return predictions_all_tickers

# Streamlit App Layout with Sidebar
st.title("Stock Price Prediction App")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
tickers = st.sidebar.text_input("Enter the tickers (comma separated):").split(',')
steps = st.sidebar.number_input("Enter the number of days to predict:", min_value=1, max_value=365, value=3)
selected_features = st.sidebar.text_input("Enter the features you want to predict (comma separated):").split(',')

if st.sidebar.button("Predict"):
    st.write(f"Predicting the next {steps} steps for selected features: {selected_features} and tickers: {tickers}")
    predicted_values = lstm_predict(tickers, steps=steps, selected_features=selected_features)
    
    # Display predicted values
    for ticker, predictions_df in predicted_values.items():
        st.write(f"\nPredictions for {ticker}:")
        st.dataframe(predictions_df)

        # Provide file download option
        st.download_button(
            label="Download Predictions",
            data=predictions_df.to_csv(index=False).encode('utf-8'),
            file_name=f'{ticker}_predictions.csv',
            mime='text/csv'
        )

