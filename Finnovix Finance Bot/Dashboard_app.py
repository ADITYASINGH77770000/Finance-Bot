import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression

# Set up the Streamlit dashboard
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Define the stock tickers
tickers = ['GOOG', 'NVDA', 'META', 'AMZN']

# Download stock data for Google, Nvidia, Facebook, and Amazon
try:
    stock_data = yf.download(tickers, start='2010-01-01', end='2023-09-15')
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Dashboard title
st.title("🔍 Advanced Stock Analysis Dashboard")

# Sidebar for user input
st.sidebar.header("User  Input")
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)
data_type = st.sidebar.selectbox("Select Data Type", ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
days_to_predict = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)
show_correlation = st.sidebar.checkbox("Show Correlation Heatmap")
show_sentiment = st.sidebar.checkbox("Show News Sentiment")

# Filter data for the selected ticker
ticker_data = stock_data.xs(selected_ticker, level=1, axis=1)

# Display additional metrics
st.header("Key Metrics")
metrics_col1, metrics_col2 = st.columns(2)

# Metrics for selected ticker
with metrics_col1:
    st.metric(label="Average " + data_type, value=f"${ticker_data[data_type].mean():.2f}")
    st.metric(label="Total Volume", value=f"{ticker_data['Volume'].sum():,}")

with metrics_col2:
    st.metric(label="Max " + data_type, value=f"${ticker_data[data_type].max():.2f}")
    st.metric(label="Min " + data_type, value=f"${ticker_data[data_type].min():.2f}")

# Create layout
st.header(f"{selected_ticker} Price Movement (Candlestick Chart)")
candlestick_chart = go.Figure(data=[go.Candlestick(
    x=ticker_data.index,
    open=ticker_data['Open'],
    high=ticker_data['High'],
    low=ticker_data['Low'],
    close=ticker_data['Close'],
    name='Candlestick',
    showlegend=False
)])
candlestick_chart.update_layout(title=f"{selected_ticker} Price Movement", xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(candlestick_chart, use_container_width=True)

# Trading Volume Bar Chart
st.header(f"{selected_ticker} Trading Volume")
volume_data = ticker_data['Volume']  # Use only the volume data for the selected ticker
bar_chart = px.bar(x=volume_data.index, y=volume_data.values, title=f"{selected_ticker} Trading Volume", labels={'x': 'Date', 'y': 'Volume'})
st.plotly_chart(bar_chart, use_container_width=True)

# Show correlation heatmap if selected
if show_correlation:
    st.header("Correlation Heatmap")
    correlation = stock_data['Close'].corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
    ))
    st.plotly_chart(fig)

# Moving Average Trend
st.header(f"{selected_ticker} Moving Averages")
ma_periods = st.sidebar.multiselect("Select Moving Averages", options=[5, 10, 20, 50, 100, 200], default=[20])

# Create a DataFrame for moving averages
ma_data = ticker_data[['Close']].copy()
for period in ma_periods:
    ma_data[f'MA_{period}'] = ma_data['Close'].rolling(window=period).mean()

# Drop rows with NaN values
ma_data = ma_data.dropna()

# Plot moving averages
ma_chart = px.line(ma_data, x=ma_data.index, y=['Close'] + [f'MA_{period}' for period in ma_periods if f'MA_{period}' in ma_data.columns], title=f"{selected_ticker} Moving Averages")
st.plotly_chart(ma_chart)

# Forecasting Feature
st.header(f"{selected_ticker} Price Forecasting")
# Prepare data for forecasting
forecast_data = ticker_data[[data_type]].copy()
forecast_data.reset_index(inplace=True)
forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
forecast_data['Date'] = forecast_data['Date'].apply(lambda date: date.timestamp())

# Split data into training and testing sets
train_size = int(len(forecast_data) * 0.8)
train_data, test_data = forecast_data[:train_size], forecast_data[train_size:]

# Create and train a linear regression model
model = LinearRegression()
model.fit(train_data[['Date']], train_data[data_type])

# Generate forecast for the next 'days_to_predict' days
future_dates = np.array([forecast_data['Date'].iloc[-1] + i for i in range(1, days_to_predict + 1)]).reshape(-1, 1)
forecast = model.predict(future_dates)

# Plot forecast
forecast_chart = px.line(x=np.concatenate((forecast_data['Date'], future_dates.flatten())), y=np.concatenate((forecast_data[data_type], forecast)), title=f"{selected_ticker} {data_type} Forecast")
st.plotly_chart(forecast_chart)

# 3 D Scatter Plot
st.header(f"{selected_ticker} 3D Scatter Plot")
scatter_data = ticker_data[['Close', 'Volume']]
scatter_chart = px.scatter_3d(scatter_data, x=scatter_data.index, y='Close', z='Volume', title=f"{selected_ticker} 3D Scatter Plot")
st.plotly_chart(scatter_chart)

# Show news sentiment if selected
if show_sentiment:
    st.header("News Sentiment Analysis")
    sentiment_data = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'], 'Sentiment': [0.5, 0.7, 0.3, 0.9, 0.1]})
    sentiment_chart = px.bar(sentiment_data, x='Date', y='Sentiment', title="News Sentiment Analysis")
    st.plotly_chart(sentiment_chart)