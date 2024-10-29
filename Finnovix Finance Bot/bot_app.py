# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import logging
from datetime import timedelta
import os
import smtplib  # For email notifications
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sklearn.linear_model import LinearRegression

# Set up the Streamlit dashboard
st.set_page_config(page_title="Finnovix Stock Analysis and Prediction App", layout="wide")

# Email details
sender_email = "your_email@gmail.com"
receiver_email = "receiver_email@gmail.com"
password = "your_app_password"  # Your App Password
smtp_server = "smtp.gmail.com"
smtp_port = 587

# Set price thresholds for notifications
price_thresholds = {
    "GOOG": {"Open": 28, "Close": 30, "High": 32, "Low": 25},
    "NVDA": {"Open": 500, "Close": 505, "High": 510, "Low": 495},
    "META": {"Open": 350, "Close": 360, "High": 365, "Low": 340},
    "AMZN": {"Open": 3200, "Close": 3220, "High": 3250, "Low": 3180}
}

# Educational insights dictionary
educational_insights = {
    "GOOG": {
        "Open": "The opening price reflects the initial market sentiment for the day, influenced by overnight news and events.",
        "Close": "The closing price is crucial as it reflects the final consensus of the market on the stock's value for the day.",
        "High": "A new high can indicate bullish momentum and increased investor confidence.",
        "Low": "A new low might suggest bearish sentiment or selling pressure."
    },
    "NVDA": {
        "Open": "A higher opening price can indicate strong demand from buyers at the start of trading.",
        "Close": "Monitoring the closing price helps identify trends over time and assess market sentiment.",
        "High": "Breaking a new high often attracts more buyers and may lead to further price increases.",
        "Low": "A significant drop in price can trigger stop-loss orders, further increasing selling pressure."
    },
    "META": {
        "Open": "The opening price can be influenced by global market trends and news related to social media regulations.",
        "Close": "The closing price is a key indicator of the day's trading activity and investor behavior.",
        "High": "Reaching a new high can signify strong market support and investor enthusiasm.",
        "Low": "A drop to a new low can indicate heightened concerns over the company's growth prospects."
    },
    "AMZN": {
        "Open": "Amazon's opening price reflects investor expectations based on recent earnings reports or market news.",
        "Close": "The closing price is vital for analyzing price trends and making future investment decisions.",
        "High": "Hitting a new high can indicate robust growth potential and confidence in Amazon's business model.",
        "Low": "A new low might signal challenges in the retail sector or increased competition."
    }
}

# Function to send email
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.success("Notification email sent successfully!")
    except Exception as e:
        st.error(f"An error occurred while sending email: {e}")

# Function to display logo
def display_logo():
    st.image("Finnovix.jpeg", use_column_width=False, width=300)  # Adjust the path as needed

# Load stock data for multiple tickers and check thresholds
def load_and_check_stock_data(tickers):
    stock_data = yf.download(tickers, start="2023-01-01", end="2023-12-31", group_by ='ticker')
    for ticker in tickers:
        if ticker in stock_data.columns:
            latest_data = stock_data[ticker].iloc[-1]
            st.subheader(f"{ticker} latest data:")
            st.write(latest_data)
            for metric in price_thresholds[ticker]:
                if metric in latest_data.index:
                    latest_price = latest_data[metric]
                    threshold = price_thresholds[ticker][metric]
                    if latest_price > threshold:
                        st.warning(f"{ticker} {metric} exceeds the threshold. Sending email...")
                        insight = educational_insights[ticker][metric]
                        email_body = (f"{ticker} {metric} is {latest_price}, exceeding the threshold of {threshold}.\n\n"
                                       f"Educational Insight: {insight}")
                        send_email(f"{ticker} Alert: {metric} Exceeded Threshold", email_body)
                    else:
                        st.info(f"{ticker} {metric} is below the threshold. No email sent.")
                else:
                    st.error(f"Metric '{metric}' not found in latest data for {ticker}.")
        else:
            st.error(f"No data found for {ticker}.")
    return stock_data

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
            with st.spinner(f'Training model for {ticker}...'):
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            y_pred = model.predict(X)
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_true_rescaled = scaler.inverse_transform(y)
            mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
            accuracy = custom_accuracy(y_true_rescaled, y_pred_rescaled)
            logging.info(f"Model trained for {ticker} - Final Loss (MSE): {mse:.4f}, Accuracy: {accuracy:.2%}")
        last_sequence = X[-1]
        predictions = []
        dates = pd.to_datetime(df.index[-1])
        for _ in range(steps):
            prediction = model.predict(last_sequence.reshape(1, X.shape[1], X.shape[2]), verbose=0)
            predictions.append(prediction[0])
            last_sequence = np.vstack((last_sequence [1:], prediction))
            dates += timedelta(days=1)
        predictions = scaler.inverse_transform(np.array(predictions))
        future_dates = pd.date_range(df.index[-1], periods=steps + 1, freq='D')[1:]
        predictions_df = pd.DataFrame({'Date': future_dates})
        for i, feature in enumerate(selected_features):
            predictions_df[f'Predicted {feature}'] = predictions[:, i]
        predictions_all_tickers[ticker] = predictions_df
        file_path = os.path.join(os.path.expanduser('~'), 'Documents', f'{ticker}_predictions.csv')
        predictions_df.to_csv(file_path, index=False)
        st.success(f'Predictions for {ticker} saved at {file_path}')
    return predictions_all_tickers

# Generate visualizations
def generate_visualizations(df, ticker, graph_type):
    if graph_type == 'Bar Graph':
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Date'], y=df['Open'], name='Open', marker_color='blue'))
        fig.add_trace(go.Bar(x=df['Date'], y=df['Close'], name='Close', marker_color='red'))
        fig.update_layout(title=f'Opening vs Closing Prices for {ticker}', barmode='group', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    elif graph_type == 'Histogram':
        fig = px.histogram(df, x='Date', y='Open', title=f'Histogram of Opening Prices for {ticker}')
        st.plotly_chart(fig)
    elif graph_type == '3D Pie Chart':
        open_close_ratio = [df['Open'].mean(), df['Close'].mean()]
        labels = ['Average Open', 'Average Close']
        fig = go.Figure(data=[go.Pie(labels=labels, values=open_close_ratio, hole=0.3)])
        fig.update_traces(marker=dict(line=dict(color='#000000', width=2)),
                          pull=[0.1, 0])
        fig.update_layout(title="Average Open vs Close Prices", showlegend=True)
        st.plotly_chart(fig)
    elif graph_type == '3D Histogram':
        fig = go.Figure(data=[go.Mesh3d(x=df['Open'], y=df['Close'], z=df['Volume'], opacity=0.6)])
        fig.update_layout(title=f'3D Histogram of Open vs Close Prices for {ticker}', scene=dict(xaxis_title='Open', yaxis_title='Close', zaxis_title='Volume'))
        st.plotly_chart(fig)
    elif graph_type == 'Pairplot':
        sns.pairplot(df[['Open', 'Close', 'High', 'Low', 'Volume']])
        plt.suptitle(f'Pairplot for {ticker}', y=1.02)
        st.pyplot()
    elif graph_type == 'Heatmap':
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    elif graph_type == 'Scatter Plot':
        fig = px.scatter(df, x='Date', y='Open', title=f'Scatter Plot of Opening Prices for {ticker}')
        st.plotly_chart(fig)
    elif graph_type == '3D Scatter Plot':
        fig = px.scatter_3d(df, x='Date', y='Open', z='Close', title=f'3D Scatter Plot of Opening and Closing Prices for {ticker}')
        st.plotly_chart(fig)

# Combined Introduction Section
def show_introduction():
    display_logo()
    st.title("Finnovix App")
    st.header("Welcome to the Finnovix Stock Analysis and Prediction App")
    st.write("""
    This app provides insights into stock market trends, predictions, and auditing functionalities.
    Use the sidebar to navigate through various sections of the app.
    """)
    st.subheader("What is Auditing?")
    st.write("""
    Auditing refers to the systematic examination and evaluation of financial records to ensure accuracy, transparency, and adherence to regulations. It helps identify discrepancies, errors, and fraud, ensuring the integrity of financial data.
    """)
    st.subheader("Why is Auditing Important in Finance?")
    st.write("""
    In the finance sector, auditing plays a critical role by maintaining trust and ensuring compliance with legal standards. It helps detect anomalies or irregularities in financial transactions, ensures proper risk management, and validates the accuracy of financial reports.
    """)
    st.subheader("Effectiveness of Auditing in Finance")
    st.write("""
    Auditing brings significant value by:
    - **Enhancing Accuracy**: Identifies any deviations in financial data, ensuring that all transactions are properly recorded and verified.
    - **Fraud Detection**: Plays a key role in uncovering fraudulent activities by closely examining financial activities.
    - **Compliance & Governance**: Ensures adherence to financial laws and regulations, improving overall governance in organizations.
    """)
    st.subheader("What are Financial Graphs?")
    st.write("""
    Financial graphs visually represent data, enabling users to understand trends, patterns, and relationships within financial datasets. Graphs like line charts, bar graphs, and histograms help analysts and investors easily interpret complex financial data.
    """)
    st.subheader("Why Use Graphs in Finance?")
    st.write("""
    Graphs make it easier to visualize stock price movements, trading volumes, and correlations between different financial metrics. They are essential tools in financial analysis, allowing users to spot trends, detect anomalies, and make informed decisions based on visual insights.
    """)
    st.subheader("Effectiveness of Graphs in Finance")
    st.write("""
    Graphs are highly effective because they:
    - **Simplify Complex Data**: Turn large datasets into easily interpretable visuals.
    - **Identify Trends**: Make it easy to see trends and patterns over time, helping with forecasting.
    - **Enhance Decision-Making**: Provide a clear view of financial performance, aiding investors and analysts in making sound financial decisions.
    """)
    st.subheader("What is Stock Prediction?")
    st.write("""
    Stock prediction involves using statistical models and algorithms to forecast future stock prices based on historical data. Machine learning models like Long Short-Term Memory (LSTM) are commonly used to predict stock market trends.
    """)
    st.subheader("Why Use Stock Prediction in Finance?")
    st.write("""
    Accurate stock predictions provide insights into potential future movements of a stock, enabling investors to make better decisions about buying, selling, or holding their assets. Predictive models help reduce risk and increase the likelihood of profitable investments.
    """)
    st.subheader("Effectiveness of Stock Prediction in Finance")
    st.write("""
    Stock prediction models are effective because they:
    - **Forecast Market Trends**: Help investors anticipate changes in the market.
    - **Reduce Risk**: By predicting stock movements, investors can minimize the chances of losses.
    - **Improve Strategy**: Enable better financial planning and strategy development based on likely future outcomes.
    """)
    st.subheader("What is a Financial Query System?")
    st.write("""
    A financial query system uses natural language processing (NLP) to allow users to ask questions about financial data and receive relevant information in response. It automates the process of searching for financial data, improving accessibility and user experience.
    """)
    st.subheader("Why Use a Query System in Finance?")
    st.write("""
    Financial queries help users quickly access specific information without manually navigating through large datasets. It simplifies data retrieval by providing answers in response to user queries, making financial analysis more user-friendly and accessible.
    """)
    st.subheader("Effectiveness of Financial Queries in Finance")
    st.write("""
    Query systems are effective because they:
    - **Enhance User Experience**: Allow users to retrieve data in a conversational manner without needing advanced technical skills.
    - **Save Time**: Provide instant answers to complex questions, saving users time when navigating financial data.
    - **Increase Accessibility**: Make financial data accessible to both technical and non-technical users.
    """)

# Streamlit App
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Auditing", "Graphs", "Prediction", "Notifications", "Query", "Dashboard"])

if section == "Introduction":
    show_introduction()
elif section == "Auditing":
    display_logo()
    st.header("Auditing")
    st.write("Auditing Options:")
    tickers = st.text_input("Enter the tickers (comma separated):", key="auditing_tickers").split(',')
    feature = st.selectbox("Select a feature", ['Check Missing Data', 'Check Missing Dates', 'Check Daily Return Anomalies', 'Check Data Integrity', 'Check Extreme Growth', 'Check Duplicates', 'Calculate Correlation Matrix', 'Generate Report', 'Check Stationarity', 'Check Autocorrelation'])
    if feature == 'Check Missing Data':
        ticker = st.selectbox('Select Ticker', tickers)
        column = st.text_input('Enter Column (e.g., Open, Close)')
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data[column]
        if st.button('Check'):
            missing_data = df.isnull().sum()
            st.write(f"Missing data in {column} column: {missing_data}")
    elif feature == 'Check Missing Dates':
        start_date = st.date_input('Start Date')
        end_date = st.date_input('End Date')
        stock_data = yf.download(tickers, start=start_date, end=end_date)
        df = stock_data.index
        if st.button('Check'):
            missing_dates = pd.date_range(start_date, end_date).difference(df)
            st.write(f"Missing dates: {missing_dates}")
    elif feature == 'Check Daily Return Anomalies':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data['Close']
        if st.button('Check'):
            daily_returns = df.pct_change()
            anomalies = daily_returns[daily_returns > 2 * daily_returns.std()]
            st.write(f"Daily return anomalies: {anomalies}")
    elif feature == 'Check Data Integrity':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start=' 2010-01-01', end='2023-09-15')
        df = stock_data
        if st.button('Check'):
            st.write(f"Data integrity check for {ticker}:")
            st.write(df.describe())
    elif feature == 'Check Extreme Growth':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data['Close']
        if st.button('Check'):
            growth_rates = df.pct_change()
            extreme_growth = growth_rates[growth_rates > 2 * growth_rates.std()]
            st.write(f"Extreme growth rates for {ticker}: {extreme_growth}")
    elif feature == 'Check Duplicates':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data
        if st.button('Check'):
            duplicates = df.duplicated().sum()
            st.write(f"Duplicate rows in {ticker} data: {duplicates}")
    elif feature == 'Calculate Correlation Matrix':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data
        if st.button('Calculate'):
            corr_matrix = df.corr()
            st.write(f"Correlation matrix for {ticker}:")
            st.write(corr_matrix)
    elif feature == 'Generate Report':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data
        if st.button('Generate'):
            st.write(f"Report for {ticker}:")
            st.write(df.describe())
    elif feature == 'Check Stationarity':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data['Close']
        if st.button('Check'):
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(df)
            st.write(f"Stationarity test for {ticker}:")
            st.write(result)
    elif feature == 'Check Autocorrelation':
        ticker = st.selectbox('Select Ticker', tickers)
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-09-15')
        df = stock_data['Close']
        if st.button('Check'):
            from statsmodels.graphics.tsaplots import plot_acf
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_acf(df, ax=ax)
            st.pyplot(fig)
elif section == "Graphs":
    display_logo()
    st.header("Graph")
    st.write("Graph Options:")
    tickers = st.text_input("Enter the tickers (comma separated):", key="graph_tickers").split(',')
    graph_type = st.selectbox("Select a graph type", ['Bar Graph', 'Histogram', '3D Pie Chart', '3D Histogram', 'Pairplot', 'Heatmap', 'Scatter Plot', '3D Scatter Plot'])
    if st.button("Generate Visualization"):
        if len(tickers) == 0 or any(ticker.strip() == "" for ticker in tickers):
            st.error("Please enter valid ticker symbols.")
        else:
            try:
                stock_data = yf.download(tickers, start='2010-01-01', end='2023-09-15')
                if stock_data.empty:
                    st.error(f"No data available for the tickers: {', '.join(tickers)}")
                else:
                    df = stock_data.reset_index()
                    for ticker in tickers:
                        st.subheader(f"{graph_type} for {ticker}")
                        generate_visualizations(df, ticker, graph_type)
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif section == "Prediction":
    display_logo()
    st.header("Prediction")
    tickers = st.text_input("Enter the tickers (comma separated):", key="prediction_tickers").split(',')
    selected_features = st.multiselect("Select features to predict:", ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    steps = st.number_input("Enter the number of steps ahead to predict:", min_value=1, value=3)
    predictions_all_tickers = {}
    if st.button("Predict"):
        if len(tickers) > 0 and len(selected_features) > 0:
             predictions_all_tickers = lstm_predict(tickers, steps, selected_features)
    for ticker, predictions_df in predictions_all_tickers.items():
                st.subheader(f"Predictions for {ticker}:")
                st.write(predictions_df)
    else:
             st.error("Please enter valid tickers and select features to predict.")

elif section == "Notifications":
    display_logo()
    st.header("Notifications for Stock Price Alerts")
    tickers = st.text_input("Enter the tickers (comma separated):", key="notification_tickers").split(',')
    if st.button("Check Stock Data"):
        load_and_check_stock_data(tickers)

elif section == "Query":
    display_logo()
    st.header("Query")
    st.write("Query Options:")
    query = st.text_input("Enter your query (e.g., GOOG, 2022-01-01, Open):", key="query_input")
    if st.button("Get Data"):
        try:
            stock_symbol, date_str, info_type = query.split(', ')
            date = pd.to_datetime(date_str)
            stock_symbol = stock_symbol.upper()
            stock_data = yf.download(stock_symbol, start='2010-01-01', end='2023-09-15')
            if date in stock_data.index:
                df = stock_data[info_type]
                response = f"The {info_type} price of {stock_symbol} on {date_str} was ${df.loc[date]:.2f}."
                st.write(response)
            else:
                st.error(f"No data available for {stock_symbol} on {date_str}.")
        except ValueError:
            st.error("Please enter the query in the correct format: 'SYMBOL, YYYY-MM-DD, FEATURE'.")
        except KeyError:
            st.error(f"The feature '{info_type}' is not available for {stock_symbol}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif section == "Dashboard":
    display_logo()
    st.header("Stock Analysis Dashboard")
    st.write("Dashboard Options:")
    
    # Define the stock tickers
    tickers = ['GOOG', 'NVDA', 'META', 'AMZN']
    
    # Download stock data for the specified tickers
    try:
        stock_data = yf.download(tickers, start='2010-01-01', end='2023-09-15')
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

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
    volume_data = ticker_data['Volume']
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
    ma_periods = st.multiselect("Select Moving Averages", options=[5, 10, 20, 50, 100, 200], default=[20])

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
    forecast_data = ticker_data[[data_type]].copy ()
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