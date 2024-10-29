import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Define the stock tickers
tickers = ['GOOG', 'NVDA', 'META', 'AMZN']

# Download stock data
@st.cache
def load_data():
    stock_data = yf.download(tickers, start='2010-01-01', end='2023-09-15')
    return pd.DataFrame(stock_data)

df = load_data()

# Streamlit app layout
st.title('Advanced Stock Analysis Web App')

# Sidebar for selecting features
st.sidebar.header('Choose Analysis Feature')
feature = st.sidebar.selectbox(
    'Select a Feature', 
    ('Check Missing Data', 'Check Missing Dates', 'Check Daily Return Anomalies',
     'Check Data Integrity', 'Check Extreme Growth', 'Check Duplicates', 
     'Calculate Correlation Matrix', 'Generate Report')
)

# Function to check for missing data
def check_missing_data(ticker, column):
    if (column, ticker) in df.columns:
        missing_values = df[(column, ticker)].isnull().sum()
        st.write(f"Missing values in {ticker} {column}: {missing_values}")
    else:
        st.write("Invalid ticker or column name.")

# Function to check for missing dates within a date range
def check_missing_dates(start_date, end_date):
    missing_dates = pd.date_range(start=start_date, end=end_date).difference(df.index)
    st.write(f"Missing Dates between {start_date} and {end_date}: {missing_dates}")

# Function to detect anomalies in daily returns
def check_daily_return_anomalies(date):
    daily_returns = df['Adj Close'].pct_change()
    if date in daily_returns.index:
        anomalies = daily_returns.loc[date].isnull()
        st.write(f"Daily Return Anomalies on {date}: {anomalies}")
    else:
        st.write("Date not found in daily returns.")

# Function to check data integrity for two columns
def check_data_integrity(ticker, col1, col2):
    if (col1, ticker) in df.columns and (col2, ticker) in df.columns:
        ticker_data = df.loc[:, [(col1, ticker), (col2, ticker)]]
        ticker_data_col1 = ticker_data[(col1, ticker)]
        ticker_data_col2 = ticker_data[(col2, ticker)]
        issues = ticker_data[ticker_data_col1 > ticker_data_col2]
        
        if not issues.empty:
            st.write(f"Issues where '{col1}' is greater than '{col2}' for {ticker}:")
            st.write(issues)
        else:
            st.write(f"No anomalies detected where '{col1}' is greater than '{col2}' for {ticker}.")
    else:
        st.write("Invalid ticker or columns.")

# Function to check extreme growth in cumulative returns
def check_extreme_growth(date):
    daily_returns = df['Adj Close'].pct_change()
    cumulative_returns = (1 + daily_returns).cumprod()
    if date in cumulative_returns.index:
        st.write(f"Cumulative Return on {date}: {cumulative_returns.loc[date]}")
    else:
        st.write("Date not found in cumulative returns.")

# Function to check for duplicate entries
def check_duplicates(column):
    if (column, 'GOOG') in df.columns:
        duplicates = df[df[column].duplicated()]
        st.write(f"Duplicate Entries in {column}:")
        st.write(duplicates)
    else:
        st.write("Column not found.")

# Function to calculate correlation matrix
def calculate_correlation():
    correlation = df['Adj Close'].corr()
    st.write("Correlation Matrix:")
    st.write(correlation)

# Function to generate a report
def generate_report():
    report = {}
    for ticker in tickers:
        report[ticker] = {
            'Missing Values': df['Close'][ticker].isnull().sum(),
            'Mean Price': df['Close'][ticker].mean(),
            'Standard Deviation': df['Close'][ticker].std()
        }
    st.write("Stock Report:")
    st.write(report)

# Display options based on selected feature
if feature == 'Check Missing Data':
    ticker = st.sidebar.selectbox('Select Ticker', tickers)
    column = st.sidebar.text_input('Enter Column (e.g., Open, Close)')
    if st.sidebar.button('Check'):
        check_missing_data(ticker, column)

elif feature == 'Check Missing Dates':
    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End Date')
    if st.sidebar.button('Check'):
        check_missing_dates(start_date, end_date)

elif feature == 'Check Daily Return Anomalies':
    date = st.sidebar.date_input('Date')
    if st.sidebar.button('Check'):
        check_daily_return_anomalies(date)

elif feature == 'Check Data Integrity':
    ticker = st.sidebar.selectbox('Select Ticker', tickers)
    col1 = st.sidebar.text_input('First Column (e.g., Open)')
    col2 = st.sidebar.text_input('Second Column (e.g., Close)')
    if st.sidebar.button('Check'):
        check_data_integrity(ticker, col1, col2)

elif feature == 'Check Extreme Growth':
    date = st.sidebar.date_input('Date')
    if st.sidebar.button('Check'):
        check_extreme_growth(date)

elif feature == 'Check Duplicates':
    column = st.sidebar.text_input('Enter Column (e.g., High)')
    if st.sidebar.button('Check'):
        check_duplicates(column)

elif feature == 'Calculate Correlation Matrix':
    if st.sidebar.button('Calculate'):
        calculate_correlation()

elif feature == 'Generate Report':
    if st.sidebar.button('Generate'):
        generate_report()
