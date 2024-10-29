import yfinance as yf
import pandas as pd
import streamlit as st

# Download stock data for Google, Nvidia, Meta, and Amazon
tickers = ['GOOG', 'NVDA', 'META', 'AMZN']
stock_data = yf.download(tickers, start='2010-01-01', end='2023-09-15')

# Flatten the multi-level columns
stock_data.columns = [' '.join(col).strip() for col in stock_data.columns.values]
df = stock_data

# Define the process_query function
def process_query(query, df):
    try:
        stock_symbol, date_str, info_type = query.split(', ')
        date = pd.to_datetime(date_str)

        # Convert stock_symbol to uppercase to match with tickers in the DataFrame
        stock_symbol = stock_symbol.upper()

        # Filter the data for the specified date and stock symbol
        date_str = date.strftime('%Y-%m-%d')

        # Construct the column name
        column_name = f'{info_type} {stock_symbol}'

        if column_name in df.columns:
            if date_str in df.index:
                value = df.loc[date_str, column_name]
                if pd.notna(value):
                    response = f"The {info_type} price of {stock_symbol} on {date_str} was ${value:.2f}." if info_type != 'Volume' else f"The volume of {stock_symbol} on {date_str} was {value}."
                else:
                    response = f"No data available for {stock_symbol} on {date_str}."
            else:
                response = f"No data available for {stock_symbol} on {date_str}. It may be a non-trading day or not in the dataset."
        else:
            response = f"Invalid info type or stock symbol. Column '{column_name}' does not exist."
    except Exception as e:
        response = f"Error processing query: {e}"

    return response

# Streamlit Web Interface
def main():
    st.title("Stock Price Query System")
    st.write("You can query historical stock prices and volumes for Google, Nvidia, Meta, and Amazon.")
    
    # Input fields for the query
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., GOOG, NVDA, META, AMZN):", "")
    date_input = st.date_input("Select Date")
    info_type = st.selectbox("Select Information Type", ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    # Check if the user has entered valid inputs
    if st.button("Get Data"):
        if stock_symbol and date_input and info_type:
            query = f"{stock_symbol}, {date_input}, {info_type}"
            result = process_query(query, df)
            st.success(result)
        else:
            st.error("Please enter all the fields correctly.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
