import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import difflib

# Download stock data for Google, Nvidia, Meta, and Amazon
tickers = ['GOOG', 'NVDA', 'META', 'AMZN']
stock_data = yf.download(tickers, start='2010-01-01', end='2023-09-15')

# Convert the stock data into a usable DataFrame format for individual companies
df = pd.DataFrame(stock_data)

# Function to generate the visualizations
def generate_visualizations(df, ticker, graph_type):
    # Filter the stock data for the specific company and drop NaN values
    stock_df = df.xs(ticker, level=1, axis=1).dropna()

    if graph_type == 'Bar Graph':
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stock_df.index, y=stock_df['Open'], name='Open', marker_color='blue'))
        fig.add_trace(go.Bar(x=stock_df.index, y=stock_df['Close'], name='Close', marker_color='red'))
        fig.update_layout(title=f'Opening vs Closing Prices for {ticker}', barmode='group', xaxis_title='Date', yaxis_title='Price')

    elif graph_type == 'Histogram':
        fig = px.histogram(stock_df, x='Volume', nbins=20, title=f'Volume Distribution for {ticker}', color_discrete_sequence=['purple'])
        fig.update_layout(xaxis_title='Volume', yaxis_title='Frequency')

    elif graph_type == '3D Pie Chart':
        open_close_ratio = [stock_df['Open'].mean(), stock_df['Close'].mean()]
        labels = ['Average Open', 'Average Close']
        fig = go.Figure(data=[go.Pie(labels=labels, values=open_close_ratio, hole=0.3)])
        fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))

    elif graph_type == '3D Histogram':
        fig = go.Figure(data=[go.Mesh3d(
            x=stock_df['Open'],
            y=stock_df['Close'],
            z=stock_df['Volume'],
            colorbar_title='Volume',
            colorscale='Viridis'
        )])
        fig.update_layout(title=f'3D Histogram of Open vs Close Prices for {ticker}', scene=dict(xaxis_title='Open', yaxis_title='Close', zaxis_title='Volume'))

    elif graph_type == 'Pairplot':
        sns.pairplot(stock_df[['Open', 'Close', 'High', 'Low', 'Volume']])
        plt.suptitle(f'Pairplot for {ticker}', y=1.02)
        st.pyplot()

    elif graph_type == 'Heatmap':
        corr_matrix = stock_df[['Open', 'Close', 'Volume', 'High', 'Low', 'Adj Close']].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title=f'Correlation Heatmap for {ticker}')
    
    elif graph_type == 'Scatter Plot':
        fig = px.scatter(stock_df, x='Open', y='Close', color='Volume', size='Volume', title=f'Open vs Close Prices for {ticker}', color_continuous_scale='Viridis')

    elif graph_type == '3D Scatter Plot':
        fig = go.Figure(data=[go.Scatter3d(
            x=stock_df['Open'],
            y=stock_df['Close'],
            z=stock_df['Volume'],
            mode='markers',
            marker=dict(size=5, color=stock_df['Volume'], colorscale='Viridis', colorbar=dict(title='Volume'))
        )])
        fig.update_layout(title=f'3D Scatter Plot: Open vs Close Prices vs Volume for {ticker}', scene=dict(xaxis_title='Open', yaxis_title='Close', zaxis_title='Volume'))

    # Show the figure
    st.plotly_chart(fig)

# Function to find the closest matching ticker
def get_closest_ticker(input_ticker, valid_tickers):
    closest_matches = difflib.get_close_matches(input_ticker.upper(), valid_tickers)
    return closest_matches[0] if closest_matches else None

# Streamlit app UI
st.title('Stock Data Visualization App')

# Sidebar for user inputs
st.sidebar.header("Select Visualization Options")

# User input for selecting tickers and graph type
selected_tickers = st.sidebar.multiselect("Select Stock Tickers", tickers, default=['GOOG'])
graph_type = st.sidebar.selectbox("Select Graph Type", ['Bar Graph', 'Histogram', '3D Pie Chart', '3D Histogram', 'Pairplot', 'Heatmap', 'Scatter Plot', '3D Scatter Plot'])

# Button to generate visualizations
if st.sidebar.button("Generate Visualization"):
    for ticker in selected_tickers:
        st.subheader(f"{graph_type} for {ticker}")
        generate_visualizations(df, ticker, graph_type)

# Footer
st.sidebar.markdown("Built using Streamlit")
