import streamlit as st
import yfinance as yf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email details
sender_email = "adityaaasingh47@gmail.com"
receiver_email = "adssingh9090@gmail.com"
password = "uzqhczruofbxlrvj"  # Your App Password
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

# Download stock data for multiple tickers and check thresholds
def load_and_check_stock_data(tickers):
    # Load stock data
    stock_data = yf.download(tickers, start="2023-01-01", end="2023-12-31", group_by='ticker')

    for ticker in tickers:
        if ticker in stock_data.columns.levels[0]:  # Check if ticker data is available
            latest_data = stock_data[ticker].iloc[-1]  # Get the latest data for the ticker
            st.subheader(f"{ticker} latest data:")
            st.write(latest_data)

            # Check thresholds for Open, Close, High, Low
            for metric in price_thresholds[ticker]:
                # Ensure that the metric exists in latest_data
                if metric in latest_data.index:
                    latest_price = latest_data[metric]
                    threshold = price_thresholds[ticker][metric]

                    # Check if the latest price exceeds the threshold
                    if latest_price > threshold:
                        st.warning(f"{ticker} {metric} exceeds the threshold. Sending email...")

                        # Fetch educational insight for the metric
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

# Streamlit user interface
st.title("Notification System")

# Define the stock tickers
tickers = ['GOOG', 'NVDA', 'META', 'AMZN']
if st.button("Check Stock Data"):
    load_and_check_stock_data(tickers)
