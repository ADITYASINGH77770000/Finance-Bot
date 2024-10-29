import streamlit as st

# Main Title for the App
st.title("Finance Stock Analysis and Prediction App")

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Auditing", "Graphs", "Prediction", "Query"])

# Auditing Section
if section == "Auditing":
    st.header("Auditing in Finance")
    
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

# Graphs Section
if section == "Graphs":
    st.header("Graphs in Finance")

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

# Prediction Section
if section == "Prediction":
    st.header("Stock Prediction")

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

# Query Section
if section == "Query":
    st.header("Financial Query System")

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
