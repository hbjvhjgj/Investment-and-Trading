import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .main {background-color: #f4f4f4;}
        .stButton button {background-color: #4CAF50; color: white; font-size: 16px;}
        .stTextInput>div>div>input {font-size: 18px;}
        .stTitle {color: #4A90E2; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("📈 Stock Market Dashboard")
st.subheader("Analyze stock trends, manage your portfolio, and predict future prices!")

# --- Get Stock Data Function ---
def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            return round(data["Close"].iloc[-1], 2)
    except:
        return "Invalid Symbol"

# --- Portfolio Management ---
portfolio = {}

def buy_stock(symbol, quantity, price):
    if symbol in portfolio:
        portfolio[symbol]["quantity"] += quantity
        portfolio[symbol]["total_cost"] += quantity * price
    else:
        portfolio[symbol] = {"quantity": quantity, "total_cost": quantity * price}

def sell_stock(symbol, quantity):
    if symbol in portfolio and portfolio[symbol]["quantity"] >= quantity:
        portfolio[symbol]["quantity"] -= quantity
        if portfolio[symbol]["quantity"] == 0:
            del portfolio[symbol]
        return f"✅ Sold {quantity} shares of {symbol}."
    return "⚠️ Not enough shares to sell!"

# --- Sidebar: Stock Selection ---
st.sidebar.header("🔍 Select a Stock")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL")

# --- Fetch Stock Data ---
current_price = get_stock_price(symbol)
st.sidebar.markdown(f"### 📊 {symbol} Current Price: **${current_price}**")

# --- Portfolio Management ---
st.sidebar.subheader("📂 Manage Your Portfolio")

buy_quantity = st.sidebar.number_input("Buy Quantity", min_value=1, value=5)
if st.sidebar.button("Buy Stock"):
    buy_stock(symbol, buy_quantity, current_price)
    st.sidebar.success(f"✅ Bought {buy_quantity} shares of {symbol} at ${current_price} each.")

sell_quantity = st.sidebar.number_input("Sell Quantity", min_value=1, value=2)
if st.sidebar.button("Sell Stock"):
    st.sidebar.warning(sell_stock(symbol, sell_quantity))

# Display portfolio
if portfolio:
    st.sidebar.subheader("📜 Your Portfolio")
    st.sidebar.table(pd.DataFrame.from_dict(portfolio, orient="index"))

# --- Main Section: Stock Analysis ---
st.header(f"📉 {symbol} Stock Trend Analysis")

# --- Fetch Historical Data ---
stock = yf.Ticker(symbol)
data = stock.history(period="6mo")

# --- Plot Stock Prices ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data["Close"], label="Closing Price", color="blue", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{symbol} Stock Price Trend (Last 6 Months)")
ax.legend()

st.pyplot(fig)

# --- Machine Learning Prediction ---
st.subheader("🔮 Predict Future Stock Prices (Next 7 Days)")
if len(data) > 10:
    # Prepare data for ML model
    data["Days"] = np.arange(len(data)).reshape(-1, 1)
    X = np.array(data["Days"]).reshape(-1, 1)
    y = np.array(data["Close"]).reshape(-1, 1)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 7 days
    future_days = np.arange(len(data), len(data) + 7).reshape(-1, 1)
    predictions = model.predict(future_days)

    # Convert to DataFrame
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]
    prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions.flatten()})
    st.table(prediction_df)

    # Plot Predictions
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data["Close"], label="Actual Prices", color="blue")
    ax.plot(future_dates, predictions, label="Predicted Prices", linestyle="dashed", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{symbol} Price Forecast (Next 7 Days)")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("⚠️ Not enough historical data to make predictions!")

st.markdown("---")

# --- Footer ---
st.markdown("""
    <h4 style="text-align: center;">📊 Developed with ❤️ using Streamlit</h4>
""", unsafe_allow_html=True)
