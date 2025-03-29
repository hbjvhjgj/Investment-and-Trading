import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")
    if not data.empty:
        return data["Close"].iloc[-1]
    return "Stock symbol not found!"

portfolio = {}

def buy_stock(symbol, quantity, price):
    if symbol in portfolio:
        portfolio[symbol]["quantity"] += quantity
        portfolio[symbol]["total_cost"] += quantity * price
    else:
        portfolio[symbol] = {"quantity": quantity, "total_cost": quantity * price}
    return f"Bought {quantity} shares of {symbol} at ${price} each."

def sell_stock(symbol, quantity):
    if symbol in portfolio and portfolio[symbol]["quantity"] >= quantity:
        portfolio[symbol]["quantity"] -= quantity
        if portfolio[symbol]["quantity"] == 0:
            del portfolio[symbol]
        return f"Sold {quantity} shares of {symbol}."
    return "Not enough shares to sell!"

# Streamlit UI
st.title("Stock Price Estimator")

symbol = st.text_input("Enter Stock Symbol:", "AAPL")

if st.button("Get Stock Price"):
    price = get_stock_price(symbol)
    st.write(f"Current Price of {symbol}: ${price}")

if st.button("Plot Stock Trend"):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1mo")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Close"], label="Close Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{symbol} Stock Price Trend")
    ax.legend()

    st.pyplot(fig)  # This ensures the plot is displayed in Streamlit
