
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


print(f"AAPL Current Price: ${get_stock_price('AAPL')}")

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

print(buy_stock("AAPL", 5, get_stock_price("AAPL")))
print(sell_stock("AAPL", 2))
print(portfolio)

def plot_stock(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1mo")

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{symbol} Stock Price Trend")
    plt.legend()
    plt.show()


plot_stock("AAPL")
