import yfinance as yf
import pandas as pd
import math
from dotenv import load_dotenv
from stocksymbol import StockSymbol
import os
load_dotenv()

def getHistory(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    return hist


def getCloseReturns(symbol) -> pd.DataFrame:
    price_df = getHistory(symbol)
    return (price_df["Close"] / price_df["Close"].shift(1)).iloc[1:, ]

def getLogReturnsFromSymbol(symbol):
    df = getCloseReturns(symbol)
    df["Returns"] = df.apply(lambda x: math.log(x))
    return df["Returns"]
    
def getLogReturnsFromList(symbols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for symbol in symbols:
        df[symbol] = getLogReturnsFromSymbol(symbol)
    return df

ss = StockSymbol(os.environ["STOCK_SYMBOL_API_KEY"])
stock_symbols = list(map(lambda x: x['symbol'], ss.get_symbol_list(market="US")))