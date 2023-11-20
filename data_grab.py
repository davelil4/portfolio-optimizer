import yfinance as yf

def getHistory(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    return hist
    
    