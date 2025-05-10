import yfinance as yf
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression

WATCHLIST_FILE = "watchlist.json"

def get_market_summary():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    data = yf.download(symbols, period="2d", group_by='ticker', threads=True)

    gainers, losers, active = [], [], []

    for symbol in symbols:
        df = data.get(symbol)
        if df is None or df.empty or len(df) < 2:
            continue

        close_today = df['Close'].iloc[-1]
        close_yesterday = df['Close'].iloc[-2]
        change = close_today - close_yesterday
        percent_change = (change / close_yesterday) * 100
        volume = df['Volume'].iloc[-1]

        if not np.isnan(percent_change):
            gainers.append({"symbol": symbol, "change": round(percent_change, 2)})
            losers.append({"symbol": symbol, "change": round(percent_change, 2)})
        
        if not np.isnan(volume):
            active.append({"symbol": symbol, "volume": int(volume)})

    gainers = sorted(gainers, key=lambda x: x['change'], reverse=True)[:5]
    losers = sorted(losers, key=lambda x: x['change'])[:5]
    active = sorted(active, key=lambda x: x['volume'], reverse=True)[:5]

    return {
        "gainers": gainers,
        "losers": losers,
        "active": active
    }

def get_trending_stocks():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    try:
        data = yf.download(symbols, period="2d", group_by='ticker', threads=True)
        trending_data = []

        for symbol in symbols:
            df = data.get(symbol)
            if df is None or df.empty or len(df) < 2:
                continue

            close_today = df['Close'].iloc[-1]
            close_yesterday = df['Close'].iloc[-2]
            change = close_today - close_yesterday
            percent_change = (change / close_yesterday) * 100
            volume = df['Volume'].iloc[-1]

            trending_data.append({
                "symbol": symbol,
                "change": round(percent_change, 2),
                "volume": int(volume),
                "close": round(close_today, 2)
            })

        top_gainers = sorted(trending_data, key=lambda x: x['change'], reverse=True)[:5]
        top_losers = sorted(trending_data, key=lambda x: x['change'])[:5]
        most_active = sorted(trending_data, key=lambda x: x['volume'], reverse=True)[:5]

        return {
            "Top Gainers": top_gainers,
            "Top Losers": top_losers,
            "Most Active": most_active
        }

    except Exception as e:
        return {"error": str(e)}

def get_index_summary():
    indices = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}
    data = yf.download(list(indices.keys()), period="2d", group_by='ticker')

    summary = []
    for symbol, name in indices.items():
        df = data.get(symbol)
        if df is None or df.empty or len(df) < 2:
            continue
        close = df['Close'].iloc[-1]  # Access the 'Close' column
        prev = df['Close'].iloc[-2]
        change = close - prev
        percent = (change / prev) * 100
        summary.append({
            "name": name,
            "value": round(close, 2),
            "change": round(change, 2),
            "percent": round(percent, 2),
            "up": change >= 0
        })
    return summary

def get_stock_details(symbol, period='1mo', start_date=None, end_date=None):
    stock = yf.Ticker(symbol)
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    hist = stock.history(start=start_date, end=end_date) if start_date or end_date else stock.history(period=period)

    if hist.empty:
        return {"status": "error", "message": f"No data for {symbol}", "symbol": symbol}

    hist = calculate_technical_indicators(hist)
    data = []
    for index, row in hist.iterrows():
        change = row['Close'] - row['Open']
        data.append({
            "Date": index.strftime('%Y-%m-%d'),
            "Open": round(row['Open'], 2),
            "High": round(row['High'], 2),
            "Low": round(row['Low'], 2),
            "Close": round(row['Close'], 2),
            "Volume": int(row['Volume']),
            "Change": round(change, 2),
            "ChangePercent": round((change / row['Open']) * 100, 2) if row['Open'] != 0 else 0,
            "MA_20": round(row['MA_20'], 2) if 'MA_20' in row else None,
            "MA_50": round(row['MA_50'], 2) if 'MA_50' in row else None,
            "RSI": round(row['RSI'], 2) if 'RSI' in row else None,
            "MACD": round(row['MACD'], 2) if 'MACD' in row else None,
            "Signal_Line": round(row['Signal_Line'], 2) if 'Signal_Line' in row else None,
            "Upper_Band": round(row['Upper_Band'], 2) if 'Upper_Band' in row else None,
            "Middle_Band": round(row['Middle_Band'], 2) if 'Middle_Band' in row else None,
            "Lower_Band": round(row['Lower_Band'], 2) if 'Lower_Band' in row else None
        })
    return {
        "status": "success",
        "symbol": symbol,
        "data": data,
        "period": period
    }

def calculate_technical_indicators(df):
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['Middle_Band'] + 2 * std
    df['Lower_Band'] = df['Middle_Band'] - 2 * std
    return df

def get_forecast(symbol, days=7):
    df = yf.download(symbol, period="3mo")
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df['Day'] = range(len(df))
    model = LinearRegression()
    model.fit(df[['Day']], df['Close'])
    future_days = np.array(range(len(df), len(df) + days)).reshape(-1, 1)
    preds = model.predict(future_days)
    forecast = pd.DataFrame({
        "Date": pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days),
        "Forecast": [round(float(pred), 2) for pred in preds]
    })
    return forecast

def get_sentiment(symbol):
    return {
        "symbol": symbol,
        "sentiment": "Positive",
        "score": 0.74,
        "summary": f"Recent news sentiment for {symbol} indicates positive outlook with good investor confidence."
    }

def get_risk_metrics(symbol):
    df = yf.download(symbol, period="3mo")
    if df.empty:
        return {"symbol": symbol, "risk": "Unknown"}

    # Compute daily returns and standard deviation
    pct_changes = df['Close'].pct_change()
    std_dev = pct_changes.std()

    # Explicitly cast to float to avoid FutureWarning
    volatility = float(std_dev) if not isinstance(std_dev, pd.Series) else float(std_dev.iloc[0])

    if volatility < 0.015:
        risk = "Low"
    elif volatility < 0.03:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "symbol": symbol,
        "volatility": round(volatility, 4),
        "risk": risk
    }

def get_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return []

def add_to_watchlist(symbol):
    watchlist = get_watchlist()
    if symbol not in watchlist:
        watchlist.append(symbol)
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(watchlist, f)

def get_news(symbol):
    # Replace these with actual dynamic content if you connect to a news API
    return [
        {
            "title": f"{symbol} hits new 52-week high",
            "url": f"https://finance.yahoo.com/quote/{symbol}/news"
        },
        {
            "title": f"{symbol} announces quarterly earnings",
            "url": f"https://finance.yahoo.com/quote/{symbol}/news"
        },
        {
            "title": f"{symbol} stock rating upgraded by analysts",
            "url": f"https://finance.yahoo.com/quote/{symbol}/news"
        },
    ]

def calculate_investment_return(symbol, amount, period):
    df = yf.download(symbol, period=period)
    if df.empty:
        return f"No data available for {symbol}."

    start_price = float(df['Close'].iloc[0])
    end_price = float(df['Close'].iloc[-1])
    shares = amount / start_price
    final_value = shares * end_price
    profit = final_value - amount

    return {
        "Symbol": symbol,
        "Investment Amount": f"${amount:.2f}",
        "Start Price": f"${start_price:.2f}",
        "End Price": f"${end_price:.2f}",
        "Shares Bought": f"{shares:.2f}",
        "Final Value": f"${final_value:.2f}",
        "Profit/Loss": f"${profit:.2f}"
    }
