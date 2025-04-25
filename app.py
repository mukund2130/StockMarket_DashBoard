import streamlit as st
import pandas as pd
from model import (
    get_index_summary,
    get_market_summary,
    get_trending_stocks,
    get_stock_details,
    get_sentiment,
    get_forecast,
    get_risk_metrics,
    get_news,  # <- this works ONLY if get_news is in model.py
    add_to_watchlist,
    get_watchlist,
)


st.set_page_config(page_title="📈 Stock Analysis Dashboard", layout="wide")

# Sidebar - Stock Search and Watchlist
st.sidebar.header("🔍 Stock Search")
stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, TSLA, GOOGL):").upper()
if stock_symbol:
    if st.sidebar.button("Add to Watchlist"):
        add_to_watchlist(stock_symbol)
    st.sidebar.subheader("📋 Watchlist")
    watchlist = get_watchlist()
    for stock in watchlist:
        st.sidebar.write(stock)

# Main Dashboard Title
st.title("📈 Stock Analysis Dashboard")

# Index Summary
st.subheader("📊 Market Indices")
indices = get_index_summary()
if indices:
    cols = st.columns(len(indices))
    for col, index in zip(cols, indices):
        delta_color = "normal" if index['up'] else "inverse"
        col.metric(label=index['name'], value=index['value'],
                   delta=f"{index['change']} ({index['percent']}%)", delta_color=delta_color)
else:
    st.warning("Index summary could not be loaded.")

# Market Summary
st.subheader("📈 Market Summary (Top 5 from Selected Stocks)")
summary = get_market_summary()
col1, col2, col3 = st.columns(3)
with col1:
    st.write("🚀 Top Gainers")
    st.dataframe(pd.DataFrame(summary.get("gainers", [])), use_container_width=True)
with col2:
    st.write("🔻 Top Losers")
    st.dataframe(pd.DataFrame(summary.get("losers", [])), use_container_width=True)
with col3:
    st.write("🔥 Most Active")
    st.dataframe(pd.DataFrame(summary.get("active", [])), use_container_width=True)

# Trending Stocks
st.subheader("💹 Trending Stocks (S&P 500)")
trending = get_trending_stocks()
if isinstance(trending, dict):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🚀 Top Gainers")
        st.dataframe(pd.DataFrame(trending.get("Top Gainers", [])), use_container_width=True)
    with col2:
        st.write("🔻 Top Losers")
        st.dataframe(pd.DataFrame(trending.get("Top Losers", [])), use_container_width=True)
    with col3:
        st.write("🔥 Most Active")
        st.dataframe(pd.DataFrame(trending.get("Most Active", [])), use_container_width=True)
else:
    st.error("Could not load trending stocks.")

# Stock Details
if stock_symbol:
    st.subheader(f"📋 Stock Details: {stock_symbol}")
    period = st.selectbox("Select period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
    details = get_stock_details(stock_symbol, period=period)
    if details.get("status") == "success":
        df = pd.DataFrame(details["data"])
        st.dataframe(df, use_container_width=True)
        if not df.empty:
            st.line_chart(df.set_index("Date")[["Close", "MA_20", "MA_50"]])
    else:
        st.error(details.get("message", "Failed to fetch stock details."))

    # Sentiment Analysis
    st.subheader("🧠 Sentiment Analysis")
    sentiment = get_sentiment(stock_symbol)
    if sentiment:
        st.write(sentiment)
    else:
        st.warning("No sentiment data available.")

    # AI-Based Forecast
    st.subheader("🤖 AI-Based Forecast")
    forecast = get_forecast(stock_symbol)
    if not forecast.empty:
        st.line_chart(forecast.set_index("Date")["Forecast"])
    else:
        st.warning("Forecast data not available.")

    # Risk Analysis
    st.subheader("⚠️ Risk Analysis")
    risk_metrics = get_risk_metrics(stock_symbol)
    if risk_metrics:
        st.write(risk_metrics)
    else:
        st.warning("Risk data not available.")

    # News & Alerts
    st.subheader("📰 Latest News")
    news = get_news(stock_symbol)
    if news:
        for article in news:
            st.write(f"- [{article['title']}]({article['url']})")
    else:
        st.warning("No news articles found.")

# Investment Calculator
st.subheader("💰 Investment Calculator")
investment_symbol = st.text_input("Enter stock symbol for investment calculation:")
investment_amount = st.number_input("Enter investment amount ($):", min_value=0.0)
investment_period = st.selectbox("Select investment period", ['1mo', '3mo', '6mo', '1y', '2y'])
if st.button("Calculate Return"):
    result = calculate_investment_return(investment_symbol, investment_amount, investment_period)
    st.write(result)
