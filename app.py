import streamlit as st
import pandas as pd
import time
from model import (
    get_index_summary,
    get_market_summary,
    get_trending_stocks,
    get_stock_details,
    get_sentiment,
    get_forecast,
    get_risk_metrics,
    get_news,
    add_to_watchlist,
    get_watchlist,
    calculate_investment_return,
)

st.set_page_config(page_title="📈 Stock Analysis Dashboard", layout="wide")

# Sidebar - Stock Search and Auto Refresh
st.sidebar.header("🔍 Stock Search")

# ⏱️ Auto-refresh logic
st.sidebar.subheader("⏱️ Auto Refresh")
refresh_rate = st.sidebar.selectbox("Refresh interval (seconds):", [0, 30, 60, 120], index=1)

if refresh_rate > 0:
    time.sleep(refresh_rate)
    st.rerun()

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

# Advanced Analytics
st.subheader("📊 Advanced Analytics")

selected_symbol = st.text_input("Enter a stock symbol for advanced analytics:")
selected_period = st.selectbox("Select a period for analysis:", ['1mo', '3mo', '6mo', '1y', '2y'])

if st.button("Analyze"):
    if selected_symbol:
        data_result = get_stock_details(selected_symbol, period=selected_period)
        if data_result["status"] == "success":
            df_adv = pd.DataFrame(data_result["data"])
            st.write(f"📈 Moving Averages & Volatility for {selected_symbol}")
            st.line_chart(df_adv.set_index("Date")[["Close", "MA_20", "MA_50"]])

            st.write("📉 RSI & MACD")
            st.line_chart(df_adv.set_index("Date")[["RSI", "MACD", "Signal_Line"]])

            st.write("📊 Bollinger Bands")
            st.line_chart(df_adv.set_index("Date")[["Upper_Band", "Middle_Band", "Lower_Band"]])

            st.write("⚠️ Volatility & Risk")
            risk = get_risk_metrics(selected_symbol)
            st.write(risk)
        else:
            st.warning(data_result.get("message", "Unable to fetch data."))
    else:
        st.warning("Please enter a stock symbol to analyze.")

# Custom Analysis Section
st.subheader("⚙️ Custom Analysis")

custom_symbol = st.text_input("Enter a stock symbol for custom analysis:")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Run Custom Analysis"):
    if custom_symbol:
        result = get_stock_details(custom_symbol, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
        if result["status"] == "success":
            df_custom = pd.DataFrame(result["data"])
            st.write(f"📅 Price & Indicator Trends for {custom_symbol} from {start_date} to {end_date}")
            st.line_chart(df_custom.set_index("Date")[["Close", "MA_20", "MA_50"]])

            st.write("📉 RSI & MACD")
            st.line_chart(df_custom.set_index("Date")[["RSI", "MACD", "Signal_Line"]])

            st.write("📊 Bollinger Bands")
            st.line_chart(df_custom.set_index("Date")[["Upper_Band", "Middle_Band", "Lower_Band"]])

            st.write("⚠️ Volatility & Risk")
            risk = get_risk_metrics(custom_symbol)
            st.write(risk)
        else:
            st.error(result.get("message", "Failed to fetch data."))
    else:
        st.warning("Please enter a valid stock symbol.")

# 📤 Export Analysis Section
st.subheader("📤 Export Custom Analysis")

export_symbol = st.text_input("Enter stock symbol to export data:")
export_period = st.selectbox("Select period to export", ['1mo', '3mo', '6mo', '1y'])

if st.button("Download Report"):
    result = get_stock_details(export_symbol, period=export_period)
    if result["status"] == "success":
        df_export = pd.DataFrame(result["data"])
        if not df_export.empty:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f'{export_symbol}_analysis_{export_period}.csv',
                mime='text/csv'
            )
        else:
            st.warning("No data available to export.")
    else:
        st.error(result.get("message", "Failed to fetch data for export."))

# 🧠 ML-Based Strategy Suggestion
st.subheader("🧠 AI Strategy Suggestion & Forecast")

ml_symbol = st.text_input("Enter stock symbol for ML prediction:")
ml_days = st.slider("Days to forecast", min_value=3, max_value=30, value=7)

if st.button("Run Prediction"):
    ml_forecast = get_forecast(ml_symbol, days=ml_days)
    if not ml_forecast.empty:
        st.line_chart(ml_forecast.set_index("Date")["Forecast"])

        # Simple strategy suggestion
        change = ml_forecast["Forecast"].iloc[-1] - ml_forecast["Forecast"].iloc[0]
        if change > 0:
            st.success(f"📈 Forecast suggests a potential **uptrend** of ${round(change, 2)} over {ml_days} days. Consider buying.")
        elif change < 0:
            st.warning(f"📉 Forecast suggests a potential **downtrend** of ${round(abs(change), 2)} over {ml_days} days. Consider caution.")
        else:
            st.info("⚖️ Forecast indicates a neutral trend. No strong movement expected.")
    else:
        st.error("Prediction data not available.")
