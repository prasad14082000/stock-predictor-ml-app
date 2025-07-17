#stock_forecast_app.py

import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="📊 Stock Predictor App", layout="wide")

st.markdown("""
# 📊 Stock Predictor App
Predict future stock prices using ML models like ElasticNet and LSTM. Run full pipeline below.
""")

with st.sidebar:
    st.markdown("""
    ### 🛠️ Input Parameters
    Fill in the stock symbol and time range to forecast future prices.
    """)
    symbol = st.selectbox("Stock Symbol", [
        "TITAN", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
        "CIPLA", "BAJFINANCE", "ITC", "ONGC", "JIOFIN", "TRENT", "NTPC", "COALINDIA", "WIPRO", "MARUTI", "HDFCLIFE"
    ], index=0)
    start_date = st.text_input("Start Date", value="2020-01-01")
    end_date = st.text_input("End Date", value="2025-01-01")
    forecast_days = st.slider("Forecast Days", 1, 30, 7)
    run_forecast = st.button("🚀 Run Forecast")

if run_forecast:
    st.info(f"\n🚀 Running pipeline for {symbol}, {start_date} to {end_date}, {forecast_days} days")
    os.chdir("C://GITHUB CODES//stock-predictor-ml")
    cmd = f"python run_pipeline.py --symbol {symbol}.NS --start {start_date} --end {end_date} --forecast_days {forecast_days}"
    st.code(cmd, language="bash")

    # Add this line in your sidebar UI
    debug = st.sidebar.checkbox("Show Debug Logs", value=False)     

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if debug:
        with st.expander("📜 STDOUT (Click to Expand)", expanded=False):
            st.text(result.stdout if result.stdout.strip() else "No STDOUT output.")

        with st.expander("❌ STDERR (Click to Expand)", expanded=False):
            st.text(result.stderr if result.stderr.strip() else "No STDERR output.")

    base_name = symbol.upper()
    reports_dir = "reports"
    lstm_forecast_plot = os.path.join(reports_dir, f"{base_name}_lstm_forecast_plot.png")
    multistep_csv = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast.csv")
    multistep_forecast_plot = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast_plot.png")

    if os.path.exists(lstm_forecast_plot) and os.path.exists(multistep_csv) and os.path.exists(multistep_forecast_plot):
        df_preview = pd.read_csv(multistep_csv).tail(10)

        if 'Forecasted_Close' not in df_preview.columns:
            st.error("❌ 'Forecasted_Close' column not found in CSV")
        else:
            latest = df_preview.tail(1).iloc[0]
            first = df_preview.head(1).iloc[0]
            percent_change = ((latest["Forecasted_Close"] - first["Forecasted_Close"]) / first["Forecasted_Close"]) * 100
            trend = "📈 Uptrend Expected" if percent_change > 0 else "📉 Downtrend Expected"
            summary = f"Final Forecast: ₹{latest['Forecasted_Close']:.2f} on {latest['Date']}\nChange: {percent_change:.2f}% → {trend}"

            st.markdown("""
            ### 🔍 Forecast Results
            Below are visual insights from LSTM and ElasticNet models.
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.image(lstm_forecast_plot, caption="📉 LSTM Forecast Plot")
                st.image(multistep_forecast_plot, caption="📈 Multi-Step Forecast Plot")

            with col2:
                st.markdown("### 📋 ElasticNet Forecast Preview")
                st.dataframe(df_preview)
                st.text_area("📌 Summary Insight", summary, height=90)
    else:
        st.error("❌ Forecast failed. One or more output files not found.")
