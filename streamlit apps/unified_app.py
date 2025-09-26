# File: apps/unified_app.py

import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.options_pricing.black_scholes import BlackScholes

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="📈 Unified Stock App", layout="wide")

st.title("📈 Stocks App: Stock Forecasting + Options Pricing")

# ----------------------------
# TABS
# ----------------------------
forecast_tab, options_tab = st.tabs(["📊 Stock Forecast", "⚖️ Option Pricing"])

# ----------------------------
# FORECASTING TAB
# ----------------------------
with forecast_tab:
    st.subheader("📊 Stock Price Predictor App")
    st.markdown("Predict future stock prices using ML models like ElasticNet and LSTM.")

    with st.sidebar:
        st.markdown("### 🛠️ Forecast Inputs")
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

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        base_name = symbol.upper()
        reports_dir = "reports"
        lstm_plot = os.path.join(reports_dir, f"{base_name}_lstm_forecast_plot.png")
        multistep_csv = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast.csv")
        multistep_plot = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast_plot.png")

        if os.path.exists(lstm_plot) and os.path.exists(multistep_csv) and os.path.exists(multistep_plot):
            df = pd.read_csv(multistep_csv).tail(10)
            if 'Forecasted_Close' not in df.columns:
                st.error("❌ 'Forecasted_Close' column not found in forecast CSV")
            else:
                first, latest = df.head(1).iloc[0], df.tail(1).iloc[0]
                pct = ((latest["Forecasted_Close"] - first["Forecasted_Close"]) / first["Forecasted_Close"]) * 100
                trend = "📈 Uptrend Expected" if pct > 0 else "📉 Downtrend Expected"
                summary = f"Final Forecast: ₹{latest['Forecasted_Close']:.2f} on {latest['Date']}\nChange: {pct:.2f}% → {trend}"

                col1, col2 = st.columns(2)
                with col1:
                    st.image(lstm_plot, caption="📉 LSTM Forecast Plot")
                    st.image(multistep_plot, caption="📈 Multi-Step Forecast Plot")
                with col2:
                    st.markdown("### 📋 ElasticNet Forecast Preview")
                    st.dataframe(df)
                    st.text_area("📌 Summary Insight", summary, height=90)

                st.session_state["latest_price"] = latest["Forecasted_Close"]

                # Save latest actual price to session_state from CSV (written by pipeline)
                try:
                    actual_csv = os.path.join("reports", f"{base_name}.NS_actual_price.csv")
                    if os.path.exists(actual_csv):
                        actual_df = pd.read_csv(actual_csv)
                        st.session_state["last_actual_price"] = float(actual_df["Close"].iloc[-1])
                    else:
                        st.session_state["last_actual_price"] = 100.0
                except Exception as e:
                    st.session_state["last_actual_price"] = 100.0
                else:
                    st.error("❌ Forecast failed. Files not found.")

# ----------------------------
# OPTIONS TAB
# ----------------------------
with options_tab:
    st.subheader("⚖️ Black-Scholes Option Pricing")

    st.markdown("---")
    st.markdown("Use the forecasted price as the Spot Price or enter manually below.")

    spot_source = st.radio("Select Spot Price Source", ["🔮 ML Forecast", "📊 Actual Price", "✍️ Manual"])
    forecasted_price = st.session_state.get("latest_price", 100.0)
    actual_price = st.session_state.get("last_actual_price", 100.0)

    if spot_source == "🔮 ML Forecast":
        S = forecasted_price
        st.success(f"Using ML Forecasted Price: ₹{S:.2f}")
    elif spot_source == "📊 Actual Price":
        S = actual_price
        st.info(f"Using Actual Last Price: ₹{S:.2f}")
    else:
        S = st.number_input("Spot Price (Manual Entry)", value=100.0)

    K = st.number_input("Strike Price", value=round(S, 2))
    T = st.number_input("Time to Maturity (Years)", value=forecast_days / 252)
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, value=0.2)
    r = st.slider("Risk-Free Interest Rate", 0.0, 0.2, value=0.05)

    if st.button("🧮 Calculate Options"):
        model = BlackScholes(T, K, S, sigma, r)
        model.calculate_prices()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Call Option Price", f"₹{model.call_price:.2f}")
            st.metric("Call Delta", f"{model.call_delta:.4f}")
            st.metric("Call Gamma", f"{model.call_gamma:.4f}")

        with col2:
            st.metric("Put Option Price", f"₹{model.put_price:.2f}")
            st.metric("Put Delta", f"{model.put_delta:.4f}")
            st.metric("Put Gamma", f"{model.put_gamma:.4f}")

        spot_range = np.linspace(S * 0.8, S * 1.2, 10)
        vol_range = np.linspace(sigma * 0.5, sigma * 1.5, 10)
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        put_prices = np.zeros((len(vol_range), len(spot_range)))

        for i, v in enumerate(vol_range):
            for j, s in enumerate(spot_range):
                tmp = BlackScholes(T, K, s, v, r)
                tmp.calculate_prices()
                call_prices[i, j] = tmp.call_price
                put_prices[i, j] = tmp.put_price

        st.subheader("📊 Option Price Heatmaps")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Call Option Heatmap")
            fig1, ax1 = plt.subplots()
            sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".1f", cmap="RdYlGn", ax=ax1)
            ax1.set_xlabel("Spot Price")
            ax1.set_ylabel("Volatility")
            st.pyplot(fig1)

        with c2:
            st.markdown("#### Put Option Heatmap")
            fig2, ax2 = plt.subplots()
            sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".1f", cmap="RdYlGn", ax=ax2)
            ax2.set_xlabel("Spot Price")
            ax2.set_ylabel("Volatility")
            st.pyplot(fig2)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Forecast + Derivatives Pricing | Not Financial Advice")
