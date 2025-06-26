import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.options.black_scholes import BlackScholes

st.set_page_config(
    page_title="📈 Black-Scholes Option Pricing",
    layout="wide",
    page_icon="📊"
)

st.title("Black-Scholes Option Pricing Model")
st.markdown("""
This app calculates **Call** and **Put** option prices and shows how they change with varying spot prices and volatilities.
""")

with st.sidebar:
    st.header("Input Parameters")
    current_price = st.number_input("Current Asset Price", min_value=1.0, value=100.0)
    strike = st.number_input("Strike Price", min_value=1.0, value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0)
    volatility = st.slider("Volatility (σ)", 0.01, 1.0, value=0.2)
    interest_rate = st.slider("Risk-Free Interest Rate (r)", 0.0, 0.2, value=0.05)

    st.markdown("---")
    st.subheader("Heatmap Settings")
    spot_range = np.linspace(current_price * 0.8, current_price * 1.2, 10)
    vol_range = np.linspace(volatility * 0.5, volatility * 1.5, 10)

# Calculate Prices
model = BlackScholes(
    time_to_maturity, strike, current_price, volatility, interest_rate
)

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

# Heatmaps
call_prices = np.zeros((len(vol_range), len(spot_range)))
put_prices = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        temp = BlackScholes(time_to_maturity, strike, s, v, interest_rate)
        temp.calculate_prices()
        call_prices[i, j] = temp.call_price
        put_prices[i, j] = temp.put_price

# Display Heatmaps
st.subheader("📊 Heatmaps")
col1, col2 = st.columns(2)

with col1:
    st.write("### Call Option Price Heatmap")
    fig_call, ax_call = plt.subplots()
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".1f", cmap="RdYlGn", ax=ax_call)
    ax_call.set_xlabel("Spot Price")
    ax_call.set_ylabel("Volatility")
    st.pyplot(fig_call)

with col2:
    st.write("### Put Option Price Heatmap")
    fig_put, ax_put = plt.subplots()
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".1f", cmap="RdYlGn", ax=ax_put)
    ax_put.set_xlabel("Spot Price")
    ax_put.set_ylabel("Volatility")
    st.pyplot(fig_put)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Not Financial Advice")
