# app_options.py
from src.options_pricing.black_scholes import BlackScholes
import streamlit as st

st.set_page_config(page_title="📈 Black-Scholes Options", layout="centered")

st.title("📈 Black-Scholes Options Pricing")

st.sidebar.header("📊 Input Parameters")
spot = st.sidebar.number_input("Current Price", value=100.0)
strike = st.sidebar.number_input("Strike Price", value=100.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (in years)", value=1.0)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.2)
rate = st.sidebar.number_input("Risk-free Interest Rate", value=0.05)

if st.sidebar.button("Calculate Options"):
    model = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=spot,
        volatility=volatility,
        interest_rate=rate
    )
    call, put = model.calculate_prices()

    st.success(f"📞 **Call Price**: ₹{call:.2f}")
    st.success(f"📩 **Put Price**: ₹{put:.2f}")
