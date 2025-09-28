# File: apps/unified_app.py

import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

# --- RAG imports ---
from src.rag.doc_indexer import files_to_documents, chunk_documents
from src.rag.chroma_db import build_chroma, load_chroma
from src.rag.query_engine import run_query
from src.rag.schema import QueryRequest, OptionForecast

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="📈 Unified Stock App", layout="wide")

st.title("📈 Stocks App: Stock Forecasting + Options Pricing")

# Useful paths (for the RAG index location and reports)
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
DATA_ROOT = REPO_ROOT / "data" / "research"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# TABS
# ----------------------------
forecast_tab, options_tab, research_tab = st.tabs(
    ["📊 Stock Forecast", "⚖️ Option Pricing", "🧠 Research (RAG)"]
)

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
    T = st.number_input("Time to Maturity (Years)", value=st.session_state.get("forecast_days", 7) / 252 if "forecast_days" in st.session_state else 7 / 252)
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, value=0.2)
    r = st.slider("Risk-Free Interest Rate", 0.0, 0.2, value=0.05)

    if st.button("🧮 Calculate Options"):
        # Lazy import to keep your existing module structure unchanged
        from src.options_pricing.black_scholes import BlackScholes
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

# ----------------------------
# RESEARCH (RAG) TAB
# ----------------------------
with research_tab:
    st.subheader("🧠 Document Research (RAG) + Model Context")

    # Use the chosen symbol if available
    default_ticker = "TITAN"
    try:
        if symbol:
            default_ticker = symbol
    except Exception:
        pass

    ticker = st.text_input("Ticker", value=default_ticker).upper()
    persist_dir = DATA_ROOT / ticker
    persist_dir.mkdir(parents=True, exist_ok=True)

    st.markdown("#### Upload research docs (PDF / TXT / CSV / XLSX)")
    up_files = st.file_uploader(
        "Drop files here", type=["pdf", "txt", "md", "csv", "xlsx"], accept_multiple_files=True
    )

    # --- Rebuild / Load controls (with confirmation) ---
    if "confirm_rebuild" not in st.session_state:
        st.session_state["confirm_rebuild"] = False

    rebuild_col, load_col = st.columns([1, 1])

    # Left: Build / Rebuild with confirmation
    with rebuild_col:
        if not st.session_state["confirm_rebuild"]:
            if st.button("🧱 Build / Rebuild Index"):
                if not up_files:
                    st.warning("Upload at least one file first.")
                else:
                    st.session_state["confirm_rebuild"] = True
        else:
            st.warning("This will overwrite the existing index for this ticker.")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Confirm rebuild ✅"):
                    try:
                        # Build the index
                        tupled = [(f.name, f.read(), f.type or "") for f in up_files]
                        page_docs = files_to_documents(tupled)
                        chunks = chunk_documents(page_docs, chunk_size=800, chunk_overlap=120)
                        build_chroma(chunks, persist_dir=str(persist_dir))
                        st.success(f"Indexed {len(chunks)} chunks → {persist_dir}")
                    except Exception as e:
                        st.error(f"Index build failed: {e}")
                    finally:
                        st.session_state["confirm_rebuild"] = False
            with c2:
                if st.button("Cancel"):
                    st.session_state["confirm_rebuild"] = False

    # Right: Load existing index
    with load_col:
        if st.button("📦 Load Existing Index"):
            try:
                _ = load_chroma(persist_dir=str(persist_dir))
                st.success(f"Loaded index at {persist_dir}")
            except Exception as e:
                st.error(f"Failed to load: {e}")


    st.markdown("#### Option forecast context (optional)")
    latest_price = st.session_state.get("latest_price", None)
    last_actual = st.session_state.get("last_actual_price", None)

    def month_end(d: date) -> date:
        # get last day of the month
        next_month = d.replace(day=28) + timedelta(days=4)  # always next month
        last_day = next_month - timedelta(days=next_month.day)
        # if weekend, roll back to Friday
        while last_day.weekday() > 4:  # 0=Mon ... 6=Sun
            last_day -= timedelta(days=1)
        return last_day

    default_expiry = month_end(date.today())
    colf = st.columns(6)
    spot_v = colf[0].number_input("Spot", value=float(latest_price or last_actual or 100.0))
    strike_v = colf[1].number_input("Strike", value=round(spot_v, 2))
    expiry_v = colf[2].text_input("Expiry (YYYY-MM-DD)", value=default_expiry.strftime("%Y-%m-%d"))
    iv_v = colf[3].number_input("IV (optional)", value=0.20, min_value=0.0, step=0.01)
    model_price_v = colf[4].number_input("Model Price (optional)", value=0.0)
    opt_type_v = colf[5].selectbox("Type", ["CALL", "PUT"], index=0)

    forecasts = [
        OptionForecast(
            ticker=ticker,
            expiry=expiry_v,
            strike=float(strike_v),
            option_type=opt_type_v,
            model_price=float(model_price_v),
            iv=float(iv_v) if iv_v else None,
            spot=float(spot_v),
            source="unified_app",
        )
    ]

    st.markdown("#### Ask a question")
    q = st.text_input("e.g., What drives H2 margins? Key risks given latest concall and ratings?")
    colq = st.columns(3)
    use_mmr = colq[0].checkbox("Use MMR retrieval", value=True)
    top_k = colq[1].slider("Top K", 3, 10, 5)

    if st.button("🔎 Research"):
        try:
            db = load_chroma(persist_dir=str(persist_dir))
        except Exception:
            st.error("No index found. Upload & build first.")
        else:
            req = QueryRequest(
                question=q or "Summarize the most material drivers and risks.",
                ticker=ticker,
                top_k=int(top_k),
                use_mmr=bool(use_mmr),
                forecasts=forecasts,
            )
            resp = run_query(db, req)

            st.markdown("### Thesis")
            thesis = resp.thesis
            st.write({
                "ticker": thesis.ticker,
                "one_line_view": thesis.one_line_view,
                "view_confidence_10pt": thesis.view_confidence_10pt,
                "key_drivers": thesis.key_drivers,
                "risks": thesis.risks,
                "suggested_actions": thesis.suggested_actions,
                "references": thesis.references,
            })

            st.markdown("### 🔗 Citations")
            if resp.thesis.references:
                for r in resp.thesis.references:
                    st.markdown(f"- `{r}`")
            else:
                st.caption("No references reported by the model (try raising Top-K or re-building the index).")

            with st.expander("📄 Retrieved context (what the model saw)"):
                for rc in resp.retrieved:
                    st.markdown(f"**{rc.source}:{rc.page}** — *{(rc.asof_date or 'n/a')}*")
                    st.write(rc.snippet[:800] + ("..." if len(rc.snippet) > 800 else ""))
                    st.divider()
                        
            st.markdown("### Retrieved Chunks")
            for ch in resp.retrieved:
                # `ResearchChunk` in your schema uses `text` as the field name
                snippet = getattr(ch, "snippet", None) or getattr(ch, "text", "")
                ref = f"{ch.source}:{ch.page}" if ch.page else ch.source
                meta = " • ".join(x for x in [ch.source_type or "other", ch.asof_date or ""] if x)
                st.caption(f"[{ref}] {meta}")
                st.text(snippet[:1200])

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Forecast + Derivatives Pricing + RAG Research | Not Financial Advice")
