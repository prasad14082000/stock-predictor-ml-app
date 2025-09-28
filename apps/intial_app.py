import gradio as gr
from data.inital_run_pipeline import run_pipeline
import subprocess
import os
import matplotlib.pyplot as plt
import pandas as pd


def run_pipeline(symbol, start_date, end_date, forecast_days):
    try:
        print(f"\n🚀 Running pipeline for {symbol}, {start_date} to {end_date}, {forecast_days} days")
        os.chdir("C://GITHUB CODES//stock-predictor-ml")
        cmd = f"python run_pipeline.py --symbol {symbol}.NS --start {start_date} --end {end_date} --forecast_days {forecast_days}"
        print("🛠️ Command:", cmd)

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("📜 STDOUT:\n", result.stdout)
        print("❌ STDERR:\n", result.stderr)

        base_name = symbol.upper()
        reports_dir = "reports"
        lstm_forecast_plot = os.path.join(reports_dir, f"{base_name}_lstm_forecast_plot.png")
        multistep_csv = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast.csv")
        multistep_forecast_plot = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast_plot.png")

        if os.path.exists(lstm_forecast_plot) and os.path.exists(multistep_csv) and os.path.exists(multistep_forecast_plot):
            df_preview = pd.read_csv(multistep_csv).tail(10)
            if 'Forecasted_Close' not in df_preview.columns:
                return None, pd.DataFrame({"Error": ["❌ 'Forecasted_Close' column not found in CSV"]}), None, "❌ Forecast failed."
            latest = df_preview.tail(1).iloc[0]
            first = df_preview.head(1).iloc[0]
            percent_change = ((latest["Forecasted_Close"] - first["Forecasted_Close"]) / first["Forecasted_Close"]) * 100
            trend = "📈 Uptrend Expected" if percent_change > 0 else "📉 Downtrend Expected"
            summary = f"Final Forecast: ₹{latest['Forecasted_Close']:.2f} on {latest['Date']}\nChange: {percent_change:.2f}% → {trend}"
            return lstm_forecast_plot, df_preview, multistep_forecast_plot, summary
        else:
            return None, pd.DataFrame({"Error": ["Forecast files not found. Please check symbol or dates."]}), None, "❌ Forecast failed."

    except Exception as e:
        return None, pd.DataFrame({"Error": [f"Error: {str(e)}"]}), None, "❌ Unexpected error occurred."


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 Stock Predictor App
    Predict future stock prices using ML models like ElasticNet and LSTM. Run full pipeline below.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 🛠️ Input Parameters
            Fill in the stock symbol and time range to forecast future prices.
            """)
            symbol = gr.Dropdown(
                choices=["TITAN", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "CIPLA", "BAJFINANCE", 'ITC', 'ONGC', 'JIOFIN', 'TRENT', 'NTPC', 'COALINDIA', 'WIPRO', 'MARUTI', 'HDFCLIFE'],
                label="Stock Symbol",
                value="TITAN"
            )
            start_date = gr.Textbox(label="Start Date", value="2020-01-01")
            end_date = gr.Textbox(label="End Date", value="2025-01-01")
            forecast_days = gr.Slider(1, 30, value=7, label="Forecast Days")
            run_btn = gr.Button("🚀 Run Forecast")

        with gr.Column(scale=2):
            gr.Markdown("""
            ### 🔍 Forecast Results
            Below are visual insights from LSTM and ElasticNet models.
            """)
            lstm_plot_output = gr.Image(label="📉 LSTM Forecast Plot", type="filepath")
            table_output = gr.Dataframe(label="📋 ElasticNet Forecast Preview")
            multistep_plot_output = gr.Image(label="📈 Multi-Step Forecast Plot", type="filepath")
            summary_output = gr.Textbox(label="📌 Summary Insight", lines=3)

    run_btn.click(fn=run_pipeline, inputs=[symbol, start_date, end_date, forecast_days],
                  outputs=[lstm_plot_output, table_output, multistep_plot_output, summary_output])

    gr.Markdown("""
    ---
    ✅ Built with ❤️, scikit-learn, and LSTM · "Not Financial Advice"
    """)


if __name__ == "__main__":
    demo.launch(share=True)
