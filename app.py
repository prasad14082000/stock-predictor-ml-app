import gradio as gr
from run_pipeline import run_pipeline
import subprocess
import os
import matplotlib.pyplot as plt
import pandas as pd

# Define the function to run your pipeline and return plot paths
def run_pipeline(symbol, start_date, end_date, forecast_days):
    try:
        # Construct CLI command to run your pipeline
        cmd = f"python run_pipeline.py --symbol {symbol}.NS --start {start_date} --end {end_date} --forecast_days {forecast_days}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Set filenames
        base_name = symbol.upper()
        reports_dir = "reports"

        lstm_forecast_plot = os.path.join(reports_dir, f"{base_name}_lstm_forecast_plot.png")
        multistep_csv = os.path.join(reports_dir, f"{base_name}_elasticnet_multi_step_forecast.csv")

        # Load plots and CSV forecast preview
        if os.path.exists(lstm_forecast_plot) and os.path.exists(multistep_csv):
            df_preview = pd.read_csv(multistep_csv).tail(10)
            return lstm_forecast_plot, df_preview
        else:
            return None, "Forecast files not found. Please check symbol or dates."

    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio UI components
symbol_input = gr.Textbox(label="Stock Symbol (e.g., TCS, RELIANCE)")
start_input = gr.Textbox(label="Start Date (YYYY-MM-DD)")
end_input = gr.Textbox(label="End Date (YYYY-MM-DD)")
forecast_days_input = gr.Slider(1, 30, value=7, label="Forecast Days")

# Launch UI
demo = gr.Interface(
    fn=run_pipeline,
    inputs=[symbol_input, start_input, end_input, forecast_days_input],
    outputs=[gr.Image(label="LSTM Forecast Plot"), gr.Dataframe(label="Multi-Step Forecast Preview")],
    title="📈 Stock Predictor App",
    description="Run full pipeline with one click: LSTM + Multi-step Forecasting"
)

demo.launch()
