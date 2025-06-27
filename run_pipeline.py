#run_pipeline.py
import argparse
from src.data_loader import download_stock_data
from src.feature_engineering import add_features
from src.eda import eda_summary
from src.train_models import train_multiple_models
from src.forecast import forecast_next_days
from src.evaluate_models import evaluate_all_models      # Multi-step forecasting with best model
from src.multistep_forecast import forecast_multi_step
from src.forecast_with_lstm import forecast_with_lstm
from src.explainability import explain_model_shap
from sklearn.pipeline import Pipeline

import os
import pandas as pd
import joblib

def run_pipeline(symbol: str, start_date: str, end_date: str, forecast_days: int):
    print(f"\n Downloading data for {symbol}...")
    df = download_stock_data(symbol, start_date, end_date)

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    symbol_upper = symbol.upper()  # SYMBOL should be defined from user input

    # Save the most recent row (actual price) to CSV
    actual_price_df = df[['Date', 'Close']].tail(1)
    actual_csv_path = os.path.join(reports_dir, f"{symbol_upper}_actual_price.csv")
    actual_price_df.to_csv(actual_csv_path, index=False)

    print("\n Performing feature engineering...")
    df = add_features(df)

    print("\n Running EDA...")
    eda_summary(df, stock_name=symbol.replace(".NS", ""))

    print("\n Training models and evaluating...")
    train_multiple_models(df, stock_name=symbol.replace(".NS", ""))

    # -------- SHAP Explanations for Models ---------
    stock_name = symbol.replace(".NS", "")
    X_train_path = f"C://GITHUB CODES//stock-predictor-ml//data/processed/{stock_name}_X_train.pkl"
    if os.path.exists(X_train_path):
        X_train = pd.read_pickle(X_train_path)
        shap_model_names = ['elasticnet', 'random_forest', 'xgboost']
        for model_name in shap_model_names:
            model_path = f"C://GITHUB CODES//stock-predictor-ml//models/{stock_name}_{model_name}.pkl"
            if os.path.exists(model_path):
                pipeline, filtered_features = joblib.load(model_path)
                # For pipeline, get the actual model
                model = pipeline.named_steps['model']
                try:
                    explain_model_shap(model, X_train[filtered_features], stock_name, model_name)
                except Exception as e:
                    print(f"SHAP explanation failed for {model_name}: {e}")
    else:
        print(f"X_train pickle not found at: {X_train_path}. Skipping SHAP explanations.")


    # Save the processed DataFrame
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    print(" Columns before saving:", df.columns.tolist())

    df.to_pickle(f"{processed_dir}/{symbol.replace('.NS', '')}_v2.pkl")
    print(f"\n Pipeline completed. Processed data saved at: {processed_dir}/{symbol.replace('.NS', '')}_v2.pkl")

    if forecast_days > 0:
        models_to_forecast = ['linear_regression', 'elasticnet', 'xgboost', 'random_forest']
        for model_name in models_to_forecast:
            forecast_next_days(stock_name=symbol.replace(".NS", ""), model_name = model_name, days_ahead=forecast_days)

    evaluate_all_models(stock_name=symbol.replace(".NS",""), lookback_days=30)
    
    print("\n Running LSTM Forecast...")
    forecast_with_lstm(
        stock_name=symbol.replace(".NS", ""),
        forecast_days=forecast_days,   # You can pass any int here
        lookback=30                    # Or adjust the lookback window as needed
    )

    # Multi-step forecasting with best model
    from src.multistep_forecast import forecast_multi_step
    
    best_model_name = "elasticnet"  # update this based on evaluation results if needed
    print(f"\n Running multi-step forecast with model: {best_model_name}...")
    forecast_multi_step(
        stock_name=symbol.replace(".NS", ""),
        model_name=best_model_name,
        forecast_days=forecast_days
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock prediction pipeline.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock ticker symbol, e.g., RELIANCE.NS")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--forecast_days", type = int, default=0, help='Number of days to forecast')
    
    args = parser.parse_args()

    run_pipeline(args.symbol, args.start, args.end, args.forecast_days)
