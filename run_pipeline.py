#run_pipeline.py
import argparse
from src.data_loader import download_stock_data
from src.feature_engineering import add_features
from src.eda import eda_summary
from src.train_models import train_multiple_models
from src.forecast import forecast_next_days
from src.evaluate_models import evaluate_all_models
import os
import pandas as pd


def run_pipeline(symbol: str, start_date: str, end_date: str, forecast_days: int):
    print(f"\n📥 Downloading data for {symbol}...")
    df = download_stock_data(symbol, start_date, end_date)

    print("\n🔧 Performing feature engineering...")
    df = add_features(df)

    print("\n📊 Running EDA...")
    eda_summary(df, stock_name=symbol.replace(".NS", ""))

    print("\n🤖 Training models and evaluating...")
    train_multiple_models(df, stock_name=symbol.replace(".NS", ""))

    if forecast_days > 0:
        models_to_forecast = ['linear_regression', 'elasticnet', 'xgboost', 'random_forest']
        for model_name in models_to_forecast:
            forecast_next_days(stock_name=symbol.replace(".NS", ""), model_name = model_name, days_ahead=forecast_days)

    # Save the processed DataFrame
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    print("✅ Columns before saving:", df.columns.tolist())
    
    df.to_pickle(f"{processed_dir}/{symbol.replace('.NS', '')}_v2.pkl")
    print(f"\n✅ Pipeline completed. Processed data saved at: {processed_dir}/{symbol.replace('.NS', '')}_v2.pkl")

    evaluate_all_models(stock_name=symbol.replace(".NS",""), lookback_days=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock prediction pipeline.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock ticker symbol, e.g., RELIANCE.NS")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--forecast_days", type = int, default=0, help='Number of days to forecast')
    
    args = parser.parse_args()

    run_pipeline(args.symbol, args.start, args.end, args.forecast_days)
