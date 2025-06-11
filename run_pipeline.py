import argparse
from src.data_loader import download_stock_data
from src.feature_engineering import add_features
from src.eda import eda_summary
from src.train_models import train_multiple_models
import os
import pandas as pd


def run_pipeline(symbol: str, start_date: str, end_date: str):
    print(f"\n📥 Downloading data for {symbol}...")
    df = download_stock_data(symbol, start_date, end_date)

    print("\n🔧 Performing feature engineering...")
    df = add_features(df)

    print("\n📊 Running EDA...")
    eda_summary(df, stock_name=symbol.replace(".NS", ""))

    print("\n🤖 Training models and evaluating...")
    train_multiple_models(df, stock_name=symbol.replace(".NS", ""))

    # Save the processed DataFrame
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    df.to_pickle(f"{processed_dir}/{symbol.replace('.NS', '')}_v2.pkl")
    print(f"\n✅ Pipeline completed. Processed data saved at: {processed_dir}/{symbol.replace('.NS', '')}_v2.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock prediction pipeline.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock ticker symbol, e.g., RELIANCE.NS")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    run_pipeline(args.symbol, args.start, args.end)
