#src/forecast.py
import pandas as pd
import os
import numpy as np
import joblib
import datetime
from dateutil.relativedelta import relativedelta
from src.feature_engineering import add_features
from src.eda import select_features_by_correlation
import matplotlib.pyplot as plt

def forecast_next_days(stock_name: str, model_name: str, days_ahead: int = 10):
    '''
    Forecast the next N days of closing prices using the best model
    '''
    model_path = f"C://GITHUB CODES//stock-predictor-ml//models/{stock_name}_{model_name}.pkl"
    data_path = f"C://GITHUB CODES//stock-predictor-ml//data//processed/{stock_name}_v2.pkl"

    # → Check if the processed data file exists
    if not os.path.exists(data_path):
        print(f"❌ Forecast aborted: processed data file not found at {data_path}")
        return

    # → Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Forecast aborted: model file not found at {model_path}")
        return
    
    # Load model and features used during training
    model_bundle = joblib.load(model_path)
    if isinstance(model_bundle, tuple):
        model, expected_features = model_bundle
    else:
        print("❌ Model bundle is not a tuple. Cannot extract features. Aborting.")
        return    
    
    df = pd.read_pickle(data_path)
    df.sort_values('Date',  inplace = True)

    if 'Nifty_Close' in df.columns:
        df['Nifty_Returns'] = df['Nifty_Close'].pct_change(fill_method=None)
        df['Nifty_Lag1'] = df['Nifty_Close'].shift(1)
        df['Nifty_MA10'] = df['Nifty_Close'].rolling(window=10).mean()
    else:
        print("⚠️ 'Nifty_Close' not found. Skipping Nifty-related feature creation.")

    future_predictions = []
    
    # Start with the last available data
    forecast_data = df.copy()

    for _ in range(days_ahead):
        # Recalculate all features after appending previous prediction
        enriched = add_features(forecast_data.copy())
        enriched.dropna(inplace=True)

        # Define essential features to always retain
        essential_features = ['Lag_1', 'MA10', 'RSI', 'momentum', 'volatitlity', 'Daily Returns', 'Day', 'Weekday', 'Month', 'Nifty_Close', 'Nifty_Returns', 'Nifty_Lag1', 'Nifty_MA10', 'StdDev10']

        # Select features by correlation, but preserve essential features
        all_features_df = enriched.drop(columns=['Date', 'Close'])
        selected_columns = select_features_by_correlation(all_features_df, always_keep=essential_features)

        # Align selected features with expected ones from training
        available_features = [f for f in expected_features if f in selected_columns]
        feature_input = enriched[available_features].iloc[[-1]]

        # Safety check
        missing_features = set(expected_features) - set(feature_input.columns)
        if missing_features:
            print(f"❌ Forecast aborted: missing features — {missing_features}")
            return

        # Predict next value
        y_pred = model.predict(feature_input)[0]

        # Predict next date
        next_date = forecast_data['Date'].max() + relativedelta(days=1)

        # Step 1: Start with a real copy to preserve feature space
        latest = forecast_data.iloc[[-1]].copy()
        latest['Date'] = next_date
        latest['Close'] = y_pred  # Update close

        forecast_data = pd.concat([forecast_data, latest], ignore_index=True)

        future_predictions.append((next_date, y_pred))

    future_df = pd.DataFrame(future_predictions, columns = ['Date', 'Forecast'])
    os.makedirs("reports", exist_ok=True)
    future_csv_path = f'C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_forecast.csv'
    future_plot_path = f'C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_forecast_plot.png'
    future_df.to_csv(future_csv_path, index=False)

    # Plotting forecast results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Historical')
    plt.plot(future_df['Date'], future_df['Forecast'], label='Forecast', linestyle='--', marker='o')
    plt.title(f"{stock_name} - {model_name} Forecast for {days_ahead} Days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"reports/{stock_name}_{model_name}_forecast_plot.png")
    plt.close()

    print(f"\U0001F4C8 Forecast complete! Results saved to reports/{stock_name}_{model_name}_forecast.csv")
