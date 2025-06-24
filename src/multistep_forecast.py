import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.feature_engineering import add_features

def forecast_multi_step(stock_name: str, model_name: str, forecast_days: int = 7):
    try:
        df_path = f"C://GITHUB CODES//stock-predictor-ml//data/processed/{stock_name}_v2.pkl"
        model_path = f"C://GITHUB CODES//stock-predictor-ml//models/{stock_name}_{model_name}.pkl"

        if not os.path.exists(df_path):
            print(f"❌ Processed file not found: {df_path}")
            return

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return

        df = pd.read_pickle(df_path)
        df = add_features(df)
        df.dropna(inplace=True)

        model_bundle = joblib.load(model_path)
        model, expected_features = model_bundle

        last_row = df.iloc[-1:].copy()
        predictions = []

        for step in range(forecast_days):
            # Use only expected features
            X_pred = last_row[expected_features]
            y_pred = model.predict(X_pred)[0]
            predictions.append(y_pred)

            # Create new row with updated 'Close' value as prediction
            new_row = last_row.copy()
            new_row['Close'] = y_pred
            new_row['Date'] = new_row['Date'] + pd.Timedelta(days=1)

            # Append and re-engineer features
            df = pd.concat([df, new_row], ignore_index=True)
            df = add_features(df)
            df.dropna(inplace=True)
            last_row = df.iloc[[-1]]

        forecast_df = pd.DataFrame({
            "Date": pd.date_range(start=df['Date'].max() - pd.Timedelta(days=forecast_days - 1), periods=forecast_days),
            "Forecasted_Close": predictions
        })

        os.makedirs("reports", exist_ok=True)
        forecast_df.to_csv(f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_multi_step_forecast.csv", index=False)
        print(f"\n✅ Multi-step forecast saved to reports/{stock_name}_{model_name}_multi_step_forecast.csv")

    except Exception as e:
        print(f"❌ Error in multi-step forecast: {str(e)}")
