#multistep_forecast.py
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

        # Maintain a rolling history window for feature engineering
        history_window = 30
        history = df.iloc[-history_window:].copy()

        predictions = []
        forecast_dates = []

        for step in range(forecast_days):
            current_df = history.copy()
            current_df = add_features(current_df)
            current_df.dropna(inplace=True)

            if len(current_df) == 0:
                print(f"❌ DataFrame became empty after adding features at step {step + 1}. Aborting.")
                return

            last_row = current_df.iloc[[-1]]

            # Use only expected features
            X_pred = last_row[expected_features]
            y_pred = model.predict(X_pred)[0]
            predictions.append(y_pred)

            # Prepare the next row using predicted close
            next_date = last_row['Date'].values[0] + pd.Timedelta(days=1)
            new_row = last_row.copy()
            new_row['Close'] = y_pred
            new_row['Date'] = next_date

            forecast_dates.append(next_date)

            forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted_Close": predictions
            })

        os.makedirs("reports", exist_ok=True)
        forecast_df.to_csv(f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_multi_step_forecast.csv", index=False)
        print(f"\n✅ Multi-step forecast saved to reports/{stock_name}_{model_name}_multi_step_forecast.csv")

    except Exception as e:
        print(f"❌ Error in multi-step forecast: {str(e)}")
