# multistep_forecast.py
import os
import pandas as pd
import joblib
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

        history = df.copy()

        predictions = []
        forecast_dates = []
        previous_y_pred = None
        stable_counter = 0

        for step in range(forecast_days):
            # Remove existing engineered columns before reapplying
            for col in history.columns:
                if col not in ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Nifty_Close']:
                    if col not in ['Nifty_Returns', 'Nifty_Lag1', 'Nifty_MA10']:  # keep necessary exogenous
                        history = history.drop(columns=[col], errors='ignore')

            # Re-engineer features
            current_df = add_features(history.copy())
            current_df.dropna(inplace=True)

            if current_df.empty:
                print(f"❌ DataFrame became empty after adding features at step {step + 1}. Aborting.")
                return

            last_row = current_df.iloc[[-1]]

            # Ensure all required features exist
            X_pred = last_row[expected_features]
            y_pred = model.predict(X_pred)[0]

            # Optional: Early stopping if predictions stop changing significantly
            if previous_y_pred is not None and abs(y_pred - previous_y_pred) < 1e-3:
                stable_counter += 1
                if stable_counter >= 5:
                    print("🛑 Stopping early due to prediction convergence.")
                    break
            else:
                stable_counter = 0

            previous_y_pred = y_pred
            
            # Predict next row
            next_date = last_row['Date'].values[0] + pd.Timedelta(days=1)
            new_row = last_row.copy()
            new_row['Close'] = y_pred
            new_row['Date'] = next_date

            # Keep only essential columns to avoid bloating
            base_columns = df.columns
            new_row = new_row.loc[:, new_row.columns.intersection(base_columns)]

            # Append predicted row to history
            history = pd.concat([history, new_row], ignore_index=True)

            # Clean up for consistency
            history = history.drop_duplicates(subset='Date', keep='last')
            history.reset_index(drop=True, inplace=True)

            predictions.append(y_pred)
            forecast_dates.append(next_date)

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted_Close": predictions
        })

        os.makedirs("reports", exist_ok=True)
        out_path = f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_multi_step_forecast.csv"
        forecast_df.to_csv(out_path, index=False)
        print(f"\n✅ Multi-step forecast saved to {out_path}")

    except Exception as e:
        print(f"❌ Error in multi-step forecast: {str(e)}")
