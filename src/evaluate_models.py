import os
import pandas as pd
import joblib
from src.feature_engineering import add_features
from src.eda import select_features_by_correlation
from dateutil.relativedelta import relativedelta
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def evaluate_all_models(stock_name: str, lookback_days: int = 30):
    try:
        df_path = f"data/processed/{stock_name}_v2.pkl"
        if not os.path.exists(df_path):
            print(f" Processed file not found: {df_path}")
            return

        df = pd.read_pickle(df_path)
        df.dropna(inplace=True)

        cutoff_date = df["Date"].max() - pd.Timedelta(days=lookback_days)
        recent_data = df[df["Date"] > cutoff_date].copy()

        if recent_data.empty:
            print(f" No recent data available in the last {lookback_days} days.")
            return

        enriched = recent_data.copy()
        all_features_df = enriched.drop(columns=['Date', 'Close'])

        essential_features = ['Lag_1', 'MA10', 'RSI', 'momentum', 'volatility', 'Daily Returns', 'Nifty_Returns', 'Nifty_Lag1', 'Nifty_MA10', 'Nifty_Close', 'MACD', 'Price Range', 'StdDev10']
        selected_columns = select_features_by_correlation(all_features_df, always_keep=essential_features)
        print(f" Features selected by correlation: {selected_columns}")

        model_dir = "models"
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        results = []

        for filename in os.listdir(model_dir):
            if not filename.lower().startswith(stock_name.lower()):
                continue

            model_name = filename.replace(f"{stock_name}_", "").replace(".pkl", "")
            model_path = os.path.join(model_dir, filename)
            print(f" Evaluating model file: {filename}")

            try:
                model_bundle = joblib.load(model_path)
                model, expected_features = model_bundle
                 
                available_features = [f for f in expected_features if f in selected_columns]
                if not available_features:
                    print(f" No usable features for {model_name}, skipping.")
                    continue
                
                X = enriched[available_features]
                y = enriched['Close']

                y_pred = model.predict(X)
                rmse = root_mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                results.append({
                    "Model": model_name,
                    "Samples": X.shape[0],
                    "RMSE": round(rmse, 2),
                    "MAE": round(mae, 2),
                    "R²": round(r2, 4)
                })

            except Exception as e:
                print(f" Skipped {model_name} due to error: {str(e)}")

        if results:
            result_df = pd.DataFrame(results)
            csv_path = f"{report_dir}/{stock_name}_evaluation.csv"
            result_df.to_csv(csv_path, index=False)
            print(f"\n Model evaluation complete. Results saved to {csv_path}")
        else:
            print(" No models were successfully evaluated.")

    except Exception as e:
        print(f" Unexpected error during model evaluation: {str(e)}")
