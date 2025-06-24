import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from src.feature_engineering import add_features


def forecast_with_lstm(stock_name: str, forecast_days: int = 7, lookback: int = 30):
    try:
        # Load processed data
        data_path = f"C://GITHUB CODES//stock-predictor-ml//data/processed/{stock_name}_v2.pkl"
        if not os.path.exists(data_path):
            print(f"❌ Processed file not found: {data_path}")
            return

        df = pd.read_pickle(data_path)
        df = add_features(df)
        df.dropna(inplace=True)

        # Use only closing price for LSTM
        data = df[['Close']].values

        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Create sequences
        X, y = [], []
        for i in range(lookback, len(data_scaled)):
            X.append(data_scaled[i - lookback:i, 0])
            y.append(data_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=20, batch_size=32, verbose=1)

        # Forecast
        last_sequence = data_scaled[-lookback:]
        forecasted = []

        for _ in range(forecast_days):
            input_seq = last_sequence[-lookback:].reshape(1, lookback, 1)
            pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
            forecasted.append(pred_scaled)
            last_sequence = np.append(last_sequence, pred_scaled)[-lookback:]

        forecasted_prices = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'LSTM_Forecast': forecasted_prices
        })

        os.makedirs("reports", exist_ok=True)
        forecast_df.to_csv(f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_lstm_forecast.csv", index=False)
        print(f"\n✅ LSTM forecast saved to reports/{stock_name}_lstm_forecast.csv")

    except Exception as e:
        print(f"❌ Error during LSTM forecast: {str(e)}")
