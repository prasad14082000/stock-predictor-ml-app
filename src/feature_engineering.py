### src/feature_engineering.py
import pandas as pd
import numpy as np

def add_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday

    df = pd.get_dummies(df, columns= ['Day', 'Month', 'Weekday'], drop_first = True)

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Daily Returns'] = df['Close'].pct_change()
    df['Price Range'] = df['High'] - df['Low']
    df['Cumulative Return'] = (1 + df['Daily Returns']).cumprod()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['StdDev5'] = df['Close'].rolling(window=5).std()
    df['StdDev10'] = df['Close'].rolling(window=10).std()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_10'] = df['Close'].shift(10)
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Mid'] = rolling_mean
    df['Bollinger_Upper'] = rolling_mean + (2 * rolling_std)
    df['Bollinger_Lower'] = rolling_mean - (2 * rolling_std)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Calculate the average gain and loss using exponential moving average (EMA is smoother than SMA)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    # Compute RSI
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['momentum'] = df['Close'] - df['Close'].shift(3)
    df['volatility'] = df['Close'].rolling(window=7).std()

    # --- Exogenous Nifty Features ---
    if 'Nifty_Close' in df.columns:
        df['Nifty_Returns'] = df['Nifty_Close'].pct_change()
        df['Nifty_Lag_1'] = df['Nifty_Close'].shift(1)
        df['Nifty_MA10'] = df['Nifty_Close'].rolling(window=10).mean()

    df = add_time_feature(df)

    # Drop NA only if RSI exists
    if 'RSI' in df.columns:
        df.dropna(subset=['RSI'], inplace=True)
    df.dropna(inplace=True)
    return df