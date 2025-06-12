### src/data_loader.py
import yfinance as yf
import pandas as pd

def download_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    stock_df = yf.download(symbol, start=start, end=end)
    nifty_df = yf.download("^NSEI", start=start, end=end)  # Nifty50 index

    stock_df.reset_index(inplace=True)
    nifty_df.reset_index(inplace=True)

    # Rename to avoid column name collision
    nifty_df.rename(columns={"Close": "Nifty_Close"}, inplace=True)

    # Merge on Date
    df = pd.merge(stock_df, nifty_df[['Date', 'Nifty_Close']], on="Date", how="left")
    
    # Add exogenous features
    df['Nifty_Returns'] = df['Nifty_Close'].pct_change()
    df['Nifty_Lag_1'] = df['Nifty_Close'].shift(1)
    df['Nifty_MA10'] = df['Nifty_Close'].rolling(window=10).mean()
     
    print("\n🔍 Preview of downloaded DataFrame:")
    print(df.head())
    print("\n📐 DataFrame Columns:", df.columns)
    print("📊 DataFrame Shape:", df.shape)

    return df