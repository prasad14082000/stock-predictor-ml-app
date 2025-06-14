### src/data_loader.py
import yfinance as yf
import pandas as pd

def download_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    stock_df = yf.download(symbol, start=start, end=end)
    nifty_df = yf.download("^NSEI", start=start, end=end)  # Nifty50 index

    # Rename to avoid column name collision
    nifty_df.rename(columns={"Close": "Nifty_Close"}, inplace=True)

    stock_df.columns = stock_df.columns.get_level_values(0)

    nifty_df.columns = nifty_df.columns.get_level_values(0)

    # Merge on Date
    df = pd.merge(stock_df, nifty_df[['Nifty_Close']], on = 'Date')
    
    df.reset_index(inplace=True)

    # Add exogenous features
    df['Nifty_Returns'] = df['Nifty_Close'].pct_change(fill_method=None)
    df['Nifty_Lag1'] = df['Nifty_Close'].shift(1)
    df['Nifty_MA10'] = df['Nifty_Close'].rolling(window=10).mean()
    
    df = df.dropna()

    print("\n🔍 Preview of downloaded DataFrame:")
    print(df.head())
    print("\n📐 DataFrame Columns:", df.columns)
    print("📊 DataFrame Shape:", df.shape)

    return df