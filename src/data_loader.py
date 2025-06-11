### src/data_loader.py
import yfinance as yf
import pandas as pd

def download_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df


