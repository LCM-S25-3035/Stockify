# 1_data_preprocessing.py
import yfinance as yf
import pandas as pd
import ta

def download_data():
    tickers = ['MGA', 'MSFT', 'CVX', 'TSLA', 'VUG', 'MGK']
    df = yf.download(tickers, start="2015-01-01", end="2025-01-01")['Close']
    df = df.dropna()
    return df

def add_technical_indicators(df):
    df_ind = pd.DataFrame(index=df.index)
    for col in df.columns:
        df_ind[f'{col}_rsi'] = ta.momentum.RSIIndicator(df[col], window=14).rsi()
        df_ind[f'{col}_sma'] = ta.trend.SMAIndicator(df[col], window=14).sma_indicator()
        df_ind[f'{col}_roc'] = ta.momentum.ROCIndicator(df[col], window=5).roc()
    df_combined = pd.concat([df, df_ind], axis=1).dropna()
    return df_combined

if __name__ == "__main__":
    df = download_data()
    df_with_indicators = add_technical_indicators(df)
    df_with_indicators.to_csv("financial_data.csv")
    print("✅ Data saved to financial_data.csv")
