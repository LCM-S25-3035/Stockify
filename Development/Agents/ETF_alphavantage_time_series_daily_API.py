#country wise ETFs data collection
import requests
import pandas as pd
import time
import os

API_KEY = '####'  # Used own API for data retrival
START_DATE = pd.to_datetime('2015-01-01')
# Using Time series daily data from AlphaVantage API 
BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={apikey}&outputsize=full'
DATA_DIR = 'ETF_data_global'
os.makedirs(DATA_DIR, exist_ok=True)

# ETF symbols grouped by country
ETF_SYMBOLS = {
    'USA': ['SPY', 'QQQ', 'VTI', 'XLF', 'XLE', 'XLK', 'XLV'],
    'Canada': ['XIC.TO', 'XIU.TO', 'VCN.TO', 'ZEB.TO', 'VEQT.TO'],
    'Europe': ['VGK', 'IEUR'],
    'AsiaPacific': ['VPL', 'AAXJ'],
    'Global': ['VT', 'ACWI']
}

def get_daily_data(symbol, api_key):
    url = BASE_URL.format(symbol=symbol, apikey=api_key)
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df = df.sort_index()
        df = df[df.index >= START_DATE]
        return df
    else:
        print(f"Error fetching {symbol}: {data.get('Note') or data.get('Error Message') or 'Unknown error'}")
        return pd.DataFrame()

def update_country_data(country, symbols, api_key):
    filename = os.path.join(DATA_DIR, f'{country}_ETFs.csv')

    # To load existing file if available
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, parse_dates=['Date'])
    else:
        existing_df = pd.DataFrame()

    all_new_data = []

    for symbol in symbols:
        print(f"Fetching data for {symbol} ({country})...")
        df = get_daily_data(symbol, api_key)
        if not df.empty:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df['Ticker'] = symbol
            df['Country'] = country
            df.reset_index(inplace=True)
            if not existing_df.empty:
                df = df[~df.set_index(['Date', 'Ticker']).index.isin(
                    existing_df.set_index(['Date', 'Ticker']).index
                )]
            if not df.empty:
                all_new_data.append(df)
        time.sleep(12)  # To stay within rate limit

    if all_new_data:
        new_combined = pd.concat(all_new_data, ignore_index=True)
        updated_df = pd.concat([existing_df, new_combined], ignore_index=True)
        updated_df.sort_values(by=['Date', 'Ticker'], inplace=True)
        updated_df.to_csv(filename, index=False)
        print(f"Updated {country}_ETFs.csv with {len(new_combined)} new rows.")
    else:
        print(f"No new data to update for {country}.")

# To run every hour (or we can remove loop for one-time run)
while True:
    request_count = 0
    for country, symbols in ETF_SYMBOLS.items():
        update_country_data(country, symbols, API_KEY)
        request_count += len(symbols)

        # Handle Alpha Vantage free plan limits
        if request_count % 5 == 0:
            print("Sleeping for 65 seconds to stay under rate limits...")
            time.sleep(65)

    print("All country ETFs updated. Sleeping for 1 hour...\n")
    time.sleep(3600)
