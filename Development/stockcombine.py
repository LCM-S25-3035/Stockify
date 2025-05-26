import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta
import time

# Finnhub API key for news only
API_KEY = 'd0ppmj9r01qgccuakg3gd0ppmj9r01qgccuakg40'

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC']

start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

def fetch_stock_prices_yf(symbol, start, end):
    data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    if data.empty:
        print(f"No price data fetched for {symbol} from yfinance.")
        return None
    data.reset_index(inplace=True)
    data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    return data[['date', 'open', 'high', 'low', 'close', 'volume']]

def fetch_company_news(symbol, start, end, api_key):
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={api_key}'
    response = requests.get(url)
    news = response.json()
    if not isinstance(news, list):
        print(f"[ERROR] Failed to fetch news for {symbol}: {news}")
        return None
    news_data = []
    for item in news:
        news_data.append({
            'datetime': datetime.fromtimestamp(item['datetime']),
            'headline': item['headline'],
            'source': item['source'],
            'url': item['url']
        })
    df_news = pd.DataFrame(news_data)
    return df_news

all_prices = []
all_news = []

for symbol in symbols:
    print(f"\n=== Processing {symbol} ===")
    
    # Get price data from yfinance
    prices_df = fetch_stock_prices_yf(symbol, start_date, end_date)
    if prices_df is not None:
        prices_df['symbol'] = symbol
        all_prices.append(prices_df)
    else:
        print(f"Price data not found for {symbol}")

    # Get news from Finnhub
    news_df = fetch_company_news(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), API_KEY)
    if news_df is not None and not news_df.empty:
        news_df['symbol'] = symbol
        all_news.append(news_df)
    else:
        print(f"News data not found for {symbol}")

    time.sleep(1)  # To respect API limits

# Combine and save all data
if all_prices:
    combined_prices_df = pd.concat(all_prices, ignore_index=True)
    combined_prices_df.to_csv('combined_prices.csv', index=False)
    print("Saved combined_prices.csv")

if all_news:
    combined_news_df = pd.concat(all_news, ignore_index=True)
    combined_news_df.to_csv('combined_news.csv', index=False)
    print("Saved combined_news.csv")
