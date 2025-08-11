#country wise ETFs data collection
import requests
import pandas as pd
import time
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETFDataCollector:
    """Class for collecting and managing ETF data from Alpha Vantage API"""
    
    def __init__(self, api_key: str, data_dir: str = 'ETF_data_global'):
        self.API_KEY = api_key
        self.BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={apikey}&outputsize=full'
        self.DATA_DIR = data_dir
        self.START_DATE = pd.to_datetime('2015-01-01')
        
        # ETF symbols grouped by country
        self.ETF_SYMBOLS = {
            'USA': ['SPY', 'QQQ', 'VTI', 'XLF', 'XLE', 'XLK', 'XLV'],
            'Canada': ['XIC.TO', 'XIU.TO', 'VCN.TO', 'ZEB.TO', 'VEQT.TO'],
            'Europe': ['VGK', 'IEUR'],
            'AsiaPacific': ['VPL', 'AAXJ'],
            'Global': ['VT', 'ACWI']
        }
        
        os.makedirs(self.DATA_DIR, exist_ok=True)

    def get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch daily data for a single symbol from Alpha Vantage"""
        url = self.BASE_URL.format(symbol=symbol, apikey=self.API_KEY)
        try:
            response = requests.get(url)
            response.raise_for_status()
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
                df = df[df.index >= self.START_DATE]
                return df
            else:
                error_msg = data.get('Note') or data.get('Error Message') or 'Unknown error'
                logger.error(f"Error fetching {symbol}: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return None

    def update_country_data(self, country: str, symbols: List[str]) -> bool:
        """Update data for all symbols in a country"""
        filename = os.path.join(self.DATA_DIR, f'{country}_ETFs.csv')
        
        try:
            # Load existing data if available
            existing_df = pd.read_csv(filename, parse_dates=['Date']) if os.path.exists(filename) else pd.DataFrame()
            all_new_data = []

            for symbol in symbols:
                logger.info(f"Fetching data for {symbol} ({country})...")
                df = self.get_daily_data(symbol)
                if df is not None and not df.empty:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df['Ticker'] = symbol
                    df['Country'] = country
                    df.reset_index(inplace=True)
                    
                    # Filter out existing data
                    if not existing_df.empty:
                        df = df[~df.set_index(['Date', 'Ticker']).index.isin(
                            existing_df.set_index(['Date', 'Ticker']).index
                        )]
                    
                    if not df.empty:
                        all_new_data.append(df)
                
                time.sleep(12)  # Respect API rate limits

            if all_new_data:
                new_combined = pd.concat(all_new_data, ignore_index=True)
                updated_df = pd.concat([existing_df, new_combined], ignore_index=True)
                updated_df.sort_values(by=['Date', 'Ticker'], inplace=True)
                updated_df.to_csv(filename, index=False)
                logger.info(f"Updated {country}_ETFs.csv with {len(new_combined)} new rows.")
                return True
            else:
                logger.info(f"No new data to update for {country}.")
                return False
                
        except Exception as e:
            logger.error(f"Error updating data for {country}: {str(e)}")
            return False

    def collect_all_data(self, continuous_mode: bool = False) -> None:
        """Collect data for all countries"""
        while True:
            request_count = 0
            for country, symbols in self.ETF_SYMBOLS.items():
                success = self.update_country_data(country, symbols)
                request_count += len(symbols)

                # Handle rate limits
                if request_count % 5 == 0:
                    logger.info("Sleeping for 65 seconds to stay under rate limits...")
                    time.sleep(65)

            if not continuous_mode:
                break
                
            logger.info("All country ETFs updated. Sleeping for 1 hour...")
            time.sleep(3600)

    def combine_all_data(self) -> pd.DataFrame:
        """Combine all country data into a single DataFrame"""
        all_files = [os.path.join(self.DATA_DIR, f) 
                    for f in os.listdir(self.DATA_DIR) 
                    if f.endswith('.csv')]
        
        if not all_files:
            raise ValueError("No data files found. Run data collection first.")
            
        combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df.sort_values(['Date', 'Country', 'Ticker'], inplace=True)
        return combined_df

def run_data_collection(api_key: str, continuous_mode: bool = False) -> pd.DataFrame:
    """Run the data collection pipeline and return combined data"""
    collector = ETFDataCollector(api_key)
    collector.collect_all_data(continuous_mode)
    return collector.combine_all_data()

if __name__ == "__main__":
    # When run directly
    API_KEY = '####'  # Replace with actual API key
    run_data_collection(API_KEY, continuous_mode=False)