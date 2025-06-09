from fredapi import Fred
import pandas as pd
import os

#Initializing FRED API
fred = Fred(api_key="api_key")

def fetch_macroeconomic_data(start_date, end_date):
    """
    Fetch common macroeconomic indicators from FRED
    """
    macro_data = pd.DataFrame()
    
    try:
        #Interest Rates
        print("Fetching interest rate data...")
        macro_data['FEDFUNDS'] = fred.get_series('FEDFUNDS', start_date, end_date)  #Federal Funds Rate
        macro_data['DGS10'] = fred.get_series('DGS10', start_date, end_date)        #10-Year Treasury Yield
        
        #Inflation
        print("Fetching inflation data...")
        macro_data['CPIAUCSL'] = fred.get_series('CPIAUCSL', start_date, end_date)  #CPI All Items
        macro_data['PCE'] = fred.get_series('PCE', start_date, end_date)            #Personal Consumption Expenditures
        
        #Economic Activity
        print("Fetching economic activity data...")
        macro_data['UNRATE'] = fred.get_series('UNRATE', start_date, end_date)     #Unemployment Rate
        macro_data['INDPRO'] = fred.get_series('INDPRO', start_date, end_date)      #Industrial Production Index
        macro_data['GDP'] = fred.get_series('GDP', start_date, end_date)            #GDP
        macro_data['USRECD'] = fred.get_series('USRECD', start_date, end_date)      #Recession Indicator
        
        #Market Sentiment
        print("Fetching market sentiment data...")
        macro_data['VIX'] = fred.get_series('VIXCLS', start_date, end_date)         #VIX (alternative source)
        
        #Forward fill missing values (monthly data)
        macro_data = macro_data.ffill()
        
        #Clean up column names for better readability
        macro_data.columns = [
            'Federal Funds Rate', 
            '10-Year Treasury Yield',
            'CPI All Items',
            'Personal Consumption Expenditures',
            'Unemployment Rate',
            'Industrial Production Index',
            'GDP',
            'Recession Indicator',
            'VIX Market Volatility'
        ]
        
        #Ensure the index is named 'Date' and is properly formatted
        macro_data.index.name = 'Date'
        macro_data.index = pd.to_datetime(macro_data.index)
        
        return macro_data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_macroeconomic_data_to_csv(start_date, end_date, filename='macroeconomic_data.csv'):
    """
    Fetch macroeconomic data and save to CSV file with proper date header
    """
    print(f"\nFetching macroeconomic data from {start_date} to {end_date}...")
    data = fetch_macroeconomic_data(start_date, end_date)
    
    if data is not None:
        print("\nData successfully fetched. Summary statistics:")
        print(data.describe())
        
        print("\nSaving data to CSV file...")
        
        #Saving to CSV with date header and proper formatting
        data.to_csv(filename, index=True, index_label='Date')
        
        print(f"\nData successfully saved to {filename}")
        print(f"Number of rows: {len(data)}")
        print("Columns:", ['Date'] + data.columns.tolist())
        
        #Printing preview
        print("\nPreview of the data:")
        print(data.tail())
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    #Setting date range
    start_date = '2010-01-01'
    end_date = '2024-12-31'  #2024 is the latest complete year
    
    #Output filename
    output_file = 'macroeconomic_data_2010_2024.csv'
    
    #Running the data fetching and saving process
    save_macroeconomic_data_to_csv(start_date, end_date, output_file)