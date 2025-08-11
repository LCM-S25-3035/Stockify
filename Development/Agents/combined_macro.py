import pandas as pd
from datetime import datetime

def combine_economic_data(cpi_file, interest_file, recession_file, unemployment_file, output_file):
   
    try:
        #Dictionary to store all dataframes
        dfs = {
            'CPI': None,
            'FedFundsRate': None,
            'RecessionIndicator': None,
            'UnemploymentRate': None
        }
        
        #Reading and process each file
        file_info = [
            (cpi_file, 'CPI'),
            (interest_file, 'FedFundsRate'),
            (recession_file, 'RecessionIndicator'),
            (unemployment_file, 'UnemploymentRate')
        ]
        
        for file_path, col_name in file_info:
            try:
                print(f"Processing {file_path}...")
                
                #Reading CSV with date parsing
                df = pd.read_csv(file_path, parse_dates=True)
                
                #Finding date column (case insensitive)
                date_col = [col for col in df.columns if 'date' in col.lower()]
                if not date_col:
                    raise ValueError(f"No date column found in {file_path}")
                
                #Keeping only date and value columns
                df = df[[date_col[0], df.columns[-1]]]
                df.columns = ['Date', col_name]
                
                #Converting Date column to datetime format
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                #Dropping rows with invalid dates
                invalid_dates = df['Date'].isna()
                if invalid_dates.any():
                    print(f"Warning: Dropped {invalid_dates.sum()} rows with invalid dates from {file_path}")
                    df = df[~invalid_dates]
                
                #Filtering dates from 1950-01-01 onwards
                df = df[df['Date'] >= pd.to_datetime('1960-01-01')]
                
                #Setting Date as index after validation
                dfs[col_name] = df.set_index('Date')
                
                print(f"Successfully processed {file_path} with {len(df)} records")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                raise
        
        #Combining all data with outer join
        print("\nMerging all datasets...")
        combined_df = pd.concat(dfs.values(), axis=1)
        
        #Sorting by date and reset index
        combined_df = combined_df.sort_index().reset_index()
        
        #Forwarding fill missing values for certain columns
        fill_cols = ['CPI', 'FedFundsRate', 'UnemploymentRate']
        for col in fill_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].ffill()
        
        #Converting Date column to consistent format before saving
        combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.strftime('%Y-%m-%d')
        
        #Saving to CSV
        combined_df.to_csv(output_file, index=False)
        
        #Printing comprehensive summary
        print(f"\nSuccessfully created combined file: {output_file}")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        
        print("\nData completeness:")
        for col in combined_df.columns[1:]:
            non_null = combined_df[col].count()
            total = len(combined_df)
            print(f"- {col}: {non_null}/{total} ({non_null/total:.1%} filled)")
        
        print("\nFirst 5 rows:")
        print(combined_df.head())
        
        return combined_df
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

if __name__ == "__main__":
    #Predefined file paths
    input_files = {
        'cpi_file': r"D:\Big Data Analytics\Term 3\Capstone Project\stockify\Data\cpi_data.csv",
        'interest_file': r"D:\Big Data Analytics\Term 3\Capstone Project\stockify\Data\interest_rate_data.csv",
        'recession_file': r"D:\Big Data Analytics\Term 3\Capstone Project\stockify\Data\recession_indicator.csv",
        'unemployment_file': r"D:\Big Data Analytics\Term 3\Capstone Project\stockify\Data\unemployment_data.csv"
    }
    
    output_file = r"combined_economic_data.csv"
    
    #Runing the combiner
    result = combine_economic_data(
        input_files['cpi_file'],
        input_files['interest_file'],
        input_files['recession_file'],
        input_files['unemployment_file'],
        output_file
    )