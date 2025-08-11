import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETFCleaner:
    """Class for cleaning and validating ETF data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.cleaned_file_path = 'cleaned_global_ETFs.csv'
        
        # Configure plotting style
        sns.set(style="whitegrid")
        
    def load_data(self) -> None:
        """Load the raw ETF data"""
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def get_summary_stats(self) -> Dict:
        """Get summary statistics before cleaning"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return {
            "initial_shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicates": {
                "full_duplicates": self.df.duplicated().sum(),
                "partial_duplicates": self.df.duplicated(subset=['Date', 'Ticker', 'Country']).sum()
            },
            "non_null_counts": self.df.count().to_dict()
        }

    def clean_data(self) -> None:
        """Perform all cleaning operations"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Starting data cleaning process...")
        
        # 1. Remove duplicates
        self._remove_duplicates()
        
        # 2. Convert and validate dates
        self._clean_dates()
        
        # 3. Validate price and volume data
        self._validate_price_data()
        
        # 4. Handle outliers
        self._handle_outliers()
        
        logger.info("Data cleaning completed successfully.")

    def _remove_duplicates(self) -> None:
        """Remove duplicate rows"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['Date', 'Ticker', 'Country'])
        removed_count = initial_count - len(self.df)
        logger.info(f"Removed {removed_count} duplicate rows")

    def _clean_dates(self) -> None:
        """Convert and validate date column"""
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        invalid_dates = self.df['Date'].isna()
        if invalid_dates.any():
            logger.warning(f"Found {invalid_dates.sum()} rows with invalid dates - removing them")
            self.df = self.df[~invalid_dates]

    def _validate_price_data(self) -> None:
        """Validate price and volume columns"""
        # Remove rows with invalid price relationships
        invalid_open = self.df['Open'] > self.df['High']
        invalid_low = self.df['Low'] > self.df['High']
        invalid_volume = self.df['Volume'] < 0
        
        invalid_rows = invalid_open | invalid_low | invalid_volume
        if invalid_rows.any():
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid price/volume data - removing them")
            self.df = self.df[~invalid_rows]

    def _handle_outliers(self, method: str = 'cap') -> None:
        """Handle outliers in numeric columns"""
        logger.info("Handling outliers...")
        
        for col in self.numeric_cols:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                # Cap outliers at bounds
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            elif method == 'remove':
                # Remove outliers
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                self.df = self.df[~outliers]
            else:
                raise ValueError("Invalid outlier handling method. Use 'cap' or 'remove'")

    def generate_plots(self, save_path: str = None) -> None:
        """Generate diagnostic plots"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Generating diagnostic plots...")
        
        # Create boxplots for numeric columns
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_cols):
            if col in self.df.columns:
                sns.boxplot(x=self.df[col], ax=axes[i], color='skyblue', fliersize=2)
                axes[i].set_title(f'Boxplot of {col}')
                axes[i].set_xlabel('')
        
        # Remove empty subplot if odd number of plots
        if len(self.numeric_cols) % 2 != 0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plots to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def save_cleaned_data(self, file_path: str = None) -> str:
        """Save cleaned data to CSV"""
        if self.df is None:
            raise ValueError("No data to save. Load and clean data first.")
            
        output_path = file_path or self.cleaned_file_path
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
        return output_path

    def run_full_clean(self, generate_plots: bool = True) -> pd.DataFrame:
        """Run the full cleaning pipeline"""
        self.load_data()
        self.clean_data()
        
        if generate_plots:
            self.generate_plots('data_quality_plots.png')
            
        self.save_cleaned_data()
        return self.df

def clean_etf_data(file_path: str) -> pd.DataFrame:
    """Convenience function to clean ETF data"""
    cleaner = ETFCleaner(file_path)
    return cleaner.run_full_clean()

if __name__ == "__main__":
    # Example usage when run directly
    input_file = 'Combined_global_ETFs.csv'
    output_file = 'cleaned_global_ETFs.csv'
    
    cleaner = ETFCleaner(input_file)
    cleaned_data = cleaner.run_full_clean()
    
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")# data_cleaning.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETFCleaner:
    """Class for cleaning and validating ETF data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.cleaned_file_path = 'cleaned_global_ETFs.csv'
        
        # Configure plotting style
        sns.set(style="whitegrid")
        
    def load_data(self) -> None:
        """Load the raw ETF data"""
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def get_summary_stats(self) -> Dict:
        """Get summary statistics before cleaning"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return {
            "initial_shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicates": {
                "full_duplicates": self.df.duplicated().sum(),
                "partial_duplicates": self.df.duplicated(subset=['Date', 'Ticker', 'Country']).sum()
            },
            "non_null_counts": self.df.count().to_dict()
        }

    def clean_data(self) -> None:
        """Perform all cleaning operations"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Starting data cleaning process...")
        
        # 1. Remove duplicates
        self._remove_duplicates()
        
        # 2. Convert and validate dates
        self._clean_dates()
        
        # 3. Validate price and volume data
        self._validate_price_data()
        
        # 4. Handle outliers
        self._handle_outliers()
        
        logger.info("Data cleaning completed successfully.")

    def _remove_duplicates(self) -> None:
        """Remove duplicate rows"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['Date', 'Ticker', 'Country'])
        removed_count = initial_count - len(self.df)
        logger.info(f"Removed {removed_count} duplicate rows")

    def _clean_dates(self) -> None:
        """Convert and validate date column"""
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        invalid_dates = self.df['Date'].isna()
        if invalid_dates.any():
            logger.warning(f"Found {invalid_dates.sum()} rows with invalid dates - removing them")
            self.df = self.df[~invalid_dates]

    def _validate_price_data(self) -> None:
        """Validate price and volume columns"""
        # Remove rows with invalid price relationships
        invalid_open = self.df['Open'] > self.df['High']
        invalid_low = self.df['Low'] > self.df['High']
        invalid_volume = self.df['Volume'] < 0
        
        invalid_rows = invalid_open | invalid_low | invalid_volume
        if invalid_rows.any():
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid price/volume data - removing them")
            self.df = self.df[~invalid_rows]

    def _handle_outliers(self, method: str = 'cap') -> None:
        """Handle outliers in numeric columns"""
        logger.info("Handling outliers...")
        
        for col in self.numeric_cols:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                # Cap outliers at bounds
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            elif method == 'remove':
                # Remove outliers
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                self.df = self.df[~outliers]
            else:
                raise ValueError("Invalid outlier handling method. Use 'cap' or 'remove'")

    def generate_plots(self, save_path: str = None) -> None:
        """Generate diagnostic plots"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Generating diagnostic plots...")
        
        # Create boxplots for numeric columns
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_cols):
            if col in self.df.columns:
                sns.boxplot(x=self.df[col], ax=axes[i], color='skyblue', fliersize=2)
                axes[i].set_title(f'Boxplot of {col}')
                axes[i].set_xlabel('')
        
        # Remove empty subplot if odd number of plots
        if len(self.numeric_cols) % 2 != 0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plots to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def save_cleaned_data(self, file_path: str = None) -> str:
        """Save cleaned data to CSV"""
        if self.df is None:
            raise ValueError("No data to save. Load and clean data first.")
            
        output_path = file_path or self.cleaned_file_path
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
        return output_path

    def run_full_clean(self, generate_plots: bool = True) -> pd.DataFrame:
        """Run the full cleaning pipeline"""
        self.load_data()
        self.clean_data()
        
        if generate_plots:
            self.generate_plots('data_quality_plots.png')
            
        self.save_cleaned_data()
        return self.df

def clean_etf_data(file_path: str) -> pd.DataFrame:
    """Convenience function to clean ETF data"""
    cleaner = ETFCleaner(file_path)
    return cleaner.run_full_clean()

if __name__ == "__main__":
    # Example usage when run directly
    input_file = 'Combined_global_ETFs.csv'
    output_file = 'cleaned_global_ETFs.csv'
    
    cleaner = ETFCleaner(input_file)
    cleaned_data = cleaner.run_full_clean()
    
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")