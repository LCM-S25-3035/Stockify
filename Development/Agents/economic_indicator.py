from fredapi import Fred
import pandas as pd

#Connecting to FRED
fred = Fred(api_key="API_KEY")

#Fetching Economic Data from FRED ===
cpi = fred.get_series("CPIAUCSL")           #Consumer Price Index
interest_rate = fred.get_series("FEDFUNDS") #Federal Funds Rate
unemployment = fred.get_series("UNRATE")    #Unemployment Rate
recession = fred.get_series("USRECD")       #NBER Recession Indicator (0 = No Recession, 1 = Recession)

#Converting to DataFrames and Saving
#CPI
cpi_df = cpi.to_frame(name="CPI")
cpi_df.index.name = "Date"
cpi_df.to_csv("../Data/cpi_data.csv")

#Interest Rate
interest_rate_df = interest_rate.to_frame(name="FedFundsRate")
interest_rate_df.index.name = "Date"
interest_rate_df.to_csv("../Data/interest_rate_data.csv")

#Unemployment Rate
unemployment_df = unemployment.to_frame(name="UnemploymentRate")
unemployment_df.index.name = "Date"
unemployment_df.to_csv("../Data/unemployment_data.csv")

#Recession Indicator
recession_df = recession.to_frame(name="RecessionIndicator")
recession_df.index.name = "Date"
recession_df.to_csv("../Data/recession_indicator.csv")

#Previewing the last few rows of each DataFrame
print("Preview CPI:\n", cpi_df.tail(), "\n")
print("Preview Interest Rate:\n", interest_rate_df.tail(), "\n")
print("Preview Unemployment:\n", unemployment_df.tail(), "\n")
print("Preview Recession Indicator:\n", recession_df.tail())
