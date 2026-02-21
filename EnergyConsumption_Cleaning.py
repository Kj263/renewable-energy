# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:42:03 2025

@author: kjirsten fastabend
Data Cleaning for Energy Consumption dataset by Our World in Data
Available at: https://www.kaggle.com/datasets/whisperingkahuna/energy-consumption-dataset-by-our-world-in-data/data
"""

import pandas as pd

energycons = pd.read_csv("/Kaggle Energy Consumption - Our World in Data/owid-energy-data.csv")

'''
Checking for record format, checking data types
'''

# Checking variable types and ensuring record format
var_types = energycons.dtypes
print(var_types)

initial_rows = len(energycons)
print("initial row count: ", initial_rows)

duplicates = energycons.duplicated().sum()
if duplicates == 0:
    print("No duplicates found.")
else:
    print(f"Found {duplicates} duplicates.")
    
'''
Finding and correcting missing values
'''    

missing_vals = energycons.isnull().sum().sum()
if missing_vals == 0:
    print("No missing values.")
else:
    print(f"Found {missing_vals} missing values prior to alteration")

duplicate_columns = energycons.columns[energycons.columns.duplicated()]
if len(duplicate_columns) == 0:
    print("No duplicate columns found.")
else:
    print(f"Found duplicate columns: {duplicate_columns.tolist()}")
    
# Calculate the proportion of null values for each column
null_proportion = energycons.isnull().mean()

# Identify columns where the proportion of null values exceeds a threshold (e.g., 50%)
threshold = 0.75
columns_with_high_nulls = null_proportion[null_proportion > threshold].index.tolist()

print("Columns with more than 50% null values:")
print(columns_with_high_nulls)

# Reporting and cleaning missing values
energycons.dropna(subset=['renewables_electricity', 'iso_code'], inplace=True)
print("rows dropped: ", initial_rows - len(energycons))

missing_vals = energycons.isnull().sum().sum()
if missing_vals == 0:
    print("No missing values.")
else:
    print(f"Found {missing_vals} missing values after dropping null 'renewables_electricity' and 'iso_code' rows.")
    
# Removing Incorrect Values
if not all(energycons['year'] >= 1750):
    print("Found year values out of expected range.")


'''
Checking Formats and ensuring that formats are correct
'''
def check_year_format(year):
    try:
        if len(str(year)) == 4 and year.isdigit():
            return int(year)
        else:
            raise ValueError
    except ValueError:
        print(f"Incorrect year format found: {year}")
        return None

energycons['year'] = energycons['year'].apply(lambda x: check_year_format(str(x)))
energycons.dropna(subset=['year'], inplace=True)
    
# correcting outlying years identified in visualization
energycons = energycons[(energycons['year'] >= 1965) & (energycons['year'] <= 2024)]

'''
Final Clean Dataset Save (Qualitative & Quantitative)
'''
# Reset the index after removing rows
energycons = energycons.reset_index(drop=True)
energycons.to_csv("EnergyConsumption_mixed_clean.csv", index=False)

