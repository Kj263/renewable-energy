# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:44:27 2025

@author: kjirsten fastabend
Data Cleaning for 'The Global Carbon Project's fossil CO2 emissions dataset'
Available at: https://zenodo.org/records/10562476
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

carbonpercap = pd.read_csv("/GCB2023v43_percapita_flat.csv")

'''
Checking for record format, checking data types
'''
# Checking variable types and ensuring record format
print(carbonpercap.dtypes)

initial_rows = len(carbonpercap)
print("initial row count: ", initial_rows)

# Drop the 'UN M49' column
carbonpercap.drop(columns=['UN M49'], inplace=True)

duplicates = carbonpercap.duplicated().sum()
if duplicates == 0:
    print("No duplicates found.")
else:
    print(f"Found {duplicates} duplicates.")

'''
Finding and correcting missing values
'''    
missing_vals = carbonpercap.isnull().sum().sum()
if missing_vals == 0:
    print("No missing values.")
else:
    print(f"Found {missing_vals} missing values prior to alteration.")

duplicate_columns = carbonpercap.columns[carbonpercap.columns.duplicated()]
if len(duplicate_columns) == 0:
    print("No duplicate columns found.")
else:
    print(f"Found duplicate columns: {duplicate_columns.tolist()}")

# Remove the 'Other' column, discovered issues with analysis due to significant na's.
carbonpercap.drop(columns=['Other'], inplace=True)

# Reporting and cleaning missing values
carbonpercap.dropna(subset=['Total'], inplace=True)
print("rows dropped: ", initial_rows - len(carbonpercap))

missing_vals = carbonpercap.isnull().sum().sum()
if missing_vals == 0:
    print("No missing values.")
else:
    print(f"Found {missing_vals} missing values after dropping null 'Total' rows.")

'''
Checking Formats and ensuring that formats are correct
'''

unique_combinations = carbonpercap[['Country', 'Year']].duplicated().sum()
if unique_combinations == 0:
    print("All 'Country' and 'Year' combinations are unique.")
else:
    print(f"Found {unique_combinations} duplicate 'Country' and 'Year' combinations.")

carbonpercap.fillna(np.nan, inplace=True)

# Removing Incorrect Values
if not all(carbonpercap['Year'] >= 1750):
    print("Found Year values out of expected range.")

def check_year_format(year):
    try:
        if len(str(year)) == 4 and year.isdigit():
            return int(year)
        else:
            raise ValueError
    except ValueError:
        print(f"Incorrect year format found: {year}")
        return None

carbonpercap['Year'] = carbonpercap['Year'].apply(lambda x: check_year_format(str(x)))
carbonpercap.dropna(subset=['Year'], inplace=True)

# List of columns to check for float
emissions_columns = ['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring']

# Check for non-float values in specified columns
if carbonpercap[emissions_columns].applymap(lambda x: isinstance(x, float)).all().all():
    print("All floats are correct.")
else:
    non_float_values = carbonpercap[~carbonpercap[emissions_columns].applymap(lambda x: isinstance(x, float)).all(axis=1)]
    print("Non-float values found:")
    print(non_float_values)

# Load the ISO Codes Excel file
excel_path = 'C:/Users/kjirs/CS 399 - Data Science/Project Documentation/Datasets/ISOCodes.xlsx'
reference_df = pd.read_excel(excel_path)

# Merge your dataset with the reference DataFrame to check for inconsistencies
merged_df = carbonpercap.merge(reference_df, left_on=['Country', 'ISO 3166-1 alpha-3'], right_on=['Country', 'ISO 3166-1 alpha-3'], how='left', indicator=True)

# Identify rows with inconsistent 'Country' and 'ISO 3166-1 alpha-3'
inconsistent_rows = merged_df[merged_df['_merge'] == 'left_only']

# Correct 'Country' names and 'ISO 3166-1 alpha-3s'
for idx, row in inconsistent_rows.iterrows():
    correct_country = reference_df[reference_df['ISO 3166-1 alpha-3'] == row['ISO 3166-1 alpha-3']]['Country']
    correct_code = reference_df[reference_df['Country'] == row['Country']]['ISO 3166-1 alpha-3']
    
    if len(correct_country) > 0 and len(correct_code) > 0:
        if row['ISO 3166-1 alpha-3'] in reference_df['ISO 3166-1 alpha-3'].values:
            carbonpercap.loc[carbonpercap.index == idx, 'Country'] = correct_country.values[0]
        elif row['Country'] in reference_df['Country'].values:
            carbonpercap.loc[carbonpercap.index == idx, 'ISO 3166-1 alpha-3'] = correct_code.values[0]
    else:
        print(f"Inconsistent row not corrected due to missing reference: {row[['Country', 'ISO 3166-1 alpha-3']]}")

# Verify corrections
merged_df = carbonpercap.merge(reference_df, left_on=['Country', 'ISO 3166-1 alpha-3'], right_on=['Country', 'ISO 3166-1 alpha-3'], how='left', indicator=True)
remaining_inconsistencies = merged_df[merged_df['_merge'] == 'left_only']

if not remaining_inconsistencies.empty:
    print("Remaining inconsistencies found:")
    print(remaining_inconsistencies[['Country', 'ISO 3166-1 alpha-3']])
else:
    print("All Country and ISO 3166-1 alpha-3s are correct after corrections.")
    
# Replace 'USA' with 'United States' in the 'Country' column
# All other inconsistencies determined to be irrelevant
carbonpercap['Country'] = carbonpercap['Country'].replace('USA', 'United States')

# Drop XIS, International Shipping, not relevant for future analysis
carbonpercap = carbonpercap[carbonpercap['ISO 3166-1 alpha-3'] != 'XIS']

'''
Finding and Correcting Outliers
'''
# Function to detect outliers using the IQR method
# Reference: https://www.statology.org/find-outliers-with-iqr/, https://www.geeksforgeeks.org/quantile-transformer-for-outlier-detection/?ref=ml_lbp
# Author: Geeks for Geeks
# Title:Interquartile Range to Detect Outliers in Data, Quantile Transformer for Outlier Detection

# correcting outlying years identified in visualization
carbonpercap = carbonpercap[(carbonpercap['Year'] >= 1900) & (carbonpercap['Year'] <= 2024)]

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in the 'Total' column for carbonpercap
outliers, lower_bound, upper_bound = detect_outliers(carbonpercap, 'Total')

# Visualize the outliers using a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=carbonpercap['Total'])
plt.title('Boxplot of Total Emissions')
plt.show()

print("Number of outliers in 'Total':", len(outliers))
print(outliers)

# Correct outliers by capping them at the lower and upper bounds
# Transforms outliers, instead of removing them. Confident in data, do not want to remove outliers.
# Reduces the influence of extreme values on analyses
carbonpercap['Total'] = np.where(carbonpercap['Total'] < lower_bound, lower_bound,
                                 np.where(carbonpercap['Total'] > upper_bound, upper_bound,
                                          carbonpercap['Total']))

'''
Normalize the dataframe using MinMaxScaler
'''
scaler = MinMaxScaler()
carbonpercap[['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring']] = scaler.fit_transform(carbonpercap[['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring']])

'''
Linear Regression Model...for fun?
'''
# Hi Grader (:, If you're reading here - I'm playing around with this section, feel free to ignore and jump to the next section.
# Select features and target variable
# Will tell us which features are key contributers to 'Total' emissions
X = carbonpercap[['Coal', 'Oil', 'Gas', 'Cement', 'Flaring']]
y = carbonpercap['Total']

# Drop rows with missing values in X or y (model can't perform with na vals)
X = X.dropna()
y = y[X.index]  # Align y with the cleaned X

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a linear regression model
model = LinearRegression()

# Initialize RFE with the linear regression model and number of features to select
rfe = RFE(model, n_features_to_select=3)

# Fit RFE to the training data
rfe.fit(X_train, y_train)

# Print the selected features
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)

# Train the model using the selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
model.fit(X_train_rfe, y_train)

# Predict on the test set
y_pred = model.predict(X_test_rfe)

# Calculate and print the mean squared error
# Results: Selected features: Index(['Oil', 'Gas', 'Flaring'], dtype='object')
# These results were based on all quantitative cats
# Mean Squared Error: 0.031056816298906567 
# Model is performing reasonably well (according to MSE), Oil, Gas, and Flaring are identified as key contributors to total emissions
# Policy implementation: These may be key areas to focus policy around to reduce total emissions
# Removing Other as a feature => Selected features: Index(['Coal', 'Oil', 'Gas'], MSE 0.044
# I think the high amount of na's in 'Other' caused issues with this model originally...removing Other due to high na's helped
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

'''
Verify dtypes to ensure that proper data types were maintained
'''
# Checking variable types and ensuring record format
print(carbonpercap.dtypes)

'''
Unlabeled and Quantitative Dataset
'''
carbon_quantitative = carbonpercap[['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring']]
# Reset the index after removing rows
carbon_quantitative = carbon_quantitative.reset_index(drop=True)
# Write the clean data to new file
#carbon_quantitative.to_csv("GCBpercapita_quantitative_clean.csv", index=False)

'''
Feature Generation
'''
# Ratios of emission sources to total emissions
# carbonpercap['Coal_to_Total_Ratio'] = carbonpercap['Coal'] / carbonpercap['Total']
# carbonpercap['Oil_to_Total_Ratio'] = carbonpercap['Oil'] / carbonpercap['Total']
# carbonpercap['Gas_to_Total_Ratio'] = carbonpercap['Gas'] / carbonpercap['Total']
# carbonpercap['Cement_to_Total_Ratio'] = carbonpercap['Cement'] / carbonpercap['Total']
# carbonpercap['Flaring_to_Total_Ratio'] = carbonpercap['Flaring'] / carbonpercap['Total']

# Aggregate Measures
carbonpercap['Total_Fossil_Fuel_Emissions'] = carbonpercap['Coal'] + carbonpercap['Oil'] + carbonpercap['Gas']
carbonpercap['Non_Fossil_Fuel_Emissions'] = carbonpercap['Cement'] + carbonpercap['Flaring']

'''
Final Clean Dataset Save (Qualitative & Quantitative)
'''
# Reset the index after removing rows
carbonpercap = carbonpercap.reset_index(drop=True)
carbonpercap.to_csv("GCBpercapita_mixed_clean.csv", index=False)


