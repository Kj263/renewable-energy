# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:16:00 2025

@author: kjirsten fastabend
Data Visualization for 'Energy Consumption Dataset by Our World in Data'
available at: https://www.kaggle.com/datasets/whisperingkahuna/energy-consumption-dataset-by-our-world-in-data/data
"""

import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


energycons = pd.read_csv("/Kaggle Energy Consumption - Our World in Data/owid-energy-data.csv")

# Select numeric columns
numeric_columns = energycons.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms in a grid layout
num_plots = len(numeric_columns)
num_cols = 4  # Number of columns in the grid
num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
axes = axes.flatten()

for i, column in enumerate(numeric_columns):
    sns.histplot(energycons[column], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.savefig('Histograms_of_Quantitative_Variables.png')
plt.show()

# Plot box plots in a grid layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
axes = axes.flatten()

for i, column in enumerate(numeric_columns):
    sns.boxplot(x=energycons[column], ax=axes[i])
    axes[i].set_title(f'Box Plot of {column}')
    axes[i].set_xlabel(column)

# Adjust layout
plt.tight_layout()
plt.savefig('Box_Plots_of_Quantitative_Variables.png')
plt.show()

# Time Series
# Reshape the data for plotting
energycons_long = pd.melt(energycons, id_vars=['year', 'country'], value_vars=energycons.select_dtypes(include=['float64', 'int64']).columns, 
                          var_name='Variable', value_name='Value')

# Create the interactive time series plot for multiple countries
fig = px.line(energycons_long, x='year', y='Value', color='Variable', facet_col='country', facet_col_wrap=5,
              title='Interactive Time Series Line Plots by Country')

# Save the interactive plot as an HTML file
fig.write_html('Interactive_Time_Series_Plot.html')

# Display the plot in Spyder
import webbrowser
webbrowser.open('Interactive_Time_Series_Plot.html')




