# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:52:58 2025

@author: kjirsten fastabend
Data Visualization for 'The Global Carbon Project's fossil CO2 emissions dataset'
Available at: https://zenodo.org/records/10562476
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

# Set the default renderer to 'browser', wasn't populating otherwise
pio.renderers.default = 'browser'

carbonpercap = pd.read_csv("/GCBpercapita/GCBpercapita_mixed_clean.csv")

"""
Heatmap for Carbon Emissions Per Capita for each variable (Time Inclusive)
"""
# List of variables to visualize
variables = ['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Total_Fossil_Fuel_Emissions', 'Non_Fossil_Fuel_Emissions']

# Create a pivot table and plot heatmaps for each variable
for var in variables:
    pivot_table = carbonpercap.pivot_table(index='Country', columns='Year', values=var)
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap="YlGnBu", linewidths=.5)
    plt.title(f'Per Capita Carbon Emissions ({var}) Over Time by Country')
    plt.xlabel('Year')
    plt.ylabel('Country')
    # Save the plot
    plt.savefig(f'Per_Capita_Emissions_Heatmap_{var}_AfterCleaning.png')
    plt.close()

"""
Line Chart for (Total) Carbon Emissions Per Capita (Time Inclusive)
"""
# Filter data for years 1960 and up
filtered_data = carbonpercap[carbonpercap['Year'] >= 1960]
                             
# Create the interactive line chart
fig = px.line(filtered_data, x='Year', y='Total', color='Country',
              title='Per Capita Carbon Emissions Over Time for All Countries',
              labels={'Total': 'Total Emissions per Capita (tCO₂/capita)', 'Year': 'Year', 'Country': 'Country'})

# Customize the layout
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Total Emissions per Capita (tCO₂/capita)',
    legend_title='Country',
    template='plotly_white'
)

# Save the interactive linechart as an HTML file
fig.write_html('Per_Capita_Total_Emissions_Linechart_AfterCleaning.html')
               
"""
Bar Chart for (Total) Carbon Emissions Per Capita (Time Specific)
"""
# List of years to visualize
years = [1800, 1900, 1950, 1970, 1980, 2000, 2010, 2020]

# Bar Chart for Total Carbon Emissions by Year
for year in years:
    # Filter data for the specific year
    data_year = carbonpercap[carbonpercap['Year'] == year]
    
    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(data_year['Country'], data_year['Total'], color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Total Emissions per Capita (tCO₂/capita)')
    plt.title(f'Total Carbon Emissions per Capita by Country in {year}')
    plt.xticks(rotation=90)
    plt.savefig(f'Per_Capita_Emissions_BarChart_{year}_AfterCleaning.png')
    plt.close()

