import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.dates import DateFormatter

#Set years IMPORTANT
years = ['2020', '2021', '2022']

# Define the path to the JSON files
json_paths = [f'{year}_clean.json' for year in years]

# Read the JSON files into a dataframe
df = pd.concat((pd.read_json(json_path) for json_path in json_paths))

# Extract the relevant columns and convert 'description' to string data type
df = df[['publication_date', 'description']].astype({'description': 'string'})

# Define the list of words to search for (case-insensitive)
words = ['maskininlärning', 'informationssäkerhet', 'blockchain', 'artificiell intelligens', 'cybersäkerhet', 'molntjänster', 'virtual reality']

# Loop through each word in the list and generate a separate forecast for each
for lang in words:
    # Escape any special characters in the word strings
    pattern = '|'.join([re.escape(lang.lower())])
    
    # Find the word in the description (case-insensitive)
    df[lang] = df['description'].str.lower().str.count(pattern)

    # Group the data by month and sum the language counts
    df['publication_date'] = pd.to_datetime(df['publication_date'], infer_datetime_format=True)

    df_lang = df.groupby(pd.Grouper(key='publication_date', freq='M')).sum().reset_index()

    # Filter data by years
    df_lang = df_lang[df_lang['publication_date'].dt.year.isin([int(year) for year in years])]

    # Rename the columns for Prophet
    df_lang = df_lang.rename(columns={'publication_date': 'ds', lang: 'y'})

    # Train the Prophet model
    model = Prophet()
    model.fit(df_lang)

    # Make a forecast for the next 12 months
    future = model.make_future_dataframe(periods=36, freq='MS')
    forecast = model.predict(future)

    # Plot the forecast
    #fig, ax = plt.subplots(figsize=(12, 6))
    #ax.plot(df_lang['ds'], df_lang['y'].to_numpy().reshape(-1), label='Actual')
    #ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    #ax.legend()
    #ax.set_xlabel('Date')
    #ax.set_ylabel('Language mentions')
    #ax.set_title(f'Popularity Forecast for {lang}')
    #date_form = DateFormatter("%Y-%m")
    #ax.xaxis.set_major_formatter(date_form)
    #plt.show()

    # Merge actual values with forecasted values
    df_lang = pd.merge(df_lang, forecast[['ds', 'yhat']], on='ds', how='left')

    # Save the forecast data to a CSV file for use in Tableau
    forecast.to_csv(f'{lang}_forecast_{years[0]}-{years[-1]}.csv', index=False)
