import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.dates import DateFormatter

#Set year IMPORTANT
year = '2020'

# Define the path to the JSON file
json_path = f'{year}_clean.json'

# Read the JSON file into a dataframe
df = pd.read_json(json_path)

# Extract the relevant columns and convert 'description' to string data type
df = df[['publication_date', 'description']].astype({'description': 'string'})

# Define the list of words to search for (case-insensitive)
words = ['maskininlärning', 'informationssäkerhet', 'cloud computing', 'artificiell intelligens', 'cybersäkerhet']

# Loop through each word in the list and generate a separate forecast for each
for lang in words:
    # Escape any special characters in the word strings
    pattern = '|'.join([re.escape(lang.lower())])
    #print(pattern)

    # Find the word in the description (case-insensitive)
    df[lang] = df['description'].str.lower().str.count(pattern)

    # Group the data by month and sum the language counts
    df['publication_date'] = pd.to_datetime(df['publication_date'], infer_datetime_format=True)

    df_lang = df.groupby(pd.Grouper(key='publication_date', freq='M')).sum().reset_index()
    df_lang = df_lang[df_lang['publication_date'].dt.year == int(year)]

    # Rename the columns for Prophet
    df_lang = df_lang.rename(columns={'publication_date': 'ds', lang: 'y'})

    # Train the Prophet model
    model = Prophet()
    model.fit(df_lang)

    # Make a forecast for the next 12 months
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)

    # Plot the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_lang['ds'], df_lang['y'].to_numpy().reshape(-1), label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Language mentions')
    ax.set_title(f'Popularity Forecast for {lang}')
    date_form = DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(date_form)
    plt.show()

    # Save the forecast data to a CSV file for use in Tableau
    forecast.to_csv(f'{lang}_forecast_{year}.csv', index=False)
