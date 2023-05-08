import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.dates import DateFormatter

# Define the path to the JSON file
json_path = '2022_clean.json'

# Read the JSON file into a dataframe
df = pd.read_json(json_path)

# Extract the relevant columns and convert 'description' to string data type
df = df[['publication_date', 'description']].astype({'description': 'string'})


# Define the list of programming languages to search for (case-insensitive)
programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift']

# Choose a language to forecast for
lang = 'Python'

# Escape any special characters in the programming language strings
pattern = '|'.join([re.escape(lang.lower()) for lang in programming_languages])

# Find the programming language in the description (case-insensitive)
df[lang] = df['description'].str.lower().str.count(pattern)

# Group the data by month and sum the language counts
df['publication_date'] = pd.to_datetime(df['publication_date'], format='%Y-%m-%d')
df = df.groupby(pd.Grouper(key='publication_date', freq='M')).sum().reset_index()
print(df.head())
print(df.columns)

# Rename the columns for Prophet
df = df.rename(columns={'publication_date': 'ds', lang: 'y'})

# Train the Prophet model
model = Prophet()
model.fit(df)

# Make a forecast for the next 12 months
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot the forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['ds'], df['y'].to_numpy().reshape(-1), label='Actual')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Language mentions')
ax.set_title(f'Popularity Forecast for {lang}')
date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.show()


# Save the forecast data to a CSV file for use in Tableau
forecast.to_csv(f'{lang}_forecast.csv', index=False)
