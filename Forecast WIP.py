import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.dates import DateFormatter

# Define the path to the CSV file
csv_path = 'jobtech_dataset2022.csv'

# Read the CSV file into a dataframe
df = pd.read_csv(csv_path)

# Convert description column to lowercase
df['description'] = df['description'].str.lower()

# Extract the relevant columns
df = df[['publication_date', 'description', 'occupation_field']]

# Filter by IT jobs
df = df[df['occupation_field'].str.contains('IT')]

# Define the list of programming languages to search for
programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift']

# Choose a language to forecast for
lang = re.escape('java')

# Count the mentions of the selected programming language in the descriptions
df[lang] = df['description'].str.count(lang)


# Group the data by month and sum the language counts
df['publication_date'] = pd.to_datetime(df['publication_date'])
df = df.groupby(pd.Grouper(key='publication_date', freq='M')).sum().reset_index()

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
ax.plot(df['ds'], df['y'], label='Actual')
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