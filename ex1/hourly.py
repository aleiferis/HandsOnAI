import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic hourly data for 3 years
hours = 365 * 3 * 24  # 3 years of data, hourly frequency
date_rng = pd.date_range(start='2020-01-01', periods=hours, freq='h')

# Time features
day_of_year = date_rng.dayofyear
hour_of_day = date_rng.hour

# Simulate urban mobility (e.g., hourly ridership)
daily_pattern = 5000 + 3000 * np.sin(2 * 2 * np.pi * hour_of_day / 24 - 2 * np.pi * 8/24)  # More during rush hours
seasonal_pattern = 2000 * np.sin(2 * np.pi * day_of_year / 365)
mobility_noise = np.random.normal(scale=1000, size=hours)
mobility = 10000 + daily_pattern + seasonal_pattern + mobility_noise

# Simulate temperature (seasonal + daily cycle)
temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + 3 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(scale=1, size=hours)

# Simulate precipitation
precipitation = np.random.gamma(shape=1.5, scale=1.5, size=hours)
precipitation[date_rng.month.isin([6, 7, 8])] *= 0.2  # Less in summer

# Simulate additional factors
green_initiatives = np.random.choice([0, 1], size=hours, p=[0.99, 0.01])  # Less frequent per hour

# Inject skewed traffic congestion values using a log-normal distribution
mu = 0.5   # mean of the log of the variable
sigma = 0.75  # standard deviation (controls skewness)
size = hours
traffic_congestion = np.random.lognormal(mean=mu, sigma=sigma, size=size)

bike_lane_availability = np.random.choice([0, 1], size=hours, p=[0.95, 0.05])
fuel_prices = 2.5 + np.random.normal(0, 0.1, size=hours)
public_transport_fares = 1.5 + np.random.normal(0, 0.05, size=hours)
event_impact = np.random.choice([0, 1], size=hours, p=[0.99, 0.01])
school_open = np.where(date_rng.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5, 6]), 1, 0)
remote_work_trends = np.random.uniform(30, 40, size=hours)
tourism_trends = np.random.uniform(0, 100, size=hours) + 30 * np.sin(2 * np.pi * day_of_year / 365)
public_transport_reliability = np.random.uniform(0, 100, size=hours)
extreme_weather_events = np.random.choice([0, 1], size=hours, p=[0.995, 0.005])
road_closures_construction = np.random.choice([0, 1], size=hours, p=[0.98, 0.02])
average_income_levels = np.random.normal(50000, 5000, size=hours)

# Create DataFrame
df = pd.DataFrame({'ds': date_rng, 'y': mobility, 'temp': temp, 'precip': precipitation,
                   'green_initiatives': green_initiatives, 'traffic_congestion': traffic_congestion,
                   'bike_lane_availability': bike_lane_availability, 'fuel_prices': fuel_prices,
                   'public_transport_fares': public_transport_fares, 'event_impact': event_impact,
                   'school_open': school_open, 'remote_work_trends': remote_work_trends,
                   'tourism_trends': tourism_trends, 'public_transport_reliability': public_transport_reliability,
                   'extreme_weather_events': extreme_weather_events, 'road_closures_construction': road_closures_construction,
                   'average_income_levels': average_income_levels})

df = df.loc[:, ~df.columns.duplicated()]

# Train Prophet models
models = {}
forecasts = {}
parameters = df.columns[1:]

for param in parameters:
    model = Prophet()
    for regressor in df.columns[2:]:
        if regressor != param:
            model.add_regressor(regressor)
    df_param = df[['ds', param] + list(df.columns[2:])].copy()
    df_param.rename(columns={param: 'y'}, inplace=True)
    df_param = df_param.loc[:, ~df_param.columns.duplicated()]
    model.fit(df_param)
    models[param] = model

# Future hourly data
future = models['y'].make_future_dataframe(periods=24 * 7, freq='h')  # 7 days of hourly future
for col in df.columns[2:]:
    if col not in ['ds', 'y']:
        future[col] = np.random.uniform(df[col].min(), df[col].max(), size=len(future))

# Predict
forecast_data = {'ds': future['ds']}
for param, model in models.items():
    forecast = model.predict(future)
    forecast_data[f'{param}_yhat'] = forecast['yhat']

# Save and plot
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('../full_forecast_hourly.csv', index=False)

fig = models['y'].plot(models['y'].predict(future))
plt.show()

fig2 = models['y'].plot_components(models['y'].predict(future))
plt.show()

print(forecast_df.tail())
