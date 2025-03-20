import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
days = 365 * 3  # 3 years of data
date_rng = pd.date_range(start='2020-01-01', periods=days, freq='D')

# Simulate urban mobility (e.g., daily ridership)
base_mobility = 50000 + 2000 * np.sin(2 * np.pi * date_rng.dayofyear / 365)
mobility_noise = np.random.normal(scale=5000, size=days)
mobility = base_mobility + mobility_noise

# Simulate temperature (seasonal pattern + noise)
temp = 15 + 10 * np.sin(2 * np.pi * date_rng.dayofyear / 365) + np.random.normal(scale=2, size=days)

# Simulate precipitation (random spikes, higher in winter)
precipitation = np.random.gamma(shape=2, scale=2, size=days)
precipitation[date_rng.month.isin([6, 7, 8])] *= 0.2  # Less rain in summer

# Simulate additional factors
green_initiatives = np.random.choice([0, 1], size=days, p=[0.9, 0.1])  # 10% days have green initiatives
daily_traffic_congestion = np.random.uniform(0, 100, size=days)  # Traffic congestion index
bike_lane_availability = np.random.choice([0, 1], size=days, p=[0.7, 0.3])  # 30% of days have bike lane expansion
fuel_prices = 2.5 + np.random.normal(0, 0.2, size=days)  # Fuel price in dollars per liter
public_transport_fares = 1.5 + np.random.normal(0, 0.1, size=days)  # Public transport fare

# Simulate event-based factors (sports events, concerts, protests combined)
event_impact = np.random.choice([0, 1], size=days, p=[0.9, 0.1])  # 10% of days have a major event

# School and university schedules (Higher mobility in academic months)
school_open = np.where(date_rng.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5, 6]), 1, 0)

# Additional parameters
remote_work_trends = np.random.uniform(30, 40, size=days)  # Percentage of remote workers

# Tourism trends (higher in summer and holiday season)
tourism_trends = np.random.uniform(0, 100, size=days) + 30 * np.sin(2 * np.pi * date_rng.dayofyear / 365)

# Public transportation reliability (higher values mean more reliable transport)
public_transport_reliability = np.random.uniform(0, 100, size=days)

# Extreme weather events (1 = extreme weather event occurs)
extreme_weather_events = np.random.choice([0, 1], size=days, p=[0.95, 0.05])

# Road closures and construction sites (1 = disruption present)
road_closures_construction = np.random.choice([0, 1], size=days, p=[0.9, 0.1])

# Average income levels (simulated economic factor)
average_income_levels = np.random.normal(50000, 5000, size=days)

# Create DataFrame
df = pd.DataFrame({'ds': date_rng, 'y': mobility, 'temp': temp, 'precip': precipitation,
                   'green_initiatives': green_initiatives, 'traffic_congestion': daily_traffic_congestion,
                   'bike_lane_availability': bike_lane_availability, 'fuel_prices': fuel_prices,
                   'public_transport_fares': public_transport_fares, 'event_impact': event_impact,
                   'school_open': school_open, 'remote_work_trends': remote_work_trends,
                   'tourism_trends': tourism_trends, 'public_transport_reliability': public_transport_reliability,
                   'extreme_weather_events': extreme_weather_events, 'road_closures_construction': road_closures_construction,
                   'average_income_levels': average_income_levels})

# Ensure there are no duplicate column names
df = df.loc[:, ~df.columns.duplicated()]

# Define and train Prophet models for each parameter
models = {}
forecasts = {}
parameters = df.columns[1:]

for param in parameters:
    model = Prophet()
    for regressor in df.columns[2:]:  # Avoid adding the target variable as a regressor
        if regressor != param:
            model.add_regressor(regressor)
    df_param = df[['ds', param] + list(df.columns[2:])].copy()
    df_param.rename(columns={param: 'y'}, inplace=True)
    df_param = df_param.loc[:, ~df_param.columns.duplicated()]
    model.fit(df_param)
    models[param] = model

# Generate future dataframe
future = models['y'].make_future_dataframe(periods=365)
for col in df.columns[2:]:
    if col not in ['ds', 'y']:
        future[col] = np.random.uniform(df[col].min(), df[col].max(), size=len(future))

# Make predictions
forecast_data = {'ds': future['ds']}
for param, model in models.items():
    forecast = model.predict(future)
    forecast_data[f'{param}_yhat'] = forecast['yhat']

# Save predictions to CSV
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('full_forecast.csv', index=False)

# Plot results for urban mobility
fig = models['y'].plot(models['y'].predict(future))
plt.show()

# Plot components
fig2 = models['y'].plot_components(models['y'].predict(future))
plt.show()

# Display sample forecast data
print(forecast_df.tail())
