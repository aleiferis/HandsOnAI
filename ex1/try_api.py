import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from fastapi import FastAPI
import uvicorn

# Set seed for reproducibility
np.random.seed(42)

app = FastAPI()
current_dataset = None

# Function to generate dataset
def generate_dataset(n):
    date_rng = pd.date_range(start='2020-01-01', periods=n, freq='D')
    base_mobility = 50000 + 2000 * np.sin(2 * np.pi * date_rng.dayofyear / 365)
    mobility_noise = np.random.normal(scale=5000, size=n)
    mobility = base_mobility + mobility_noise

    temp = 15 + 10 * np.sin(2 * np.pi * date_rng.dayofyear / 365) + np.random.normal(scale=2, size=n)
    precipitation = np.random.gamma(shape=2, scale=2, size=n)
    precipitation[date_rng.month.isin([6, 7, 8])] *= 0.2
    
    green_initiatives = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    traffic_congestion = np.random.uniform(0, 100, size=n)
    bike_lane_availability = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    fuel_prices = 2.5 + np.random.normal(0, 0.2, size=n)
    public_transport_fares = 1.5 + np.random.normal(0, 0.1, size=n)
    event_impact = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    school_open = np.where(date_rng.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5, 6]), 1, 0)
    remote_work_trends = np.random.uniform(0, 100, size=n)
    tourism_trends = np.random.uniform(0, 100, size=n) + 30 * np.sin(2 * np.pi * date_rng.dayofyear / 365)
    public_transport_reliability = np.random.uniform(0, 100, size=n)
    extreme_weather_events = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    road_closures_construction = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    average_income_levels = np.random.normal(50000, 5000, size=n)
    
    df = pd.DataFrame({
        'ds': date_rng, 'y': mobility, 'temp': temp, 'precip': precipitation,
        'green_initiatives': green_initiatives, 'traffic_congestion': traffic_congestion,
        'bike_lane_availability': bike_lane_availability, 'fuel_prices': fuel_prices,
        'public_transport_fares': public_transport_fares, 'event_impact': event_impact,
        'school_open': school_open, 'remote_work_trends': remote_work_trends,
        'tourism_trends': tourism_trends, 'public_transport_reliability': public_transport_reliability,
        'extreme_weather_events': extreme_weather_events, 'road_closures_construction': road_closures_construction,
        'average_income_levels': average_income_levels
    })
    return df

@app.post("/generate")
def generate(n: int):
    global current_dataset
    current_dataset = generate_dataset(n)
    return {"message": f"Dataset with {n} lines created."}

@app.get("/get")
def get_dataset():
    if current_dataset is None:
        return {"message": "No dataset available."}
    return current_dataset.to_dict(orient='records')

@app.post("/add")
def add(n: int):
    global current_dataset
    if current_dataset is None:
        return {"message": "No dataset available. Use /generate first."}
    new_data = generate_dataset(n)
    current_dataset = pd.concat([current_dataset, new_data], ignore_index=True)
    return {"message": f"{n} new lines added to the dataset."}

@app.delete("/delete")
def delete():
    global current_dataset
    current_dataset = None
    return {"message": "Dataset deleted."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
