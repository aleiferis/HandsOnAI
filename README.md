# HandsOnAI Exercise 1
Hands-on AI - Assignment 1

Chosen topic: Mobility --> Urban Mobility and Parameters that affect it.
The developed model makes predictions about the following parameters: 

Core Urban Mobility Metrics
	•	Urban Mobility (y) – The main target variable representing the number of people using public transport.
    •	Basic periodic urban mobility + noise

Weather & Environmental Factors
	•	Temperature (temp) – Daily temperature variations. (basic periodic + noise)
	•	Precipitation (precip) – Amount of rainfall affecting mobility. (depends on accumulation of water droplets, thus random gamma distribution is used related to waiting times + a periodic component is added to account for the lack of rain in the summer)
	•	Extreme Weather Events (extreme_weather_events) – Binary indicator (1 = extreme weather event occurred).


Traffic & Infrastructure
	•	Traffic Congestion (traffic_congestion) – Traffic index affecting public transport and mobility patterns. (random uniform)
	•	Bike Lane Availability (bike_lane_availability) – Binary indicator (1 = new bike lanes available).
	•	Road Closures & Construction (road_closures_construction) – Binary indicator (1 = road closure or construction present).

Economic & Financial Factors
	•	Fuel Prices (fuel_prices) – Daily fuel cost fluctuations.
	•	Public Transport Fares (public_transport_fares) – Changes in transit fare costs.
	•	Average Income Levels (average_income_levels) – Represents the general economic conditions influencing mobility behavior. This is relatively stable with small random variation. 

Policy & Behavioral Trends
	•	Green Initiatives (green_initiatives) – Binary indicator (1 = government policies promoting eco-friendly transport). (about 10% of days)
	•	Remote Work Trends (remote_work_trends) – Percentage of people working remotely, reducing peak-hour commutes. Random with small variation around a typical perentage of 35%

Tourism & Event-Based Factors
	•	Tourism Trends (tourism_trends) – A measure of how tourism activity influences transport demand. (shows yearly periodical variation)
	•	Event Impact (event_impact) – Binary indicator (1 = major events such as sports, concerts, or protests).

Public Transport Reliability
	•	Public Transport Reliability (public_transport_reliability) – Index representing transit reliability (delays, strikes, etc.).

Education & School Impact
	•	School & University Schedules (school_open) – Binary indicator (1 = school/university is in session). --> Accounts for the months that the schools are open and not the days. This can be refined. 

All these parameters are considered as regressors in the Prophet model, meaning they help the model understand patterns and improve future forecasts.

Important note: Parameters such as the public transport fare, it might be argued that they shoud remain constant. However, the values considered here are effective values for the cost of public transport, which potential include the use of taxis or transfer to different charging zones. These values change from day to day and the variation that is introduced reflects that. 


# To run the postgres container
cd postgres-docker
docker compose up -d --build
cd .. 

# To edit manualy
docker exec -it postgres_db psql -U user -d db_name


# HandsOnAI Exercise 2

ex1/hourly.py # modified code to produce the same data as exercise 1 but with hourly frequency

$ cd ex2

# Create virtual environment 
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install torch torchvision torchaudio
$ pip install tensorflow tensorflow-macos tensorflow-metal
$ pip install pandas

The dataset was extended to hourly frequency.

The code now uses a Generative Adversarial Network (GAN) to simulate hourly mobility data:The generator learns to produce realistic mobility patterns.
	•	The discriminator learns to distinguish between real and fake mobility values.
	•	Once trained, you can sample from the generator to simulate future mobility scenarios (i.e., your digital twin).

Let me know if you’d like to condition the GAN on weather or policy inputs, or visualize multiple scenarios! 🧠📈

The code now removes outliers from the training set using the IQR (Interquartile Range) method on the mobility target variable before training the GAN. This improves model robustness and reduces noise caused by extreme values.


