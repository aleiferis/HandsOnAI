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

	Problem to solve: Simulate hourly mobility data using Generative Adversarial Network (GAN).
		GANs might be an overkill, and a simpler solution could be provided by LSTMs which are reliable and less sensitive to hyperparameters. However GANs have the following advantages: 
			1. Better for what-if policy testing and not just predicting the future based on the past. 
			2. Distribution outcome instead of single point prediction
			3. I was looking for an opportunity to use them !!! 

	NOTE: venv and requirements are optimized for Apple Silicon (M1) through MPS, but should be able to run in regular CUDA environemnt as well.  

	Instructions: 

	$ cd ex2

	# Create virtual environment 
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -U pip
	pip install -r ../requirements.txt
	python3 main.py

	The dataset was extended to hourly frequency:
	ex1/hourly.py # modified code to produce the same data as exercise 1 but with hourly frequency
		new file data full_forecast_hourly.csv
		trafic_congestions distribution was replaced with lognormal which is righ-skewed

	The code now uses a Generative Adversarial Network (GAN) to simulate hourly mobility data: 
		The generator learns to produce realistic mobility patterns.
		The discriminator learns to distinguish between real and fake mobility values.
		Once trained, you can sample from the generator to simulate future mobility scenarios (much like a digital twin).

	Main functionality: 
	•	Use your dataframe to train a Conditional GAN (cGAN) that learns to simulate future mobility demand (y_yhat) under various urban condition scenarios (e.g., events, weather, policies).
	•	Use the full range of urban condition features
	•	Allow user to input real or hypothetical urban scenarios
	•	Plot the simulated demand distribution per scenario

	All code can be run from ex2/main.py except for the optimization with GridSearch which is implemented in ex2/main_part5.py
	Optimization output (Step 5): 
		Running manual grid search...
		Config: latent_dim=16, hidden_size=64, lr=0.0001 → Val MAE: 2117.1714
		Config: latent_dim=16, hidden_size=64, lr=0.0002 → Val MAE: 2564.1125
		Config: latent_dim=16, hidden_size=128, lr=0.0001 → Val MAE: 2406.1328
		Config: latent_dim=16, hidden_size=128, lr=0.0002 → Val MAE: 3060.2461
		Config: latent_dim=32, hidden_size=64, lr=0.0001 → Val MAE: 2126.6619
		Config: latent_dim=32, hidden_size=64, lr=0.0002 → Val MAE: 2422.7285
		Config: latent_dim=32, hidden_size=128, lr=0.0001 → Val MAE: 2604.9985
		Config: latent_dim=32, hidden_size=128, lr=0.0002 → Val MAE: 2969.8474

		Best Config: {'latent_dim': 16, 'hidden_size': 64, 'lr': 0.0001} → Val MAE: 2117.1714
		Final Test MAE (best config): 2155.6846

	Additional functionality: 
		- Mean imputation is used to deal with missing values
		- Outliers are removed using IQR
		- MinMax Scaler is used for normalization
		- Tahn were tried in place of ReLU 
		- Feature importance is used for evaluation
			- Tourism Trends and Temperature appear to be the most important features
		- Since we are doing regression we use Mean Absolute Error (MAE) as and Accuracy metric.

	6. Endpoint structure: 
		project/
		├── app.py                 # FastAPI app
		├── saved_model/
		│   ├── generator_model.pkl
		│   ├── scaler_X.pkl
		│   └── scaler_y.pkl

		Request JSON (e.g.):
			{
				"data": {
					"temp": 15.2,
					"precip": 0.1,
					"green_initiatives": 0.5,
					"traffic_congestion": 0.6,
					"bike_lane_availability": 0.8,
					"fuel_prices": 1.3,
					"public_transport_fares": 0.9,
					"event_impact": 1.0,
					"school_open": 1.0,
					"remote_work_trends": 0.7,
					"tourism_trends": 0.5,
					"public_transport_reliability": 0.85,
					"extreme_weather_events": 0.0,
					"road_closures_construction": 0.1,
					"average_income_levels": 0.6
				}
			}

		Response JSON: 
			{
				"predicted_demand": 1537.27
			}

	Next Steps: 
	- Bayesian optimization (e.g., Optuna)

	Other ideas / potential applications: 
	1. Mobility Equity Analysis (requires breakdown to interconnected graph of mobility predictions)
	2. Simulating Policy Impact
	3. Plan for Tourism-Driven Transportation Needs
	4. Study Fare Price Sensitivity
	5. Weather Resilience Planning
	6. Event Impact Analysis