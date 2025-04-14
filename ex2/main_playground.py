import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, LeakyReLU
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic hourly dataset
def generate_hourly_dataset(n):
    date_rng = pd.date_range(start='2020-01-01', periods=n, freq='h')
    hour_of_day = date_rng.hour
    day_of_year = date_rng.dayofyear

    # Simulate urban mobility (e.g., daily ridership) with sinusoidal functions
    base_mobility = (
        30000
        + 10000 * np.sin(2 * 2 * np.pi * hour_of_day / 24 - 2 * np.pi * 8/24) # 2 peak times per day
        + 5000 * np.sin(2 * np.pi * day_of_year / 365)
    )
    mobility_noise = np.random.normal(scale=2000, size=n)
    mobility = base_mobility + mobility_noise
    # # Plot mobility data
    # plt.figure(figsize=(12, 6))
    # plt.plot(date_rng, mobility, label='Mobility ')
    # plt.xlabel('Hour')
    # plt.ylabel('Mobility')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # exit()

    # Simulate temperature (constant + seasonal pattern + daily pattern + noise)
    temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + 8 * np.sin(2 * np.pi * hour_of_day / 24 - 2 * np.pi * 10/24) + np.random.normal(scale=2, size=n)
    # Simulate precipitation (random spikes, higher in winter)
    precipitation = np.random.gamma(shape=2, scale=2, size=n)
    precipitation[date_rng.month.isin([6, 7, 8])] *= 0.2 # Less rain in summer

    df = pd.DataFrame({
        'ds': date_rng,
        'mobility': mobility,
        'temp': temp,
        'precip': precipitation,
        'extreme_weather_events': np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        'traffic_congestion': np.random.uniform(0, 100, size=n), # Traffic congestion index
        'bike_lane_availability': np.random.choice([0, 1], size=n, p=[0.7, 0.3]), # 30% of days have bike lane expansion
        'road_closures_construction': np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        'fuel_prices': 2.5 + np.random.normal(0, 0.2, size=n), # Fuel price in dollars per liter
        'public_transport_fares': 1.5 + np.random.normal(0, 0.1, size=n), # Public transport fare
        'average_income_levels': np.random.normal(50000, 5000, size=n), # 10% of days have a major event
        'green_initiatives': np.random.choice([0, 1], size=n, p=[0.9, 0.1]), # 10% days have green initiatives
        'tourism_trends': np.random.uniform(0, 100, size=n) + 30 * np.sin(2 * np.pi * day_of_year / 365), # Tourism trends (higher in summer and holiday season)
        'event_impact': np.random.choice([0, 1], size=n, p=[0.9, 0.1]), # Extreme weather events (1 = extreme weather event occurs)
        'public_transport_reliability': np.random.uniform(0, 100, size=n), # Public transportation reliability (higher values mean more reliable transport)
        'school_open': np.where(date_rng.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5, 6]), 1, 0) # School and university schedules (Higher mobility in academic months)
    })
    return df


# Generate dataset for 3 years
real_df = generate_hourly_dataset(24 * 365 * 3)
# corrupt the dataset with NaNs
np.random.seed(42)
for col in real_df.columns[1:]:  # exclude 'ds'
    nan_indices = np.random.choice(real_df.index, size=10, replace=False)
    real_df.loc[nan_indices, col] = np.nan


# Mean imputation for all features except datetime
imputed_df = real_df.copy()
for col in imputed_df.columns[1:]:
    imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mean())


# Ensure there are no duplicate column names
imputed_df = imputed_df.loc[:, ~imputed_df.columns.duplicated()]

# Define and train Prophet models for each parameter
models = {}
forecasts = {}
parameters = imputed_df.columns[1:]

for param in parameters:
    model = Prophet()
    for regressor in imputed_df.columns[2:]:  # Avoid adding the target variable as a regressor
        if regressor != param:
            model.add_regressor(regressor)
    df_param = imputed_df[['ds', param] + list(imputed_df.columns[2:])].copy()
    df_param.rename(columns={param: 'y'}, inplace=True)
    df_param = df_param.loc[:, ~df_param.columns.duplicated()]
    model.fit(df_param)
    models[param] = model

print(models.keys())
# Generate future dataframe
future = models['mobility'].make_future_dataframe(periods=365*24)
for col in imputed_df.columns[2:]:
    if col not in ['ds', 'y']:
        future[col] = np.random.uniform(imputed_df[col].min(), imputed_df[col].max(), size=len(future))

# Make predictions
forecast_data = {'ds': future['ds']}
for param, model in models.items():
    forecast = model.predict(future)
    forecast_data[f'{param}_yhat'] = forecast['yhat']

# Save predictions to CSV
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('../full_forecast.csv', index=False)

# Plot results for urban mobility
fig = models['y'].plot(models['y'].predict(future))
plt.show()

# Plot components
fig2 = models['y'].plot_components(models['y'].predict(future))
plt.show()

# Display sample forecast data
print(forecast_df.tail())

df = forecast_df

features = df.columns.drop(['ds', 'mobility'])
target = 'mobility'

X_raw = df[features]
y_raw = df[[target]]


# Compute IQR
q1 = np.percentile(y_raw, 25)
q3 = np.percentile(y_raw, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify outliers
outliers = (y_raw < lower_bound) | (y_raw > upper_bound)

# # Plot real data
# plt.figure(figsize=(12, 6))
# plt.plot(y_raw, label='Mobility (Training Set)')
# plt.scatter(np.where(outliers)[0], y_raw[outliers], color='red', label='Outliers')
# plt.axhline(lower_bound, color='orange', linestyle='--', label='Lower Bound')
# plt.axhline(upper_bound, color='orange', linestyle='--', label='Upper Bound')
# plt.title('Real Mobility Data (Training Set) with Outliers Highlighted')
# plt.xlabel('Hour Index')
# plt.ylabel('Mobility')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# exit()

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw)


# Spectral analysis for seasonality extraction

from scipy.fft import fft, fftfreq

n = len(y)
timestep = 1  # hourly
frequencies = fftfreq(n, d=timestep)
amplitudes = np.abs(fft(y - np.mean(y)))

plt.figure(figsize=(10, 5))
plt.plot(frequencies[1:n // 2], amplitudes[1:n // 2])
plt.title('Frequency Spectrum of Mobility Data')
plt.xlabel('Frequency (1/hour)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
exit()




# Split into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

# Conditional GAN configuration
latent_dim = 10
condition_dim = X_train.shape[1]

# Generator
def build_generator():
    noise_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(condition_dim,))
    merged = Concatenate()([noise_input, condition_input])
    x = Dense(64, activation='relu')(merged)
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='tanh')(x)
    return Model([noise_input, condition_input], out)

# Discriminator
def build_discriminator():
    mobility_input = Input(shape=(1,))
    condition_input = Input(shape=(condition_dim,))
    merged = Concatenate()([mobility_input, condition_input])
    x = Dense(128, activation=LeakyReLU(0.2))(merged)
    x = Dense(64, activation=LeakyReLU(0.2))(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model([mobility_input, condition_input], out)

# Build and compile models
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()
z = Input(shape=(latent_dim,))
condition = Input(shape=(condition_dim,))
generated = generator([z, condition])
discriminator.trainable = False
validity = discriminator([generated, condition])

cgan = Model([z, condition], validity)
cgan.compile(loss='binary_crossentropy', optimizer=optimizer)

# Training loop
epochs = 5000
batch_size = 64
half_batch = batch_size // 2

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_conditions = X_train[idx]
    real_mobility = y_train[idx]

    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    fake_mobility = generator.predict([noise, real_conditions])

    d_loss_real = discriminator.train_on_batch([real_mobility, real_conditions], np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch([fake_mobility, real_conditions], np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    conditions = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
    g_loss = cgan.train_on_batch([noise, conditions], np.ones((batch_size, 1)))

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f}, Acc.: {100*d_loss[1]:.2f}% | G Loss: {g_loss:.4f}")

# Simulate mobility under custom condition
custom_conditions = scaler_x.transform([df[features].iloc[-1].values])
simulated_noise = np.random.normal(0, 1, (24, latent_dim))
simulated_conditions = np.repeat(custom_conditions, 24, axis=0)
simulated_output = generator.predict([simulated_noise, simulated_conditions])
simulated_mobility = scaler_y.inverse_transform(simulated_output)

# Plot simulation
plt.figure(figsize=(10, 5))
plt.plot(simulated_mobility, label='Simulated Mobility (Conditional GAN)')
plt.title("Mobility Digital Twin Simulation via Conditional GAN")
plt.xlabel("Hour")
plt.ylabel("Mobility")
plt.legend()
plt.grid()
plt.show()








q