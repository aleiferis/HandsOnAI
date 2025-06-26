import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error

# Step 1: Download and unzip the dataset
print("Downloading dataset...")
zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
zip_file = 'household_power_consumption.zip'
txt_file = 'household_power_consumption.txt'

if not os.path.exists(txt_file):
    urlretrieve(zip_url, zip_file)
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

# Step 2: Load and clean the dataset
print("Reading data...")
df = pd.read_csv(txt_file, sep=';', low_memory=False, na_values='?',nrows=20000)
df.dropna(inplace=True)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.sort_values('datetime', inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

for col in df.columns.difference(['datetime']):
    df[col] = df[col].astype(float)

# Step 3: Feature engineering
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['global_active_power_ma'] = df['Global_active_power'].rolling(window=6, min_periods=1).mean()

# Step 4: Outlier removal
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

df = remove_outliers(df, 'Global_active_power')

# Step 5: Normalize
features = df.columns.difference(['datetime', 'Global_active_power'])
target = 'Global_active_power'

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(df[features])
y = scaler_y.fit_transform(df[[target]])

# Step 6: Chronological split
n = len(df)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
ds_train, ds_val, ds_test = df['datetime'][:train_end], df['datetime'][train_end:val_end], df['datetime'][val_end:]

# Step 7: Feature selection
rf = RandomForestRegressor()
rfe = RFE(rf, n_features_to_select=5)
rfe.fit(X_train, y_train.ravel())
selected_rfe = np.array(features)[rfe.support_]
print("RFE Selected Features:", selected_rfe)

# Tree-based importance
rf.fit(X_train, y_train.ravel())
importances = rf.feature_importances_
important_features = [features[i] for i in np.argsort(importances)[-5:]]

# Step 8: t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_embedded = tsne.fit_transform(X_train)

plt.figure(figsize=(8, 5))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train.ravel(), cmap='viridis', s=5)
plt.title("t-SNE Visualization of Training Data")
plt.colorbar(label='Scaled Power Consumption')
plt.show()

# Step 9: Final Model
final_model = RandomForestRegressor()
final_model.fit(X_train, y_train.ravel())
y_pred = final_model.predict(X_test)

# Inverse transform
y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_true_real = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_true_real, y_pred_real)
print(f"Final Test MAE: {mae:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(ds_test.values, y_true_real, label='Actual')
plt.plot(ds_test.values, y_pred_real, label='Predicted', linestyle='--')
plt.title("Actual vs Predicted Household Power Consumption")
plt.xlabel("Datetime")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()