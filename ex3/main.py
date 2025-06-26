import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

# ---------------------------
# 1. Load Dataset
# ---------------------------
if not os.path.exists("household_power_consumption.txt"):
    urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip", "data.zip")
    with ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall()

df = pd.read_csv("household_power_consumption.txt", sep=';', na_values='?', low_memory=False)
df.dropna(inplace=True)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

for col in df.columns.difference(['datetime']):
    df[col] = df[col].astype(float)

# ---------------------------
# 2. Feature Engineering
# ---------------------------
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['global_active_power_ma'] = df['Global_active_power'].rolling(6, min_periods=1).mean()

def remove_outliers(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

df = remove_outliers(df, 'Global_active_power')

# ---------------------------
# 3. Scaling and Splitting
# ---------------------------
features = df.columns.difference(['datetime', 'Global_active_power'])
target = 'Global_active_power'

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(df[features])
y = scaler_y.fit_transform(df[[target]])

n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
ds_test = df['datetime'][val_end:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ---------------------------
# 4. Model Classes
# ---------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, cond):
        return self.model(torch.cat((z, cond), dim=1))


class Discriminator(nn.Module):
    def __init__(self, condition_dim, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(condition_dim + 1, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, y, cond):
        return self.model(torch.cat((y, cond), dim=1))

# ---------------------------
# 5. Hyperparameter Tuning
# ---------------------------
param_grid = {
    "latent_dim": [16, 32],
    "hidden_size": [64, 128],
    "lr": [0.0002]
}
grid = list(itertools.product(*param_grid.values()))
best_val_mae = float("inf")

history = []

print("🔍 Starting grid search...")
for latent_dim, hidden_size, lr in grid:
    G = Generator(latent_dim, X_train.shape[1], hidden_size)
    D = Discriminator(X_train.shape[1], hidden_size)
    g_opt = optim.Adam(G.parameters(), lr=lr)
    d_opt = optim.Adam(D.parameters(), lr=lr)
    criterion = nn.BCELoss()

    val_maes = []
    for epoch in range(1000):
        idx = np.random.randint(0, X_train_tensor.shape[0], 64)
        cond = X_train_tensor[idx]
        real = y_train_tensor[idx]
        z = torch.randn(64, latent_dim)

        fake = G(z, cond)
        d_loss = criterion(D(real, cond), torch.ones((64, 1))) + criterion(D(fake.detach(), cond), torch.zeros((64, 1)))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        z = torch.randn(64, latent_dim)
        fake = G(z, cond)
        g_loss = criterion(D(fake, cond), torch.ones((64, 1)))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        if epoch % 100 == 0:
            z_val = torch.randn(X_val_tensor.shape[0], latent_dim)
            val_out = G(z_val, X_val_tensor).detach().numpy()
            val_pred = scaler_y.inverse_transform(val_out)
            val_true = scaler_y.inverse_transform(y_val_tensor.numpy())
            val_mae = mean_absolute_error(val_true, val_pred)
            val_maes.append(val_mae)
            print(f"Epoch {epoch} | MAE: {val_mae:.4f}")

    if val_maes[-1] < best_val_mae:
        best_val_mae = val_maes[-1]
        best_G = G
        best_params = (latent_dim, hidden_size, lr)
        best_history = val_maes

# ---------------------------
# 6. Final Evaluation
# ---------------------------
latent_dim = best_params[0]
z_test = torch.randn(X_test_tensor.shape[0], latent_dim)
y_gen = best_G(z_test, X_test_tensor).detach().numpy()

y_pred = scaler_y.inverse_transform(y_gen)
y_true = scaler_y.inverse_transform(y_test_tensor.numpy())

# Evaluate
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Fixed here
r2 = r2_score(y_true, y_pred)
print(f"\nBest Params: {best_params} → MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# ---------------------------
# 7. Visualizations
# ---------------------------

# Learning Curve
plt.figure()
plt.plot(np.arange(0, len(best_history)) * 100, best_history)
plt.title("Learning Curve (Validation MAE)")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.grid()
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_true.flatten() - y_pred.flatten()
plt.figure()
plt.scatter(y_true, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Actual Power")
plt.ylabel("Residuals")
plt.grid()
plt.tight_layout()
plt.show()

# Feature Importance from surrogate model
surrogate = RandomForestRegressor()
surrogate.fit(X_test, y_pred.flatten())
importances = surrogate.feature_importances_

plt.figure(figsize=(10, 5))
plt.barh(df[features].columns, importances)
plt.title("Feature Importance (Random Forest Surrogate)")
plt.xlabel("Importance")
plt.tight_layout()
plt.grid(True)
plt.show()

# ROC Curve (binary: is power > median?)
threshold = np.median(y_true)
y_true_binary = (y_true > threshold).astype(int)
y_score = y_pred.flatten()
fpr, tpr, _ = roc_curve(y_true_binary, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Binary: High vs Low Power)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
