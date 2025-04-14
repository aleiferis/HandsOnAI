import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load and preprocess data
# ---------------------------
df = pd.read_csv("urban_mobility.csv")
df.columns = [col.replace('_yhat', '') for col in df.columns]
df['ds'] = pd.to_datetime(df['ds'])

feature_cols = df.columns.difference(['ds', 'y'])
target_col = 'y'

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# Split
X_temp, X_test, y_temp, y_test, ds_temp, ds_test = train_test_split(X_scaled, y_scaled, df['ds'], test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val, ds_train, ds_val = train_test_split(X_temp, y_temp, ds_temp, test_size=0.1111, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ---------------------------
# 2. Hyperparameter Grid
# ---------------------------
param_grid = {
    "latent_dim": [16, 32],
    "hidden_size": [64, 128],
    "lr": [0.0001, 0.0002]
}
grid = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

best_val_mae = float("inf")
best_config = None
best_G = None

# ---------------------------
# 3. Define model classes
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
        x = torch.cat((z, cond), dim=1)
        return self.model(x)

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
    def forward(self, demand, cond):
        x = torch.cat((demand, cond), dim=1)
        return self.model(x)

# ---------------------------
# 4. Manual Grid Search
# ---------------------------
print("🔍 Running manual grid search...")
for config in grid:
    latent_dim, hidden_size, lr = config
    condition_dim = X_train.shape[1]

    G = Generator(latent_dim, condition_dim, hidden_size)
    D = Discriminator(condition_dim, hidden_size)
    g_opt = optim.Adam(G.parameters(), lr=lr)
    d_opt = optim.Adam(D.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Train each GAN for fewer epochs (for tuning)
    for epoch in range(500):
        idx = np.random.randint(0, X_train_tensor.shape[0], 64)
        real_cond = X_train_tensor[idx]
        real_demand = y_train_tensor[idx]
        real_labels = torch.ones((64, 1))
        fake_labels = torch.zeros((64, 1))

        z = torch.randn(64, latent_dim)
        fake_demand = G(z, real_cond)

        d_real = D(real_demand, real_cond)
        d_fake = D(fake_demand.detach(), real_cond)
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        z = torch.randn(64, latent_dim)
        gen_demand = G(z, real_cond)
        d_pred = D(gen_demand, real_cond)
        g_loss = criterion(d_pred, real_labels)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    # Validation MAE
    z_val = torch.randn(X_val_tensor.shape[0], latent_dim)
    val_generated = G(z_val, X_val_tensor).detach().numpy()
    val_pred = scaler_y.inverse_transform(val_generated)
    val_true = scaler_y.inverse_transform(y_val_tensor.numpy())
    val_mae = mean_absolute_error(val_true, val_pred)

    print(f"Config: latent_dim={latent_dim}, hidden_size={hidden_size}, lr={lr} → Val MAE: {val_mae:.4f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_config = dict(zip(param_names, config))
        best_G = G

print(f"\nBest Config: {best_config} → Val MAE: {best_val_mae:.4f}")

# ---------------------------
# 5. Final Evaluation on Test Set with Best Generator
# ---------------------------
latent_dim = best_config['latent_dim']
z_test = torch.randn(X_test_tensor.shape[0], latent_dim)
test_generated = best_G(z_test, X_test_tensor).detach().numpy()
test_pred = scaler_y.inverse_transform(test_generated).flatten()
test_true = scaler_y.inverse_transform(y_test_tensor.numpy()).flatten()
test_mae = mean_absolute_error(test_true, test_pred)
print(f"Final Test MAE (best config): {test_mae:.4f}")

# ---------------------------
# 6. Compare Actual vs Simulated Demand Over Time
# ---------------------------
df_compare = pd.DataFrame({
    'ds': ds_test.values,
    'Actual_Demand': test_true,
    'Simulated_Demand': test_pred
}).sort_values('ds')

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_compare, x='ds', y='Actual_Demand', label='Actual Demand')
sns.lineplot(data=df_compare, x='ds', y='Simulated_Demand', label='Simulated Demand (GAN)', linestyle='--')
plt.title("Actual vs Simulated Mobility Demand Over Time (Test Set)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()