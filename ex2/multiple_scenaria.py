import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle 

# ---------------------------
# 1. Load and preprocess data
# ---------------------------
df = pd.read_csv("../full_forecast_hourly.csv")
df.columns = [col.replace('_yhat', '') for col in df.columns]
df['ds'] = pd.to_datetime(df['ds'])

feature_cols = df.columns.difference(['ds', 'y'])
target_col = 'y'

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# ---------------------------
# 2. Train / Val / Test Split
# ---------------------------
X_temp, X_test, y_temp, y_test, ds_temp, ds_test = train_test_split(
    X_scaled, y_scaled, df['ds'], test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val, ds_train, ds_val = train_test_split(
    X_temp, y_temp, ds_temp, test_size=0.1111, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# ---------------------------
# 3. GAN Setup
# ---------------------------
latent_dim = 16
condition_dim = X_train.shape[1]
output_dim = 1

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    def forward(self, z, cond):
        x = torch.cat((z, cond), dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(condition_dim + output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, demand, cond):
        x = torch.cat((demand, cond), dim=1)
        return self.model(x)

G = Generator()
D = Discriminator()

g_opt = optim.Adam(G.parameters(), lr=0.0002)
d_opt = optim.Adam(D.parameters(), lr=0.0002)
criterion = nn.BCELoss()


# # ---------------------------
# # 3. DEEPER GAN SETUP
# # ---------------------------
# latent_dim = 32
# condition_dim = X_train.shape[1]
# output_dim = 1

# # Change Tahn back to ReLU
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim + condition_dim, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 64),
#             nn.Tanh(),
#             nn.Dropout(0.2),
#             nn.Linear(64, output_dim),
#             nn.Sigmoid()
#         )
#     def forward(self, z, cond):
#         x = torch.cat((z, cond), dim=1)
#         return self.model(x)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(condition_dim + output_dim, 128),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, demand, cond):
#         x = torch.cat((demand, cond), dim=1)
#         return self.model(x)

# G = Generator()
# D = Discriminator()

# g_opt = optim.Adam(G.parameters(), lr=0.0001)
# d_opt = optim.Adam(D.parameters(), lr=0.0001)
# criterion = nn.BCELoss()

# ---------------------------
# 4. Train GAN with Validation
# ---------------------------
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    idx = np.random.randint(0, X_train_tensor.shape[0], batch_size)
    real_cond = X_train_tensor[idx]
    real_demand = y_train_tensor[idx]
    real_labels = torch.ones((batch_size, 1))
    fake_labels = torch.zeros((batch_size, 1))

    # Discriminator
    z = torch.randn(batch_size, latent_dim)
    fake_demand = G(z, real_cond)
    d_real = D(real_demand, real_cond)
    d_fake = D(fake_demand.detach(), real_cond)
    d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()

    # Generator
    z = torch.randn(batch_size, latent_dim)
    gen_demand = G(z, real_cond)
    d_pred = D(gen_demand, real_cond)
    g_loss = criterion(d_pred, real_labels)
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()

    if epoch % 200 == 0:
        z_val = torch.randn(X_val_tensor.shape[0], latent_dim)
        val_generated = G(z_val, X_val_tensor).detach().numpy()
        val_pred = scaler_y.inverse_transform(val_generated)
        val_true = scaler_y.inverse_transform(y_val_tensor.numpy())
        val_mae = mean_absolute_error(val_true, val_pred)
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Val MAE: {val_mae:.4f}")



model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)

# Save generator model
torch.save(G.state_dict(), os.path.join(model_dir, "generator_model.pkl"))

# Save scalers
with open(os.path.join(model_dir, "scaler_X.pkl"), 'wb') as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(model_dir, "scaler_y.pkl"), 'wb') as f:
    pickle.dump(scaler_y, f)

print("Model and scalers saved in 'saved_model/'")


# ---------------------------
# 5. Final Evaluation on Test Set
# ---------------------------
z_test = torch.randn(X_test_tensor.shape[0], latent_dim)
test_generated = G(z_test, X_test_tensor).detach().numpy()
test_pred = scaler_y.inverse_transform(test_generated).flatten()
test_true = scaler_y.inverse_transform(y_test_tensor.numpy()).flatten()
test_mae = mean_absolute_error(test_true, test_pred)
print(f"\nFinal Test MAE: {test_mae:.4f}")

# ---------------------------
# 6. Feature Importance (Random Forest)
# ---------------------------
rf = RandomForestRegressor()
rf.fit(X_train, y_train.ravel())
importances = rf.feature_importances_

plt.figure(figsize=(10, 6))
feat_names = feature_cols.tolist()
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.grid(True)
plt.show()

# # ---------------------------
# # 7. Plot Actual vs Simulated Demand Over Time
# # ---------------------------
# df_compare = pd.DataFrame({
#     'ds': ds_test.values,
#     'Actual_Demand': test_true,
#     'Simulated_Demand': test_pred
# })
# df_compare = df_compare.sort_values('ds')

# plt.figure(figsize=(14, 6))
# sns.lineplot(data=df_compare, x='ds', y='Actual_Demand', label='Actual Demand')
# sns.lineplot(data=df_compare, x='ds', y='Simulated_Demand', label='Simulated Demand (GAN)', linestyle='--')
# plt.title("Actual vs Simulated Mobility Demand Over Time (Test Set)")
# plt.xlabel("Date")
# plt.ylabel("Demand")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ---------------------------
# 7. Simulate & Compare Demand Under Different Scenarios
# ---------------------------

def simulate_demand(G, scaler_X, scaler_y, base_df, scenario_mods, scenario_names, latent_dim=16):
    scenario_outputs = []

    for mod, label in zip(scenario_mods, scenario_names):
        # Create a copy of the test feature set
        X_scenario = base_df.copy()

        # Apply scenario modifications
        for feature, value in mod.items():
            if feature in X_scenario.columns:
                X_scenario[feature] = value

        # Scale and convert to tensor
        X_scaled = scaler_X.transform(X_scenario)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Generate demand
        z = torch.randn(X_tensor.shape[0], latent_dim)
        gen_demand_scaled = G(z, X_tensor).detach().numpy()
        gen_demand = scaler_y.inverse_transform(gen_demand_scaled).flatten()

        scenario_outputs.append({
            "label": label,
            "ds": ds_test.values,
            "simulated_demand": gen_demand
        })

    return scenario_outputs

# Prepare test data (DataFrame with named columns for modification)
df_test_features = pd.DataFrame(X_test, columns=feature_cols)

# Define scenario modifications
scenario_mods = [
    {},  # baseline (actual test features)
    {"fuel_prices": df_test_features["fuel_prices"].mean() * 1.5},
    {"event_impact": 1.0},
    {"remote_work_trends": 1.0, "school_open": 0.0},
]

scenario_names = [
    "Baseline",
    "High Fuel Prices",
    "Event Day",
    "Remote Work & Schools Closed"
]

# Generate simulated demand for each scenario
scenario_results = simulate_demand(
    G, scaler_X, scaler_y,
    base_df=df_test_features,
    scenario_mods=scenario_mods,
    scenario_names=scenario_names,
    latent_dim=latent_dim
)

# Plot all scenario simulations
plt.figure(figsize=(14, 6))

for result in scenario_results:
    df_result = pd.DataFrame({
        "ds": result["ds"],
        "Simulated_Demand": result["simulated_demand"]
    }).sort_values("ds")

    sns.lineplot(data=df_result, x="ds", y="Simulated_Demand", label=result["label"])

plt.title("Simulated Mobility Demand Under Different Scenarios (Test Set)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()