import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load & preprocess data
df = pd.read_csv('../full_forecast_hourly.csv')
df.columns = [col.replace('_yhat', '') for col in df.columns]

# Remove outliers
def remove_outliers_iqr(data):
    for col in data.select_dtypes(include='number').columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df = remove_outliers_iqr(df)

# Normalize
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Optional: encode interventions
# Example: df['event_flag'] = (df['event_type'] != 'none').astype(int)

# Split data
train_data, _ = train_test_split(df[numeric_cols], test_size=0.2, random_state=42)
train_tensor = torch.tensor(train_data.values, dtype=torch.float32)

# Step 2: Define GAN components
latent_dim = 32
data_dim = train_tensor.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# Instantiate
generator = Generator()
discriminator = Discriminator()

# Optimizers
lr = 0.0002
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Loss
loss_fn = nn.BCELoss()

# Step 3: Train the GAN
epochs = 5000
batch_size = 128

for epoch in range(epochs):
    # Real samples
    real = train_tensor[np.random.randint(0, train_tensor.shape[0], batch_size)]
    real_labels = torch.ones((batch_size, 1))

    # Fake samples
    z = torch.randn(batch_size, latent_dim)
    fake = generator(z)
    fake_labels = torch.zeros((batch_size, 1))

    # Train discriminator
    d_loss_real = loss_fn(discriminator(real), real_labels)
    d_loss_fake = loss_fn(discriminator(fake.detach()), fake_labels)
    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # Train generator
    z = torch.randn(batch_size, latent_dim)
    generated = generator(z)
    g_loss = loss_fn(discriminator(generated), real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Step 4: Generate new data (simulate intervention scenario)
z = torch.randn(1000, latent_dim)
simulated_data = generator(z).detach().numpy()

# Invert normalization
simulated_df = pd.DataFrame(scaler.inverse_transform(simulated_data), columns=numeric_cols)

print(simulated_df.head())