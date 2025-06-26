# Not working yet

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "saved_model/generator.pth"
SCALER_X_PATH = "saved_model/scaler_X.pkl"
SCALER_Y_PATH = "saved_model/scaler_y.pkl"
LATENT_DIM = 32  # must match training setup

# ---------------------------
# Load Scalers
# ---------------------------
with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)

with open(SCALER_Y_PATH, "rb") as f:
    scaler_y = pickle.load(f)

input_features = scaler_X.feature_names_in_.tolist()
input_dim = len(input_features)

# ---------------------------
# Define Generator Class
# ---------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_size=128):
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

# Load generator
G = Generator(LATENT_DIM, input_dim)
G.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
G.eval()

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="Power Consumption Predictor (GAN)")

class PredictionInput(BaseModel):
    features: dict

@app.post("/predict")
def predict(input_data: PredictionInput):
    data = input_data.features

    # Validate all features present
    missing = [feat for feat in input_features if feat not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Order and scale
    input_vals = np.array([data[feat] for feat in input_features]).reshape(1, -1)
    try:
        X_scaled = scaler_X.transform(input_vals)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    z = torch.randn(1, LATENT_DIM)

    # Predict
    with torch.no_grad():
        y_scaled = G(z, X_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_scaled)[0][0]

    return {"predicted_power_kw": round(float(y_pred), 4)}
