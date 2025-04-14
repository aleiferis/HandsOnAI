####################################
# Not fully functional yet
####################################
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np

# ---------------------------
# 1. FastAPI Setup
# ---------------------------
app = FastAPI(title="Mobility Demand Predictor")

# ---------------------------
# 2. Load Model & Scalers
# ---------------------------
MODEL_PATH = "saved_model/generator_model.pth"
SCALER_X_PATH = "saved_model/scaler_X.pkl"
SCALER_Y_PATH = "saved_model/scaler_y.pkl"

with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)

with open(SCALER_Y_PATH, "rb") as f:
    scaler_y = pickle.load(f)

input_features = scaler_X.feature_names_in_.tolist()
latent_dim = 16  # use the same as during training
condition_dim = len(input_features)
output_dim = 1

# ---------------------------
# 3. Define Model Classes
# ---------------------------
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

# Load the generator
G = Generator()
G.load_state_dict(torch.load(MODEL_PATH))
G.eval()

# ---------------------------
# 4. Define Input Schema
# ---------------------------
class ScenarioInput(BaseModel):
    data: dict  # expects {"fuel_prices": 1.2, "event_impact": 0.5, ...}

# ---------------------------
# 5. Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict(input_data: ScenarioInput):
    # Extract feature values
    input_dict = input_data.data

    # Validate keys
    missing = [f for f in input_features if f not in input_dict]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Order feature values to match training
    input_vals = np.array([input_dict[feat] for feat in input_features]).reshape(1, -1)

    # Scale input
    X_scaled = scaler_X.transform(input_vals)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Sample latent noise
    z = torch.randn(1, latent_dim)

    # Generate prediction
    with torch.no_grad():
        y_scaled = G(z, X_tensor).numpy()

    # Inverse transform to real demand
    y_pred = scaler_y.inverse_transform(y_scaled)[0][0]

    return {"predicted_demand": float(y_pred)}