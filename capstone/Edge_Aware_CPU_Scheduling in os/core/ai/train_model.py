# core/ai/train_model.py
"""
AI Training Pipeline for Edge Node CPU Load Prediction.
Trains a Random Forest Regressor and exports the serialized model.
"""

import os
import random
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def generate_training_data(num_samples: int = 600) -> pd.DataFrame:
    """
    Generate synthetic telemetry samples for ML training.
    """
    data = []
    for _ in range(num_samples):
        current_load = random.randint(10, 90)
        network_latency = random.randint(5, 50)
        queue_length = random.randint(0, 15)
        cpu_capacity = random.randint(80, 160)
        energy_usage = random.randint(80, 300)

        # Ground truth simulated load with non-linear relationships & noise
        future_load = (
            current_load * 0.55
            + queue_length * 2.1
            + network_latency * 0.28
            + energy_usage * 0.03
            - cpu_capacity * 0.08
            + random.uniform(-4.0, 4.0)
        )
        future_load = max(0.0, min(100.0, future_load))

        data.append({
            "Current_Load": current_load,
            "Network_Latency": network_latency,
            "Queue_Length": queue_length,
            "CPU_Capacity": cpu_capacity,
            "Energy_Usage": energy_usage,
            "Future_Load": round(future_load, 2)
        })

    return pd.DataFrame(data)


def train_model() -> RandomForestRegressor:
    """
    Train and save the Random Forest model to disk.
    """
    df = generate_training_data()

    feature_cols = [
        "Current_Load",
        "Network_Latency",
        "Queue_Length",
        "CPU_Capacity",
        "Energy_Usage"
    ]
    X = df[feature_cols]
    y = df["Future_Load"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Random Forest Training Completed. MAE: {round(mae, 2)}")

    # Target export directories
    target_dirs = [
        "models",
        os.path.join("core", "ai", "data", "models"),
        os.path.join(os.path.dirname(__file__), "data", "models")
    ]

    for d in target_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            joblib.dump(model, os.path.join(d, "load_predictor.pkl"))
        except Exception:
            pass

    return model


if __name__ == "__main__":
    train_model()