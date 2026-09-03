# core/ai/predictor.py
"""
AI-Based CPU Load Predictor for Edge Nodes.
Uses Machine Learning (Random Forest) to forecast future CPU load and preemptive bottleneck formation.
"""

import os
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np


def find_model_path() -> Optional[str]:
    """
    Search standard locations for the trained model file.
    """
    candidate_paths = [
        "models/load_predictor.pkl",
        "core/ai/data/models/load_predictor.pkl",
        os.path.join(os.path.dirname(__file__), "data", "models", "load_predictor.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "load_predictor.pkl")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


_CACHED_MODEL = None


def load_model():
    """
    Load the serialized model or train a lightweight fallback model.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    path = find_model_path()
    if path and os.path.exists(path):
        try:
            _CACHED_MODEL = joblib.load(path)
            return _CACHED_MODEL
        except Exception:
            pass

    # If model file is absent, train one seamlessly
    try:
        from core.ai.train_model import train_model
        _CACHED_MODEL = train_model()
        return _CACHED_MODEL
    except Exception:
        return None


def predict_future_load(
    current_load: float,
    network_latency: float,
    queue_length: int,
    cpu_capacity: float,
    energy_usage: float
) -> float:
    """
    Predict future CPU load (%) for an individual edge node using ML with heuristic fallback.
    """
    model = load_model()

    if model is not None:
        try:
            input_df = pd.DataFrame(
                [[
                    float(current_load),
                    float(network_latency),
                    int(queue_length),
                    float(cpu_capacity),
                    float(energy_usage)
                ]],
                columns=[
                    "Current_Load",
                    "Network_Latency",
                    "Queue_Length",
                    "CPU_Capacity",
                    "Energy_Usage"
                ]
            )
            prediction = float(model.predict(input_df)[0])
            bounded = max(0.0, min(100.0, prediction))
            return round(bounded, 2)
        except Exception:
            pass

    # Heuristic fallback estimation
    load_component = float(current_load) * 0.55
    queue_component = float(queue_length) * 2.0
    latency_component = float(network_latency) * 0.25
    energy_component = float(energy_usage) * 0.03
    cap_discount = float(cpu_capacity) * 0.08

    fallback_pred = load_component + queue_component + latency_component + energy_component - cap_discount
    bounded_fallback = max(0.0, min(100.0, fallback_pred))
    return round(bounded_fallback, 2)


def predict_node_load(node: pd.Series) -> float:
    """
    Convenience helper to predict future load from a pandas Series node record.
    """
    return predict_future_load(
        current_load=node.get("Current_Load", 50),
        network_latency=node.get("Network_Latency", 15),
        queue_length=node.get("Queue_Length", 2),
        cpu_capacity=node.get("CPU_Capacity", 100),
        energy_usage=node.get("Energy_Usage", 150)
    )


def predict_all_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ML predicted load for all nodes in the cluster and return enriched DataFrame.
    """
    if nodes.empty:
        return nodes.copy()

    df = nodes.copy()
    predictions: List[float] = []

    for _, node_row in df.iterrows():
        pred = predict_node_load(node_row)
        predictions.append(pred)

    df["Predicted_Load"] = predictions
    return df


if __name__ == "__main__":
    test_load = predict_future_load(45, 12, 3, 120, 150)
    print(f"Predicted Future CPU Load: {test_load}%")