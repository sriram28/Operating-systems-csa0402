# core/load_balancer.py
"""
Load Balancing and Node Health Analysis for Edge Computing Clusters.
"""

from typing import Optional
import numpy as np
import pandas as pd


def get_overloaded_nodes(nodes: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    """
    Identify online edge nodes exceeding the CPU load threshold.
    """
    if nodes.empty:
        return pd.DataFrame()
    return nodes[
        (nodes["Status"] == "Online") &
        (nodes["Current_Load"] >= threshold)
    ].copy()


def get_best_available_node(nodes: pd.DataFrame, exclude_node: Optional[str] = None) -> Optional[pd.Series]:
    """
    Find the best underloaded online node for task offloading/migration.
    """
    if nodes.empty:
        return None

    available = nodes[
        (nodes["Status"] == "Online") &
        (nodes["Current_Load"] < 80)
    ].copy()

    if exclude_node is not None and not available.empty:
        available = available[available["Node_ID"] != exclude_node]

    if available.empty:
        # Fallback to least loaded online node
        online_nodes = nodes[nodes["Status"] == "Online"].copy()
        if exclude_node is not None and not online_nodes.empty:
            online_nodes = online_nodes[online_nodes["Node_ID"] != exclude_node]
        if online_nodes.empty:
            return None
        return online_nodes.sort_values("Current_Load").iloc[0]

    # Calculate balance suitability score: 60% Load + 40% Latency
    available["Balance_Score"] = (
        available["Current_Load"] * 0.60 +
        available["Network_Latency"] * 0.40
    )

    sorted_avail = available.sort_values("Balance_Score")
    return sorted_avail.iloc[0]


def calculate_load_variance(nodes: pd.DataFrame) -> float:
    """
    Compute variance of current load across all online edge nodes.
    Lower variance indicates better load balancing.
    """
    if nodes.empty:
        return 0.0
    online_nodes = nodes[nodes["Status"] == "Online"]
    if online_nodes.empty:
        return 0.0
    return float(np.var(online_nodes["Current_Load"]))