# core/advanced_scheduler.py
"""
Advanced Multi-Objective AI/Heuristic Edge Scheduler.
Optimizes for:
- Latency minimization
- Load balancing across edge nodes
- Queue wait reduction
- Deadline compliance & starvation prevention
- Energy efficiency
"""

from typing import Tuple
import pandas as pd


def calculate_advanced_score(task: pd.Series, node: pd.Series) -> Tuple[float, float, float]:
    """
    Compute multi-factor assignment score for a task on a candidate edge node.
    Returns (composite_score, estimated_execution_time, estimated_energy).
    """
    net_latency = max(1.0, float(node.get("Network_Latency", 10)))
    curr_load = max(0.0, float(node.get("Current_Load", 0)))
    queue_len = max(0.0, float(node.get("Queue_Length", 0)))
    energy_rate = max(10.0, float(node.get("Energy_Usage", 100)))
    burst = max(1.0, float(task.get("Burst_Time", 5)))
    arrival = max(0.0, float(task.get("Arrival_Time", 0)))
    deadline = max(arrival + burst + 1, float(task.get("Deadline", arrival + burst + 20)))

    # Normalized component terms (0 to ~1 scale)
    latency_term = net_latency / 40.0
    load_term = curr_load / 100.0
    queue_term = queue_len / 10.0
    energy_term = energy_rate / 300.0

    # Estimated latency: Network + Execution + Queue wait
    queue_wait = queue_len * 1.8 + (curr_load / 20.0)
    estimated_time = net_latency + burst + queue_wait

    remaining_deadline = max(1.0, deadline - arrival)
    deadline_risk = estimated_time / remaining_deadline

    # Energy cost: Power rate * execution time / 100
    estimated_energy = (energy_rate * (burst + queue_wait * 0.3)) / 100.0

    # Multi-factor weighted composite score (Lower = Better)
    score = (
        0.30 * latency_term +
        0.25 * load_term +
        0.20 * queue_term +
        0.15 * deadline_risk +
        0.10 * energy_term
    )

    return float(score), float(estimated_time), float(estimated_energy)


def advanced_edge_schedule(tasks: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced Multi-Objective Scheduling with overload prevention and energy awareness.
    """
    if tasks.empty or nodes.empty:
        return pd.DataFrame(columns=[
            "Task_ID", "Edge_Node", "Advanced_Score", "Estimated_Latency",
            "Energy_Consumed", "Arrival_Time", "Completion_Time", "Deadline", "Deadline_Missed"
        ])

    working_nodes = nodes.copy().reset_index(drop=True)

    # Ensure numerical columns can hold floating point values safely
    for col in ["Current_Load", "Queue_Length", "Network_Latency", "CPU_Capacity", "Energy_Usage"]:
        if col in working_nodes.columns:
            working_nodes[col] = working_nodes[col].astype(float)

    results = []

    for _, task in tasks.iterrows():
        # First preference: Online nodes with load < 95%
        avail_mask = (working_nodes["Status"] == "Online") & (working_nodes["Current_Load"] < 95)
        avail_indices = working_nodes[avail_mask].index

        if len(avail_indices) == 0:
            # Second preference: Any Online node
            avail_mask = working_nodes["Status"] == "Online"
            avail_indices = working_nodes[avail_mask].index

        if len(avail_indices) == 0:
            # Fallback to any node in pool
            avail_indices = working_nodes.index

        best_idx = None
        best_score = float("inf")
        best_time = 0.0
        best_energy = 0.0

        for idx in avail_indices:
            node_row = working_nodes.loc[idx]
            score, est_time, est_energy = calculate_advanced_score(task, node_row)
            if score < best_score:
                best_score = score
                best_idx = idx
                best_time = est_time
                best_energy = est_energy

        selected_node = working_nodes.loc[best_idx]
        arrival = float(task.get("Arrival_Time", 0))
        deadline = float(task.get("Deadline", arrival + best_time + 20))
        completion_time = arrival + best_time
        deadline_missed = completion_time > deadline

        results.append({
            "Task_ID": task["Task_ID"],
            "Edge_Node": selected_node["Node_ID"],
            "Advanced_Score": round(best_score, 3),
            "Estimated_Latency": round(best_time, 2),
            "Energy_Consumed": round(best_energy, 2),
            "Arrival_Time": round(arrival, 2),
            "Completion_Time": round(completion_time, 2),
            "Deadline": round(deadline, 2),
            "Deadline_Missed": bool(deadline_missed)
        })

        burst = float(task.get("Burst_Time", 5))
        working_nodes.loc[best_idx, "Current_Load"] = float(min(
            100.0,
            float(selected_node.get("Current_Load", 0)) + (burst * 1.5)
        ))
        working_nodes.loc[best_idx, "Queue_Length"] = float(selected_node.get("Queue_Length", 0)) + 1.0

    return pd.DataFrame(results)