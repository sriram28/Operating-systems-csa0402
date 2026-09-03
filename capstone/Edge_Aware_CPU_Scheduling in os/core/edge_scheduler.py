# core/edge_scheduler.py
"""
Edge-Aware Task Scheduler.
Distributes tasks across geographically distributed Edge Nodes based on:
1. Network Latency
2. Real-time Node CPU Load & Capacity
3. Task Burst Time and Deadlines
"""

import pandas as pd


def calculate_edge_score(task: pd.Series, node: pd.Series) -> float:
    """
    Calculate edge assignment suitability score (Lower score = Better choice).
    Considers network latency, current CPU load ratio, and task burst requirements.
    """
    cpu_cap = max(1.0, float(node.get("CPU_Capacity", 100)))
    curr_load = max(0.0, float(node.get("Current_Load", 0)))
    load_ratio = curr_load / cpu_cap

    latency = max(0.0, float(node.get("Network_Latency", 10)))
    burst = max(1.0, float(task.get("Burst_Time", 5)))

    # Weighted scoring formula:
    # 45% Network Latency + 35% Current CPU Load Percentage + 20% Task Burst Time
    score = (
        0.45 * latency +
        0.35 * (load_ratio * 100.0) +
        0.20 * burst
    )
    return float(score)


def edge_aware_schedule(tasks: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Schedule a batch of tasks onto edge nodes using Edge-Aware heuristics.
    """
    if tasks.empty or nodes.empty:
        return pd.DataFrame(columns=[
            "Task_ID", "Edge_Node", "Edge_Score", "Network_Latency", "Burst_Time",
            "Queue_Delay", "Total_Latency", "Deadline", "Deadline_Missed"
        ])

    working_nodes = nodes.copy().reset_index(drop=True)
    
    # Ensure numerical columns can hold floating point values safely
    for col in ["Current_Load", "Queue_Length", "Network_Latency", "CPU_Capacity", "Energy_Usage"]:
        if col in working_nodes.columns:
            working_nodes[col] = working_nodes[col].astype(float)

    results = []

    for _, task in tasks.iterrows():
        # Find active online nodes
        online_mask = working_nodes["Status"] == "Online"
        online_indices = working_nodes[online_mask].index

        if len(online_indices) == 0:
            # Fallback if all nodes are marked offline
            online_indices = working_nodes.index

        # Evaluate candidate nodes
        best_idx = None
        best_score = float("inf")

        for idx in online_indices:
            node_row = working_nodes.loc[idx]
            score = calculate_edge_score(task, node_row)
            if score < best_score:
                best_score = score
                best_idx = idx

        selected_node = working_nodes.loc[best_idx]
        net_latency = float(selected_node.get("Network_Latency", 10))
        burst_time = float(task.get("Burst_Time", 5))
        curr_load = float(selected_node.get("Current_Load", 0))
        queue_len = float(selected_node.get("Queue_Length", 0))

        # Queue delay estimation based on pending queue and current load
        queue_delay = (queue_len * 1.5) + (curr_load / 15.0)
        total_latency = net_latency + burst_time + queue_delay

        arrival = float(task.get("Arrival_Time", 0))
        deadline = float(task.get("Deadline", arrival + burst_time + 20))
        completion_time = arrival + total_latency
        deadline_missed = completion_time > deadline

        results.append({
            "Task_ID": task["Task_ID"],
            "Edge_Node": selected_node["Node_ID"],
            "Edge_Score": round(best_score, 2),
            "Network_Latency": round(net_latency, 2),
            "Burst_Time": round(burst_time, 2),
            "Queue_Delay": round(queue_delay, 2),
            "Total_Latency": round(total_latency, 2),
            "Arrival_Time": round(arrival, 2),
            "Completion_Time": round(completion_time, 2),
            "Deadline": round(deadline, 2),
            "Deadline_Missed": bool(deadline_missed)
        })

        # Update node state
        working_nodes.loc[best_idx, "Current_Load"] = float(min(
            100.0,
            curr_load + (burst_time * 1.2)
        ))
        working_nodes.loc[best_idx, "Queue_Length"] = float(queue_len + 1)

    return pd.DataFrame(results)