# core/task_generator.py
"""
Task and Edge Node Generator for Edge Computing Simulations.
Provides synthetic and preset workload generators with automatic data persistence.
"""

import os
import random
from typing import Tuple, List, Optional
import pandas as pd

# Supported application workload categories
TASK_TYPES = ["IoT", "Video", "Healthcare", "Gaming", "Sensor"]


def generate_tasks(
    num_tasks: int = 30,
    workload_type: str = "Balanced",
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic task dataset with realistic attributes.
    """
    if seed is not None:
        random.seed(seed)

    tasks = []

    for i in range(1, num_tasks + 1):
        if workload_type == "IoT / Sensor":
            burst_time = random.randint(1, 6)
            data_size = random.randint(20, 150)
            slack = random.randint(5, 15)
            priority = random.choice([2, 3, 4])
            t_type = random.choice(["IoT", "Sensor"])

        elif workload_type == "Healthcare / Mission-Critical":
            burst_time = random.randint(3, 10)
            data_size = random.randint(50, 300)
            slack = random.randint(3, 10)  # Very tight deadline
            priority = 1  # Highest priority
            t_type = "Healthcare"

        elif workload_type == "Video / Gaming (Heavy Compute)":
            burst_time = random.randint(8, 20)
            data_size = random.randint(200, 800)
            slack = random.randint(12, 30)
            priority = random.choice([2, 3, 4, 5])
            t_type = random.choice(["Video", "Gaming"])

        else:  # Balanced / Mixed
            burst_time = random.randint(2, 14)
            data_size = random.randint(50, 500)
            slack = random.randint(5, 20)
            priority = random.randint(1, 5)
            t_type = random.choice(TASK_TYPES)

        arrival_time = random.randint(0, max(5, num_tasks))
        deadline = arrival_time + burst_time + slack

        tasks.append({
            "Task_ID": f"T{i}",
            "Arrival_Time": arrival_time,
            "Burst_Time": burst_time,
            "Priority": priority,
            "Deadline": deadline,
            "Data_Size": data_size,
            "Task_Type": t_type
        })

    df = pd.DataFrame(tasks).sort_values(by=["Arrival_Time", "Task_ID"])
    return df.reset_index(drop=True)


def generate_edge_nodes(
    num_nodes: int = 5,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate edge nodes with capacity, latency, load, and energy profiles.
    """
    if seed is not None:
        random.seed(seed)

    nodes = []
    locations = ["Edge-Hub-A (Urban)", "Edge-Hub-B (Metro)", "Edge-Hub-C (Suburb)", "Edge-Hub-D (Rural)", "Edge-Hub-E (Mobile)"]

    for i in range(1, num_nodes + 1):
        loc = locations[(i - 1) % len(locations)]
        nodes.append({
            "Node_ID": f"Edge Node {i}",
            "Location": loc,
            "CPU_Capacity": random.randint(80, 160),
            "Current_Load": random.randint(15, 65),
            "Network_Latency": random.randint(5, 35),
            "Queue_Length": random.randint(0, 5),
            "Energy_Usage": random.randint(80, 240),
            "Status": "Online" if random.random() > 0.1 else "Online"  # default online
        })

    return pd.DataFrame(nodes)


def save_data(tasks: pd.DataFrame, nodes: pd.DataFrame, target_dir: str = "data") -> None:
    """
    Save tasks and nodes to CSV files in the target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    tasks.to_csv(os.path.join(target_dir, "tasks.csv"), index=False)
    nodes.to_csv(os.path.join(target_dir, "edge_nodes.csv"), index=False)


def find_existing_data_file(filename: str) -> Optional[str]:
    """
    Locate a data file across standard search directories.
    """
    candidate_paths = [
        os.path.join("data", filename),
        os.path.join("core", "ai", "data", filename),
        os.path.join(os.path.dirname(__file__), "ai", "data", filename),
        os.path.join(os.path.dirname(__file__), "..", "data", filename)
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None


def load_or_generate_data(
    num_tasks: int = 30,
    num_nodes: int = 5,
    force_generate: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from disk if present, or generate fresh dataset.
    """
    tasks_path = find_existing_data_file("tasks.csv")
    nodes_path = find_existing_data_file("edge_nodes.csv")

    if not force_generate and tasks_path and nodes_path:
        try:
            tasks = pd.read_csv(tasks_path)
            nodes = pd.read_csv(nodes_path)
            return tasks, nodes
        except Exception:
            pass

    tasks = generate_tasks(num_tasks)
    nodes = generate_edge_nodes(num_nodes)
    save_data(tasks, nodes)
    return tasks, nodes


if __name__ == "__main__":
    t, n = load_or_generate_data()
    print(f"Generated {len(t)} tasks and {len(n)} edge nodes successfully.")