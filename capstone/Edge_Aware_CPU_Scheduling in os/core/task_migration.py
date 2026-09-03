# core/task_migration.py
"""
Dynamic Task Migration Engine for Edge Computing.
Rebalances overloaded nodes by migrating active/queued tasks to underutilized nodes.
"""

from typing import Tuple
import pandas as pd
from core.load_balancer import (
    get_overloaded_nodes,
    get_best_available_node
)


def migrate_tasks(
    assignments: pd.DataFrame,
    nodes: pd.DataFrame,
    threshold: float = 80.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Migrate tasks away from overloaded nodes to balanced target nodes.
    Returns:
        (updated_assignments_df, migrations_log_df, updated_nodes_df)
    """
    if assignments.empty or nodes.empty:
        return (
            assignments.copy(),
            pd.DataFrame(columns=["Task_ID", "From_Node", "To_Node", "Reason"]),
            nodes.copy()
        )

    updated_assignments = assignments.copy()
    updated_nodes = nodes.copy()

    # Ensure numeric columns can store floats safely
    for col in ["Current_Load", "Queue_Length", "Network_Latency", "CPU_Capacity", "Energy_Usage"]:
        if col in updated_nodes.columns:
            updated_nodes[col] = updated_nodes[col].astype(float)

    migrations = []

    overloaded = get_overloaded_nodes(updated_nodes, threshold=threshold)

    for _, over_node in overloaded.iterrows():
        node_id = over_node["Node_ID"]
        node_tasks = updated_assignments[
            updated_assignments["Edge_Node"] == node_id
        ]

        if node_tasks.empty:
            continue

        # Migrate half of the assigned tasks to reduce contention
        tasks_to_move_count = max(1, len(node_tasks) // 2)
        tasks_to_move = node_tasks.tail(tasks_to_move_count)

        for idx, task_row in tasks_to_move.iterrows():
            best_node = get_best_available_node(
                updated_nodes,
                exclude_node=node_id
            )

            if best_node is not None:
                new_node_id = best_node["Node_ID"]
                old_node_id = updated_assignments.loc[idx, "Edge_Node"]

                # Perform migration in assignment table
                updated_assignments.loc[idx, "Edge_Node"] = new_node_id

                # Update loads
                over_node_idx = updated_nodes[updated_nodes["Node_ID"] == old_node_id].index
                target_node_idx = updated_nodes[updated_nodes["Node_ID"] == new_node_id].index

                if not over_node_idx.empty:
                    updated_nodes.loc[over_node_idx[0], "Current_Load"] = float(max(
                        10.0,
                        float(updated_nodes.loc[over_node_idx[0], "Current_Load"]) - 15.0
                    ))
                    updated_nodes.loc[over_node_idx[0], "Queue_Length"] = float(max(
                        0.0,
                        float(updated_nodes.loc[over_node_idx[0], "Queue_Length"]) - 1.0
                    ))

                if not target_node_idx.empty:
                    updated_nodes.loc[target_node_idx[0], "Current_Load"] = float(min(
                        95.0,
                        float(updated_nodes.loc[target_node_idx[0], "Current_Load"]) + 12.0
                    ))
                    updated_nodes.loc[target_node_idx[0], "Queue_Length"] = float(
                        float(updated_nodes.loc[target_node_idx[0], "Queue_Length"]) + 1.0
                    )

                migrations.append({
                    "Task_ID": task_row["Task_ID"],
                    "From_Node": old_node_id,
                    "To_Node": new_node_id,
                    "Reason": f"Overload Mitigation (> {threshold}% CPU Load)"
                })

    return updated_assignments, pd.DataFrame(migrations), updated_nodes