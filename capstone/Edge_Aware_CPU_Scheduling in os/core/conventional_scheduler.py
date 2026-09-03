# core/conventional_scheduler.py
"""
Conventional CPU Scheduling algorithms operating on pandas DataFrames.
Provides FCFS, SJF, Priority, and Round Robin scheduling.
"""

import pandas as pd


def fcfs(tasks: pd.DataFrame) -> pd.DataFrame:
    """
    First Come First Serve (FCFS) Scheduling for Tasks DataFrame.
    """
    if tasks.empty:
        return pd.DataFrame(columns=[
            "Task_ID", "Waiting_Time", "Turnaround_Time", "Response_Time", "Completion_Time"
        ])

    df = tasks.sort_values(by=["Arrival_Time", "Task_ID"]).copy()
    current_time = 0
    results = []

    for _, task in df.iterrows():
        arrival = max(0, float(task["Arrival_Time"]))
        burst = max(1, float(task["Burst_Time"]))

        current_time = max(current_time, arrival)
        start_time = current_time
        completion_time = start_time + burst
        waiting_time = start_time - arrival
        turnaround_time = completion_time - arrival
        response_time = waiting_time

        results.append({
            "Task_ID": task["Task_ID"],
            "Arrival_Time": arrival,
            "Burst_Time": burst,
            "Start_Time": start_time,
            "Completion_Time": completion_time,
            "Waiting_Time": waiting_time,
            "Turnaround_Time": turnaround_time,
            "Response_Time": response_time
        })

        current_time = completion_time

    return pd.DataFrame(results)


def sjf(tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Shortest Job First (SJF) Non-preemptive Scheduling for Tasks DataFrame.
    """
    if tasks.empty:
        return pd.DataFrame(columns=[
            "Task_ID", "Waiting_Time", "Turnaround_Time", "Response_Time", "Completion_Time"
        ])

    remaining = tasks.copy()
    results = []
    current_time = 0

    while not remaining.empty:
        available = remaining[remaining["Arrival_Time"] <= current_time]

        if available.empty:
            current_time = remaining["Arrival_Time"].min()
            continue

        # Sort by Burst_Time, then Arrival_Time as tie-breaker
        sorted_avail = available.sort_values(by=["Burst_Time", "Arrival_Time"])
        selected_index = sorted_avail.index[0]
        task = remaining.loc[selected_index]

        arrival = max(0, float(task["Arrival_Time"]))
        burst = max(1, float(task["Burst_Time"]))

        start_time = current_time
        completion_time = start_time + burst
        waiting_time = start_time - arrival
        turnaround_time = completion_time - arrival
        response_time = waiting_time

        results.append({
            "Task_ID": task["Task_ID"],
            "Arrival_Time": arrival,
            "Burst_Time": burst,
            "Start_Time": start_time,
            "Completion_Time": completion_time,
            "Waiting_Time": waiting_time,
            "Turnaround_Time": turnaround_time,
            "Response_Time": response_time
        })

        current_time = completion_time
        remaining = remaining.drop(selected_index)

    return pd.DataFrame(results)


def priority_scheduling(tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Priority Scheduling (Non-preemptive, lower number = higher priority).
    """
    if tasks.empty:
        return pd.DataFrame(columns=[
            "Task_ID", "Waiting_Time", "Turnaround_Time", "Response_Time", "Completion_Time"
        ])

    priority_col = "Priority" if "Priority" in tasks.columns else None
    remaining = tasks.copy()
    results = []
    current_time = 0

    while not remaining.empty:
        available = remaining[remaining["Arrival_Time"] <= current_time]

        if available.empty:
            current_time = remaining["Arrival_Time"].min()
            continue

        if priority_col:
            sorted_avail = available.sort_values(by=[priority_col, "Arrival_Time"])
        else:
            sorted_avail = available.sort_values(by=["Arrival_Time"])

        selected_index = sorted_avail.index[0]
        task = remaining.loc[selected_index]

        arrival = max(0, float(task["Arrival_Time"]))
        burst = max(1, float(task["Burst_Time"]))
        priority_val = task[priority_col] if priority_col else 1

        start_time = current_time
        completion_time = start_time + burst
        waiting_time = start_time - arrival
        turnaround_time = completion_time - arrival
        response_time = waiting_time

        results.append({
            "Task_ID": task["Task_ID"],
            "Priority": priority_val,
            "Arrival_Time": arrival,
            "Burst_Time": burst,
            "Start_Time": start_time,
            "Completion_Time": completion_time,
            "Waiting_Time": waiting_time,
            "Turnaround_Time": turnaround_time,
            "Response_Time": response_time
        })

        current_time = completion_time
        remaining = remaining.drop(selected_index)

    return pd.DataFrame(results)


def round_robin(tasks: pd.DataFrame, quantum: int = 3) -> pd.DataFrame:
    """
    Round Robin (RR) Preemptive Scheduling for Tasks DataFrame.
    """
    if tasks.empty:
        return pd.DataFrame(columns=[
            "Task_ID", "Waiting_Time", "Turnaround_Time", "Response_Time", "Completion_Time"
        ])

    if quantum <= 0:
        quantum = 3

    task_list = []
    for _, task in tasks.iterrows():
        arrival = max(0, float(task["Arrival_Time"]))
        burst = max(1, float(task["Burst_Time"]))
        task_list.append({
            "Task_ID": task["Task_ID"],
            "Arrival_Time": arrival,
            "Burst_Time": burst,
            "Remaining_Time": burst,
            "First_Start": None
        })

    task_list.sort(key=lambda x: (x["Arrival_Time"], str(x["Task_ID"])))

    current_time = 0
    queue = []
    completed = []

    while task_list or queue:
        while task_list and task_list[0]["Arrival_Time"] <= current_time:
            queue.append(task_list.pop(0))

        if not queue:
            if task_list:
                current_time = task_list[0]["Arrival_Time"]
            continue

        task = queue.pop(0)

        if task["First_Start"] is None:
            task["First_Start"] = current_time

        execution = min(quantum, task["Remaining_Time"])
        current_time += execution
        task["Remaining_Time"] -= execution

        while task_list and task_list[0]["Arrival_Time"] <= current_time:
            queue.append(task_list.pop(0))

        if task["Remaining_Time"] > 0:
            queue.append(task)
        else:
            turnaround = current_time - task["Arrival_Time"]
            waiting = turnaround - task["Burst_Time"]
            response = task["First_Start"] - task["Arrival_Time"]

            completed.append({
                "Task_ID": task["Task_ID"],
                "Arrival_Time": task["Arrival_Time"],
                "Burst_Time": task["Burst_Time"],
                "Start_Time": task["First_Start"],
                "Completion_Time": current_time,
                "Waiting_Time": max(0.0, waiting),
                "Turnaround_Time": max(0.0, turnaround),
                "Response_Time": max(0.0, response)
            })

    return pd.DataFrame(completed)