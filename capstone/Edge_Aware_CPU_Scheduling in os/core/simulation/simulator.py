# core/simulation/simulator.py
"""
Robust CPU Scheduling Simulator supporting FCFS, SJF, Priority, and Round Robin.
Includes execution timeline (Gantt chart data) and standard OS performance metrics.
"""

from typing import List, Dict, Any


def fcfs(processes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    First-Come, First-Served (FCFS) Non-preemptive CPU Scheduling.
    """
    sorted_procs = sorted(
        [p.copy() for p in processes],
        key=lambda x: (x["arrival"], x["pid"])
    )

    current_time = 0
    results = []
    timeline = []

    for p in sorted_procs:
        arrival = max(0, int(p["arrival"]))
        burst = max(1, int(p["burst"]))
        pid = str(p["pid"])

        # CPU idle handling
        if current_time < arrival:
            timeline.append({
                "pid": "Idle",
                "start": current_time,
                "end": arrival,
                "duration": arrival - current_time
            })
            current_time = arrival

        start_time = current_time
        completion_time = start_time + burst
        waiting_time = start_time - arrival
        turnaround_time = completion_time - arrival
        response_time = waiting_time

        timeline.append({
            "pid": pid,
            "start": start_time,
            "end": completion_time,
            "duration": burst
        })

        results.append({
            "pid": pid,
            "arrival": arrival,
            "burst": burst,
            "priority": p.get("priority", 1),
            "start": start_time,
            "completion": completion_time,
            "waiting": waiting_time,
            "turnaround": turnaround_time,
            "response": response_time
        })

        current_time = completion_time

    return {
        "results": results,
        "timeline": timeline
    }


def sjf(processes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shortest Job First (SJF) Non-preemptive CPU Scheduling.
    """
    remaining = [
        {
            "pid": str(p["pid"]),
            "arrival": max(0, int(p["arrival"])),
            "burst": max(1, int(p["burst"])),
            "priority": p.get("priority", 1)
        }
        for p in processes
    ]

    current_time = 0
    results = []
    timeline = []

    while remaining:
        available = [
            p for p in remaining
            if p["arrival"] <= current_time
        ]

        if not available:
            next_arrival = min(p["arrival"] for p in remaining)
            timeline.append({
                "pid": "Idle",
                "start": current_time,
                "end": next_arrival,
                "duration": next_arrival - current_time
            })
            current_time = next_arrival
            continue

        # Select process with shortest burst time, breaking ties by arrival time
        selected = min(
            available,
            key=lambda x: (x["burst"], x["arrival"], x["pid"])
        )

        start_time = current_time
        completion_time = start_time + selected["burst"]
        waiting_time = start_time - selected["arrival"]
        turnaround_time = completion_time - selected["arrival"]
        response_time = waiting_time

        timeline.append({
            "pid": selected["pid"],
            "start": start_time,
            "end": completion_time,
            "duration": selected["burst"]
        })

        results.append({
            "pid": selected["pid"],
            "arrival": selected["arrival"],
            "burst": selected["burst"],
            "priority": selected.get("priority", 1),
            "start": start_time,
            "completion": completion_time,
            "waiting": waiting_time,
            "turnaround": turnaround_time,
            "response": response_time
        })

        current_time = completion_time
        remaining.remove(selected)

    results.sort(key=lambda x: x["pid"])

    return {
        "results": results,
        "timeline": timeline
    }


def priority_scheduling(processes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Non-preemptive Priority Scheduling (Lower value = Higher Priority).
    """
    remaining = [
        {
            "pid": str(p["pid"]),
            "arrival": max(0, int(p["arrival"])),
            "burst": max(1, int(p["burst"])),
            "priority": int(p.get("priority", 1))
        }
        for p in processes
    ]

    current_time = 0
    results = []
    timeline = []

    while remaining:
        available = [
            p for p in remaining
            if p["arrival"] <= current_time
        ]

        if not available:
            next_arrival = min(p["arrival"] for p in remaining)
            timeline.append({
                "pid": "Idle",
                "start": current_time,
                "end": next_arrival,
                "duration": next_arrival - current_time
            })
            current_time = next_arrival
            continue

        # Select highest priority (lowest numerical priority value)
        selected = min(
            available,
            key=lambda x: (x["priority"], x["arrival"], x["pid"])
        )

        start_time = current_time
        completion_time = start_time + selected["burst"]
        waiting_time = start_time - selected["arrival"]
        turnaround_time = completion_time - selected["arrival"]
        response_time = waiting_time

        timeline.append({
            "pid": selected["pid"],
            "start": start_time,
            "end": completion_time,
            "duration": selected["burst"]
        })

        results.append({
            "pid": selected["pid"],
            "arrival": selected["arrival"],
            "burst": selected["burst"],
            "priority": selected["priority"],
            "start": start_time,
            "completion": completion_time,
            "waiting": waiting_time,
            "turnaround": turnaround_time,
            "response": response_time
        })

        current_time = completion_time
        remaining.remove(selected)

    results.sort(key=lambda x: x["pid"])

    return {
        "results": results,
        "timeline": timeline
    }


def round_robin(processes: List[Dict[str, Any]], quantum: int = 2) -> Dict[str, Any]:
    """
    Preemptive Round Robin CPU Scheduling with Time Quantum.
    """
    if quantum <= 0:
        quantum = 2

    proc_list = sorted(
        [
            {
                "pid": str(p["pid"]),
                "arrival": max(0, int(p["arrival"])),
                "burst": max(1, int(p["burst"])),
                "priority": int(p.get("priority", 1)),
                "remaining": max(1, int(p["burst"])),
                "first_start": None,
                "completion": 0
            }
            for p in processes
        ],
        key=lambda x: (x["arrival"], x["pid"])
    )

    current_time = 0
    index = 0
    queue = []
    timeline = []
    completed_map = {}

    while index < len(proc_list) or queue:
        # Enqueue newly arrived processes
        while index < len(proc_list) and proc_list[index]["arrival"] <= current_time:
            queue.append(proc_list[index])
            index += 1

        if not queue:
            if index < len(proc_list):
                idle_start = current_time
                idle_end = proc_list[index]["arrival"]
                timeline.append({
                    "pid": "Idle",
                    "start": idle_start,
                    "end": idle_end,
                    "duration": idle_end - idle_start
                })
                current_time = idle_end
            continue

        current_proc = queue.pop(0)
        pid = current_proc["pid"]

        if current_proc["first_start"] is None:
            current_proc["first_start"] = current_time

        exec_time = min(quantum, current_proc["remaining"])
        start_t = current_time
        current_time += exec_time
        current_proc["remaining"] -= exec_time

        timeline.append({
            "pid": pid,
            "start": start_t,
            "end": current_time,
            "duration": exec_time
        })

        # Add any newly arrived processes during this time slice
        while index < len(proc_list) and proc_list[index]["arrival"] <= current_time:
            queue.append(proc_list[index])
            index += 1

        # Requeue or mark complete
        if current_proc["remaining"] > 0:
            queue.append(current_proc)
        else:
            current_proc["completion"] = current_time
            completed_map[pid] = current_proc

    results = []
    for p in proc_list:
        c_proc = completed_map[p["pid"]]
        completion = c_proc["completion"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = (c_proc["first_start"] if c_proc["first_start"] is not None else p["arrival"]) - p["arrival"]

        results.append({
            "pid": p["pid"],
            "arrival": p["arrival"],
            "burst": p["burst"],
            "priority": p["priority"],
            "start": c_proc["first_start"],
            "completion": completion,
            "waiting": max(0, waiting),
            "turnaround": max(0, turnaround),
            "response": max(0, response)
        })

    results.sort(key=lambda x: x["pid"])

    return {
        "results": results,
        "timeline": timeline
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate summary performance metrics for scheduling results.
    """
    if not results:
        return {
            "average_waiting_time": 0.0,
            "average_turnaround_time": 0.0,
            "average_response_time": 0.0,
            "cpu_utilization": 0.0,
            "throughput": 0.0,
            "total_burst_time": 0.0,
            "total_execution_time": 0.0
        }

    n = len(results)
    total_waiting = sum(p["waiting"] for p in results)
    total_turnaround = sum(p["turnaround"] for p in results)
    total_response = sum(p.get("response", p["waiting"]) for p in results)
    total_burst = sum(p["burst"] for p in results)

    first_arrival = min(p["arrival"] for p in results)
    last_completion = max(p["completion"] for p in results)
    total_time = max(1, last_completion - first_arrival)

    cpu_utilization = min(100.0, (total_burst / total_time) * 100.0) if total_time > 0 else 0.0
    throughput = (n / total_time) if total_time > 0 else 0.0

    return {
        "average_waiting_time": round(total_waiting / n, 2),
        "average_turnaround_time": round(total_turnaround / n, 2),
        "average_response_time": round(total_response / n, 2),
        "cpu_utilization": round(cpu_utilization, 2),
        "throughput": round(throughput, 4),
        "total_burst_time": round(float(total_burst), 2),
        "total_execution_time": round(float(total_time), 2)
    }


def run_simulation(processes: List[Dict[str, Any]], algorithm: str, quantum: int = 2) -> Dict[str, Any]:
    """
    Main simulator entrypoint called by app.py and CLI scripts.
    """
    if not processes:
        raise ValueError("No processes were provided for simulation.")

    clean_algo = str(algorithm).strip().lower()

    if clean_algo in ["fcfs", "first come first serve"]:
        sim_data = fcfs(processes)
    elif clean_algo in ["sjf", "shortest job first"]:
        sim_data = sjf(processes)
    elif clean_algo in ["priority", "priority scheduling"]:
        sim_data = priority_scheduling(processes)
    elif clean_algo in ["round robin", "round_robin", "rr"]:
        sim_data = round_robin(processes, quantum=quantum)
    else:
        raise ValueError(f"Unknown scheduling algorithm: '{algorithm}'. Supported: FCFS, SJF, Priority, Round Robin.")

    results = sim_data["results"]
    timeline = sim_data.get("timeline", [])
    metrics = calculate_metrics(results)

    return {
        "algorithm": algorithm,
        "results": results,
        "timeline": timeline,
        "metrics": metrics
    }