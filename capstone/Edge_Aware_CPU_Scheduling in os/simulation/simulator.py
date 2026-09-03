# simulation/simulator.py
"""
Re-export simulator functions for top-level simulation package.
"""

from core.simulation.simulator import (
    fcfs,
    sjf,
    priority_scheduling,
    round_robin,
    calculate_metrics,
    run_simulation
)

__all__ = [
    "fcfs",
    "sjf",
    "priority_scheduling",
    "round_robin",
    "calculate_metrics",
    "run_simulation"
]
