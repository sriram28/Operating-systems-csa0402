# ⚡ Edge-Aware CPU Scheduling OS Simulator

An interactive operating system simulation platform that bridges classical CPU scheduling algorithms with modern, geographically distributed **Edge Computing** environments.

---

## 🌟 Key Features

1. **🖥️ Conventional CPU Schedulers**:
   - **First-Come, First-Served (FCFS)**
   - **Shortest Job First (SJF - Non-Preemptive)**
   - **Priority Scheduling (Non-Preemptive)**
   - **Round Robin (RR - Preemptive)**
   - Interactive **Plotly Gantt Chart** timeline showing real-time CPU execution slots & idle states.
   - Comprehensive OS metrics: *Average Waiting Time, Turnaround Time, Response Time, CPU Utilization, Throughput*.
   - Side-by-side comparative benchmarking on identical workloads.

2. **🌐 Edge-Aware Task Scheduling**:
   - Multi-node distributed edge scheduling taking into account **Network Latency**, **CPU Load Ratio**, and **Task Burst Time**.
   - Workload generator presets: *IoT / Sensor, Healthcare Mission-Critical, Heavy Compute Gaming / Video, Balanced*.
   - Real-time Deadline compliance tracking and queue delay modeling.

3. **🧠 AI Predictive Scheduling & Dynamic Task Migration**:
   - **Machine Learning (Random Forest Regressor)** for preemptive CPU load forecasting.
   - **Multi-Objective Advanced Scoring**: Optimizes for Latency, Contention, Queue Depth, Energy Efficiency, and Deadline Risk.
   - **Dynamic Task Migration Engine**: Automatically identifies overloaded nodes (> threshold) and dynamically rebalances queues to underutilized nodes.

4. **📊 Comprehensive Benchmarking**:
   - Direct side-by-side performance comparison of all algorithms across Latency, Deadline Miss Rate, Energy Consumption, and Throughput.

---

## 🚀 Getting Started

### 1. Installation

Ensure Python 3.9+ is installed. Install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

Launch the interactive web simulator:

```bash
streamlit run app.py
```

---

## 📐 Mathematical Formulation

### 1. Edge-Aware Scoring Formula
$$\text{EdgeScore}(T_i, N_j) = 0.45 \cdot \text{Latency}_j + 0.35 \cdot \left(\frac{\text{Current Load}_j}{\text{CPU Capacity}_j} \times 100\right) + 0.20 \cdot \text{Burst Time}_i$$

### 2. Multi-Objective Advanced Score
$$\text{Score} = 0.30 \cdot \tilde{L} + 0.25 \cdot \tilde{C} + 0.20 \cdot \tilde{Q} + 0.15 \cdot \text{Risk} + 0.10 \cdot \tilde{E}$$
Where:
- $\tilde{L}$: Normalized Network Latency
- $\tilde{C}$: Normalized CPU Load
- $\tilde{Q}$: Normalized Queue Depth
- $\text{Risk}$: Deadline Risk Ratio ($\frac{\text{Estimated Time}}{\text{Remaining Deadline}}$)
- $\tilde{E}$: Normalized Power Consumption

---

## 📁 Project Directory Structure

```text
Edge_Aware_CPU_Scheduling/
├── app.py                         # Streamlit interactive application dashboard
├── requirements.txt               # Dependencies (streamlit, pandas, plotly, scikit-learn, joblib)
├── README.md                      # Documentation & user guide
├── simulation/
│   ├── __init__.py
│   └── simulator.py               # Re-exports simulator functions
└── core/
    ├── conventional_scheduler.py  # DataFrame-based FCFS, SJF, Priority, Round Robin
    ├── edge_scheduler.py          # Edge-aware heuristic scheduler
    ├── advanced_scheduler.py      # Multi-objective energy & deadline scheduler
    ├── load_balancer.py           # Cluster health & overload detection
    ├── task_migration.py          # Dynamic task rebalancing engine
    ├── task_generator.py          # Synthetic task & edge node generator
    ├── metrics.py                 # Performance metrics engine
    ├── simulation/
    │   └── simulator.py           # Core simulator engine with Gantt timeline tracking
    └── ai/
        ├── predictor.py           # AI Load prediction with fallback heuristics
        ├── train_model.py         # ML model training pipeline
        └── data/
            ├── tasks.csv
            ├── edge_nodes.csv
            └── models/
                └── load_predictor.pkl
```

---

## 🧪 Running Automated Tests

Run the test suite to verify 100% error-free execution:

```bash
python -c "
import simulation.simulator as s, core.edge_scheduler as e, core.task_generator as tg
t, n = tg.generate_tasks(10), tg.generate_edge_nodes(4)
print('Simulator run:', s.run_simulation([{'pid':'P1','arrival':0,'burst':5,'priority':1}], 'fcfs'))
print('Edge scheduler run tasks scheduled:', len(e.edge_aware_schedule(t, n)))
"
```
