# 🚕 Taxi Navigation using Q-Learning

## 📌 Project Description

This project demonstrates how a taxi agent learns to navigate a 5×5 grid world using **Reinforcement Learning**. The agent learns to pick up a passenger from one location and drop them off at a specified destination efficiently.

It applies:

* **Q-Learning** → A model-free RL algorithm that builds a Q-table mapping state-action pairs to expected rewards
* **ε-Greedy Policy** → Balances exploration and exploitation during training
* **Bellman Equation** → Used to update Q-values iteratively across episodes

---

## 🗺️ Environment Overview

The project uses the **Taxi-v3** environment from Gymnasium (OpenAI):

* **Grid:** 5×5 world with 4 fixed locations — 🔴 Red, 🟢 Green, 🟡 Yellow, 🔵 Blue
* **States:** 500 possible states (taxi position × passenger location × destination)
* **Actions:** 6 discrete actions — Move South, North, East, West, Pickup, Dropoff
* **Rewards:**
  * `+20` → Correct passenger dropoff
  * `-10` → Incorrect pickup or dropoff attempt
  * `-1` → Every step taken (encourages shortest path)

---

## 🛠️ Requirements

* Python 3.10+
* numpy
* gymnasium
* streamlit


## 📁 Project Structure

```
Assn_2/
├── app.py       # Streamlit UI and simulation logic
├── taxi.py      # Q-learning training logic
└── README.md    # Project documentation
```

---

## ▶️ How to Run the Project

1. Create and activate a dedicated conda environment:

```bash
conda create -n taxi_rl python=3.10
conda activate taxi_rl
```

2. Install dependencies:

```bash
pip install gymnasium streamlit numpy
```

3. Navigate to the project folder:

```bash
cd Assn_2
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the browser at:

```
http://localhost:8501
```

---

## 🎮 How to Use

1. Adjust **Training Episodes** and **Simulation Speed** from the sidebar.
2. Click **🎓 Train Model** — trains the Q-learning agent (only needs to be done once).
3. Click **▶️ Run Simulation** — watches the trained agent navigate the grid.
4. Simulate as many times as you want **without retraining**.

---

## 🎯 Output

The system displays:

* Live taxi grid with custom symbols:
  * `🔍` → Taxi searching for passenger
  * `🚖` → Taxi carrying passenger to destination
* Step-by-step action log (Move North, Pickup, Dropoff, etc.)
* Per-step reward and cumulative total reward
* Passenger pickup and dropoff location (🔴 Red / 🟢 Green / 🟡 Yellow / 🔵 Blue)
* Final success or failure message

---

## ⚙️ Q-Learning Parameters

| Parameter | Value | Description |
|---|---|---|
| Alpha (α) | 0.1 | Learning rate |
| Gamma (γ) | 0.99 | Discount factor |
| Epsilon (ε) | 1.0 → 0.01 | Exploration rate (decays over episodes) |
| Epsilon Decay | 0.995 | Rate at which exploration reduces |