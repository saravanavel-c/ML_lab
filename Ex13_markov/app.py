import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

st.title("Weather Prediction using Markov Model")

# -------------------------------
# 1 Load Dataset
# -------------------------------

st.header("1 Dataset Preview")

df = pd.read_csv("../Datasets/weatherHistory.csv")

st.write("Dataset Shape:", df.shape)

st.dataframe(df.head())

# -------------------------------
# 2 Prepare Sequence Data
# -------------------------------

st.header("2 Preparing Markov States")

weather_seq = df["Summary"].astype(str)

# reduce categories (top 6 only)
top_states = weather_seq.value_counts().head(6).index

weather_seq = weather_seq[weather_seq.isin(top_states)]

states = weather_seq.unique()

state_to_idx = {s:i for i,s in enumerate(states)}
idx_to_state = {i:s for s,i in state_to_idx.items()}

sequence = weather_seq.map(state_to_idx).values

st.write("States Used:", list(states))

# -------------------------------
# 3 Build Transition Matrix
# -------------------------------

st.header("3 Training Markov Model")

n_states = len(states)

transition_matrix = np.zeros((n_states,n_states))

for i in range(len(sequence)-1):
    transition_matrix[sequence[i],sequence[i+1]] += 1

# normalize
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

st.write("Transition Matrix")

st.dataframe(pd.DataFrame(
    transition_matrix,
    index=states,
    columns=states
))

# -------------------------------
# 4 Visualize State Transitions
# -------------------------------

st.header("4 State Transition Visualization")

G = nx.DiGraph()

for i in range(n_states):
    for j in range(n_states):
        prob = round(transition_matrix[i][j],2)
        if prob > 0.05:
            G.add_edge(idx_to_state[i], idx_to_state[j], weight=prob)

pos = nx.circular_layout(G)

edge_labels = nx.get_edge_attributes(G,'weight')

fig, ax = plt.subplots(figsize=(8,6))

nx.draw(G, pos, ax=ax, with_labels=True, node_size=2500, node_color="lightblue")

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

st.pyplot(fig)

# -------------------------------
# 5 Prediction
# -------------------------------

st.header("5 Predict Next Weather")

current_state = st.selectbox(
"Select Current Weather Condition",
states
)

if st.button("Predict Next State"):

    idx = state_to_idx[current_state]

    probs = transition_matrix[idx]

    next_state_idx = np.random.choice(len(probs), p=probs)

    predicted = idx_to_state[next_state_idx]

    st.success(f"Predicted Next Weather: {predicted}")

# -------------------------------
# 6 Evaluation
# -------------------------------

st.header("6 Model Evaluation")

correct = 0
total = 0

for i in range(len(sequence)-1):

    current = sequence[i]

    predicted = np.argmax(transition_matrix[current])

    actual = sequence[i+1]

    if predicted == actual:
        correct += 1

    total += 1

accuracy = correct/total

st.write("Prediction Accuracy:", round(accuracy*100,2), "%")