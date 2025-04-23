import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Page config
st.set_page_config(page_title="Lotto Bayesian Network", layout="centered")
st.title("🎲 Lotto Bayesian Network Simulator")
st.markdown("""
This app simulates a **biased Bayesian Network** for Lotto numbers (1–52).
- **7 Lotto Balls** (6 normal + 1 bonus).
- **Constructive bias** added to each ball.
- Bias favors lower numbers (simulating real-world perception).
""")

# --- DAG Construction ---

# Define nodes and create uniform DAG edges
balls = [f"Ball_{i}" for i in range(1, 8)]
edges = [(balls[i], balls[i+1]) for i in range(len(balls) - 1)]

# Build Bayesian Network
model = BayesianNetwork(edges)

# Function to generate a biased distribution favoring lower numbers
def get_biased_distribution():
    raw = np.array([1 / (i+1) for i in range(52)])  # inverse bias: smaller numbers more probable
    return list(raw / raw.sum())  # normalize

# Create CPDs with constructive bias for each ball
cpds = []
for ball in balls:
    cpd = TabularCPD(variable=ball, variable_card=52, values=[get_biased_distribution()])
    cpds.append(cpd)
    model.add_cpds(cpd)

# Validate model
if model.check_model():
    st.success("✅ Bayesian Network with CPDs created and validated successfully!")
else:
    st.error("❌ Model validation failed.")

# --- Visualization ---

# Build NetworkX graph for drawing
G = nx.DiGraph()
G.add_edges_from(edges)

pos = nx.spring_layout(G, seed=42)  # consistent layout

fig, ax = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_weight='bold', font_size=12, arrows=True, ax=ax)
ax.set_title("Lotto Bayesian Network DAG", fontsize=16)
st.pyplot(fig)

# --- Simulate Draw ---
if st.button("🎰 Simulate Lotto Draw"):
    st.subheader("🔢 Simulated Lotto Draw:")
    draw = [np.random.choice(range(1, 53), p=get_biased_distribution()) for _ in range(7)]
    st.write("Your numbers:", draw)
