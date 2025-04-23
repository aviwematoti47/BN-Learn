import streamlit as st
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

# Define nodes and edges (chain structure)
balls = [f"Ball_{i}" for i in range(1, 8)]
edges = [(balls[i], balls[i+1]) for i in range(len(balls) - 1)]

# Initialize Bayesian Network manually
try:
    model = BayesianNetwork(edges)
except ImportError:
    st.error("pgmpy requires optional dependencies (like pygraphviz). Ensure your environment includes `pydot` or `graphviz`.")
    st.stop()

# Biased probability distribution (favor lower numbers)
def get_biased_distribution():
    raw = np.array([1 / (i+1) for i in range(52)])  # i from 0 to 51 → 1 to 52
    return list(raw / raw.sum())

# Define CPDs
cpds = []
for ball in balls:
    dist = get_biased_distribution()
    cpd = TabularCPD(variable=ball, variable_card=52, values=[dist])
    cpds.append(cpd)
    model.add_cpds(cpd)

# Validate model
if model.check_model():
    st.success("✅ Bayesian Network with CPDs created and validated successfully!")
else:
    st.error("❌ Model validation failed.")
    st.stop()

# Visualize the DAG
st.subheader("📊 Lotto DAG Structure")
G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True, ax=ax)
ax.set_title("Lotto Bayesian Network DAG")
st.pyplot(fig)

# Simulate lotto draw
st.subheader("🎰 Simulate Lotto Draw")
if st.button("Draw Numbers"):
    draw = [np.random.choice(range(1, 53), p=get_biased_distribution()) for _ in range(7)]
    st.success(f"Your Numbers: {draw}")
