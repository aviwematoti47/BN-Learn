import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(page_title="Biased Lotto Bayesian Network", layout="centered")
st.title("🎯 Lotto Bayesian Network Simulator")
st.markdown("""
This app builds a **Bayesian Network** for a Lotto draw using **constructive bias**.

- 7 balls drawn per round (1–52).
- Biased toward 'hot' numbers for realism.
- Use this to simulate and visualize dependencies.
""")

# Define hot numbers
hot_numbers = [3, 7, 19, 23, 34, 42, 45]  # User can change these if needed

# Define function to create biased distributions
def get_biased_distribution():
    probs = [0.07 if i+1 in hot_numbers else 0.002 for i in range(52)]
    total = sum(probs)
    return [[p / total for p in probs]]  # Must return 2D list

# Build the Bayesian Network
model = BayesianNetwork()

# Define nodes and edges
balls = [f"Ball_{i}" for i in range(1, 8)]
edges = [(balls[i], balls[i+1]) for i in range(len(balls)-1)]  # Uniform chain: Ball_1 → Ball_2 → ... → Ball_7
model.add_edges_from(edges)

# Add CPDs
cpds = []

# First ball has no parent — just biased CPD
cpd_first = TabularCPD(variable=balls[0], variable_card=52, values=get_biased_distribution())
cpds.append(cpd_first)

# For dependent balls, apply same bias but with uniform conditional probability (simplified assumption)
for i in range(1, len(balls)):
    child = balls[i]
    parent = balls[i-1]
    # Repeat biased dist for each parent state (so 52 rows of 52 each = 52x52 matrix)
    biased_row = get_biased_distribution()[0]
    values = [biased_row for _ in range(52)]
    cpd = TabularCPD(variable=child, variable_card=52, values=values, evidence=[parent], evidence_card=[52])
    cpds.append(cpd)

# Add all CPDs
for cpd in cpds:
    model.add_cpds(cpd)

# Validate model
if model.check_model():
    st.success("✅ Bayesian Network created successfully!")

    # Visualize using networkx
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=12, ax=ax)
    ax.set_title("Lotto DAG Structure", fontsize=16)
    st.pyplot(fig)
else:
    st.error("❌ The Bayesian Network model is invalid.")
