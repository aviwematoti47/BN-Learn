import numpy as np
import random
import pandas as pd
import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="PowerBall Bayesian Network Simulator", layout="centered")
st.title("🎲 PowerBall Bayesian Network Simulator")
st.markdown("""
This app simulates a **biased Bayesian Network** for PowerBall numbers:
- **5 Main Balls** drawn from numbers **1 to 45**
- **1 PowerBall** drawn from numbers **1 to 20**
""")

# Upload historical data
uploaded_file = st.file_uploader("📁 Upload your PowerBall historical Excel file", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        historical_df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully!")
        st.write(historical_df.head())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

# Define nodes
main_balls = [f"Ball_{i}" for i in range(1, 6)]
all_balls = main_balls + ["PowerBall"]

# Define edges for the main balls chain
edges = [(main_balls[i], main_balls[i+1]) for i in range(len(main_balls)-1)]

# Create model and add PowerBall as a separate node (no parent)
model = BayesianNetwork(edges)
model.add_node("PowerBall")  # Ensure it's known to the model

# Biased distribution helper
def get_biased_distribution(n):
    raw = np.array([1 / (i + 1) for i in range(n)])
    return list(raw / raw.sum())

# Define CPDs
cpds = []
for i, ball in enumerate(all_balls):
    cardinality = 45 if ball != "PowerBall" else 20
    dist = get_biased_distribution(cardinality)

    if ball == "Ball_1" or ball == "PowerBall":
        cpd = TabularCPD(variable=ball, variable_card=cardinality, values=[[p] for p in dist])
    else:
        parent = all_balls[i - 1]
        parent_cardinality = 45
        values = np.tile(dist, (parent_cardinality, 1)).T.tolist()
        cpd = TabularCPD(variable=ball, variable_card=cardinality,
                         values=values,
                         evidence=[parent],
                         evidence_card=[parent_cardinality])
    cpds.append(cpd)

# Add CPDs to model
try:
    model.add_cpds(*cpds)
except Exception as e:
    st.error(f"❌ Error while adding CPDs: {e}")
    st.stop()

# Validate model
try:
    if model.check_model():
        st.success("✅ Bayesian Network with CPDs created and validated!")
    else:
        st.error("❌ Model is invalid.")
        st.stop()
except Exception as e:
    st.error(f"❌ Validation error: {e}")
    st.stop()

# Visualize DAG
st.subheader("🧠 DAG Structure")
G = nx.DiGraph()
G.add_edges_from(edges)
G.add_node("PowerBall")  # Include PowerBall as a node
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(7, 5))
nx.draw_networkx_nodes(G, pos, nodelist=main_balls, node_color="lightgreen", node_size=3000, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=["PowerBall"], node_color="lightcoral", node_size=3000, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
ax.set_title("PowerBall DAG")
st.pyplot(fig)

# -----------------------------
# 🎯 Strategy-Based Simulation
# -----------------------------
st.subheader("🎯 Strategy-Based PowerBall Simulation")

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(1, 46) if is_prime(n)]
evens = [n for n in range(1, 46) if n % 2 == 0]
odds = [n for n in range(1, 46) if n % 2 != 0]
low_range = list(range(1, 26))

# Strategy generators
def generate_favor_odd():
    return sorted(random.sample(odds, 3) + random.sample(evens, 2)), random.randint(1, 20)

def generate_favor_even():
    return sorted(random.sample(evens, 3) + random.sample(odds, 2)), random.randint(1, 20)

def generate_avoid_primes():
    non_primes = [n for n in range(1, 46) if n not in primes]
    return sorted(random.sample(non_primes, 5)), random.randint(1, 20)

def generate_no_sequential():
    nums = []
    while len(nums) < 5:
        n = random.randint(1, 45)
        if all(abs(n - x) > 1 for x in nums):
            nums.append(n)
    return sorted(nums), random.randint(1, 20)

def generate_avoid_high():
    return sorted(random.sample(low_range, 5)), random.randint(1, 20)

def generate_favor_primes():
    return sorted(random.sample(primes, 3) + random.sample([n for n in range(1, 46) if n not in primes], 2)), random.randint(1, 20)

strategy_funcs = {
    "Favor Odd Numbers": generate_favor_odd,
    "Favor Even Numbers": generate_favor_even,
    "Avoid Prime Numbers": generate_avoid_primes,
    "Avoid Sequential Numbers": generate_no_sequential,
    "Avoid High Numbers": generate_avoid_high,
    "Favor Prime Numbers": generate_favor_primes
}

# Run simulation
selected_strategy = st.selectbox("Select a Bias Strategy", list(strategy_funcs.keys()))
if st.button("Run Strategy Simulation"):
    nums, powerball = strategy_funcs[selected_strategy]()
    st.success(f"🎲 Strategy: {selected_strategy}")
    st.write(f"🟢 Main Balls: {nums} + 🔴 PowerBall: {powerball}")
