import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import random
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Ensure compatibility with older NumPy versions
if not hasattr(np, "product"):
    np.product = np.prod

# Page config
st.set_page_config(page_title="PowerBall Bayesian Simulator", layout="centered")
st.title("🎲 PowerBall Bayesian Network Simulator")
st.markdown("""
This app simulates a **biased Bayesian Network** for South African PowerBall numbers:
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

# Define nodes and edges (chain structure)
balls = [f"Ball_{i}" for i in range(1, 6)] + ["PowerBall"]
edges = [(balls[i], balls[i+1]) for i in range(len(balls) - 1)]

# Create model
model = BayesianNetwork(edges)

# Biased distribution function
def get_biased_distribution(n):
    raw = np.array([1 / (i + 1) for i in range(n)])
    return list(raw / raw.sum())

# Add CPDs (including parent-child relationships)
cpd_error = False
for i, ball in enumerate(balls):
    dist = get_biased_distribution(45 if i < 5 else 20)
    if len(dist) != (45 if i < 5 else 20):
        st.error(f"Invalid distribution for {ball}.")
        cpd_error = True
        break
    try:
        dist_array = np.array(dist).reshape(-1, 1)
        cpd = TabularCPD(variable=ball, variable_card=dist_array.shape[0], values=dist_array)
        model.add_node(ball)
        model.add_cpds(cpd)
    except Exception as e:
        st.error(f"🔥 Error while creating CPD for {ball}: {e}")
        cpd_error = True
        st.stop()

# Validate model
if not cpd_error:
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
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightgreen', font_size=12, font_weight='bold', arrows=True, ax=ax)
ax.set_title("PowerBall DAG")
st.pyplot(fig)

# -----------------------------
# 🎯 Strategy-Based Simulation
# -----------------------------
st.subheader("🎯 Strategy-Based PowerBall Simulation")

# Helper functions
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

# Strategy options
strategy_funcs = {
    "Favor Odd Numbers": generate_favor_odd,
    "Favor Even Numbers": generate_favor_even,
    "Avoid Prime Numbers": generate_avoid_primes,
    "Avoid Sequential Numbers": generate_no_sequential,
    "Avoid High Numbers": generate_avoid_high,
    "Favor Prime Numbers": generate_favor_primes
}

# User selection
selected_strategy = st.selectbox("Select a Bias Strategy", list(strategy_funcs.keys()))

if st.button("Run Strategy Simulation"):
    nums, powerball = strategy_funcs[selected_strategy]()
    st.success(f"🎲 Strategy: {selected_strategy}")
    st.write(f"🟢 Main Balls: {nums} + 🔴 PowerBall: {powerball}")
