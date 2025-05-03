
import numpy as np
import random
import pandas as pd
import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="PowerBall Bayesian Network Simulator", layout="centered")
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

# Define balls
main_balls = [f"Ball_{i}" for i in range(1, 6)]
all_balls = main_balls + ["PowerBall"]

# Define edges
edges_main = [(main_balls[i], main_balls[i+1]) for i in range(len(main_balls) - 1)]
edges_power = []  # PowerBall has no parent or child
edges = edges_main + edges_power

# Create model
model = BayesianNetwork(edges)

# Define cardinalities
cardinalities = {ball: 45 for ball in main_balls}
cardinalities["PowerBall"] = 20

# Biased distribution function
def get_biased_distribution(n):
    raw = np.array([1 / (i + 1) for i in range(n)])
    return list(raw / raw.sum())

# Conditional distribution based on parent value
def conditional_dist_given_parent(parent_val, child_cardinality):
    probs = np.array([1 / (abs(i - parent_val) + 1) for i in range(1, child_cardinality + 1)])
    return (probs / probs.sum()).tolist()

# Define CPDs
cpds = []
for i, ball in enumerate(all_balls):
    card = cardinalities[ball]

    if ball == "Ball_1" or ball == "PowerBall":
        dist = get_biased_distribution(card)
        cpd = TabularCPD(variable=ball, variable_card=card,
                         values=[[p] for p in dist])
    else:
        parent = all_balls[i - 1]
        parent_card = cardinalities[parent]

        matrix = []
        for parent_val in range(1, parent_card + 1):
            cond_dist = conditional_dist_given_parent(parent_val, card)
            matrix.append(cond_dist)
        
        matrix_T = np.array(matrix).T.tolist()

        cpd = TabularCPD(variable=ball, variable_card=card,
                         values=matrix_T,
                         evidence=[parent],
                         evidence_card=[parent_card])

    cpds.append(cpd)

# Add CPDs to the model
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
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', arrows=True, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=["PowerBall"], node_color="lightcoral", node_size=3000, ax=ax)
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
