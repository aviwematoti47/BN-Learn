import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import random
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Patch np.product if it's missing (older NumPy versions)
if not hasattr(np, "product"):
    np.product = np.prod

# Streamlit page config
st.set_page_config(page_title="Powerball Bayesian Network", layout="centered")
st.title("🎲 Powerball Bayesian Network Simulator")
st.markdown("""
This app simulates a **biased Bayesian Network** for Powerball numbers (1–52).
- **5 Powerball Balls** (5 normal + 1 bonus).
- **Constructive bias** added to each ball.
- Bias favors lower numbers (simulating real-world perception).
""")

# --- Upload historical data section ---
st.subheader("📁 Upload Historical Powerball Data")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Bayesian Network Construction ---
balls = [f"Ball_{i}" for i in range(1, 8)]
edges = []  # ✅ Removed edges to use independent CPDs (avoids check_model errors)

model = BayesianNetwork(edges)

def get_biased_distribution():
    raw = np.array([1 / (i + 1) for i in range(52)])  # Bias toward lower numbers
    return list(raw / raw.sum())

cpd_error = False
for ball in balls:
    dist = get_biased_distribution()

    if len(dist) != 52:
        st.error(f"❌ Distribution for {ball} is invalid. Expected 52 values, got {len(dist)}.")
        cpd_error = True
        break

    try:
        dist_array = np.array(dist).reshape(52, 1)
        cpd = TabularCPD(variable=ball, variable_card=52, values=dist_array)
        model.add_cpds(cpd)
    except Exception as e:
        st.error(f"🔥 Error while creating CPD for {ball}: {e}")
        cpd_error = True
        st.stop()

# Validate model
if not cpd_error:
    if model.check_model():
        st.success("✅ Bayesian Network with CPDs created and validated successfully!")
    else:
        st.error("❌ Model structure or CPDs are invalid.")
        st.stop()
else:
    st.stop()

# --- Visualize DAG ---
st.subheader("📊 Powerball DAG Structure")
G = nx.DiGraph()
G.add_nodes_from(balls)
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue',
        font_size=12, font_weight='bold', arrows=True, ax=ax)
ax.set_title("Powerball Bayesian Network DAG (Independent Balls)")
st.pyplot(fig)

# --- Strategy-Based PowerBall Simulation ---
st.subheader("🧠 Strategy-Based PowerBall Simulation")

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(1, 51) if is_prime(n)]
evens = [n for n in range(1, 51) if n % 2 == 0]
odds = [n for n in range(1, 51) if n % 2 != 0]
low_range = list(range(1, 31))

def generate_favor_odd():
    return sorted(random.sample(odds, 3) + random.sample(evens, 2)), random.randint(1, 20)

def generate_favor_even():
    return sorted(random.sample(evens, 3) + random.sample(odds, 2)), random.randint(1, 20)

def generate_avoid_primes():
    non_primes = [n for n in range(1, 51) if n not in primes]
    return sorted(random.sample(non_primes, 5)), random.randint(1, 20)

def generate_no_sequential():
    nums = []
    while len(nums) < 5:
        n = random.randint(1, 50)
        if all(abs(n - x) > 1 for x in nums):
            nums.append(n)
    return sorted(nums), random.randint(1, 20)

def generate_avoid_high():
    return sorted(random.sample(low_range, 5)), random.randint(1, 20)

def generate_favor_primes():
    return sorted(random.sample(primes, 3) + random.sample([n for n in range(1, 51) if n not in primes], 2)), random.randint(1, 20)

strategy_funcs = {
    "Favor Odd Numbers": generate_favor_odd,
    "Favor Even Numbers": generate_favor_even,
    "Avoid Prime Numbers": generate_avoid_primes,
    "Avoid Sequential Numbers": generate_no_sequential,
    "Avoid High Numbers": generate_avoid_high,
    "Favor Prime Numbers": generate_favor_primes
}

selected_strategy = st.selectbox("Select a Bias Strategy", list(strategy_funcs.keys()))

if st.button("Run Strategy Simulation"):
    nums, powerball = strategy_funcs[selected_strategy]()
    st.success(f"🎲 Strategy: {selected_strategy}")
    st.write(f"🟢 Numbers: {nums} + 🔴 PowerBall: {powerball}")
