import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import random  # ✅ This was missing — now fixed
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import pandas as pd

# Patch np.product if it's missing (older NumPy versions)
if not hasattr(np, "product"):
    np.product = np.prod

# Page configuration
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

# Initialize Bayesian Network
model = BayesianNetwork(edges)

# Function to generate a biased distribution favoring lower numbers
def get_biased_distribution():
    raw = np.array([1 / (i + 1) for i in range(52)])  # Bias toward lower numbers
    return list(raw / raw.sum())

# Define CPDs with error handling
cpds = []
cpd_error = False

for ball in balls:
    dist = get_biased_distribution()

    # Validate distribution length
    if len(dist) != 52:
        st.error(f"❌ Distribution for {ball} is invalid. Expected 52 values, got {len(dist)}.")
        cpd_error = True
        break

    try:
        dist_array = np.array(dist).reshape(52, 1)
        cpd = TabularCPD(variable=ball, variable_card=52, values=dist_array)
        model.add_cpds(cpd)
        cpds.append(cpd)

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

# Visualize DAG
st.subheader("📊 Lotto DAG Structure")
G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(7, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True, ax=ax)
ax.set_title("Lotto Bayesian Network DAG")
st.pyplot(fig)

# --------------------------------------
# 🎯 Strategy-Based Simulation Section
# --------------------------------------
st.subheader("🧠 Strategy-Based PowerBall Simulation")

# Helper functions
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

# Strategy generators
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
    st.write(f"🟢 Numbers: {nums} + 🔴 PowerBall: {powerball}")

# --------------------------------------
# 🗂️ Upload Historical Lotto Data Section
# --------------------------------------
st.subheader("📥 Upload Historical Lotto Data")

# Add the file uploader widget
uploaded_file = st.file_uploader("Upload Historical Lotto Data", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)

        # Display the first few rows of the dataframe
        st.write("Historical Data:", df.head())

        # You can also plot or manipulate the data as needed
        # For example, visualize some data
        st.subheader("Lotto Number Distribution")
        plt.figure(figsize=(10, 6))
        df['Lotto Number'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title("Lotto Number Frequency")
        plt.xlabel("Lotto Numbers")
        plt.ylabel("Frequency")
        st.pyplot()

    except Exception as e:
        st.error(f"Error reading the file: {e}")
