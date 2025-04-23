import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import networkx as nx
import matplotlib.pyplot as plt

# -------------- Streamlit Config --------------
st.set_page_config(page_title="Bayesian Lotto Predictor", layout="centered")
st.title("🎯 Bayesian Lotto Predictor")
st.markdown("Build a uniform Bayesian Network and simulate biased lotto draws (1–52, 6 main + 1 bonus).")

# -------------- DAG Structure Setup --------------
st.header("1️⃣ Uniform DAG Creator")

nodes = [f'Ball_{i}' for i in range(1, 8)]
edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
st.code(str(edges), language='python')
model = DiscreteBayesianNetwork(edges)

# -------------- Constructive Bias Setup --------------
st.header("2️⃣ Apply Constructive Bias to All Balls")

apply_bias = st.checkbox("Apply constructive bias to all balls?", value=True)
hot_numbers = [3, 7, 15, 28, 33, 44, 51]  # Example: perceived lucky numbers

def get_biased_distribution():
    probs = [0.07 if i+1 in hot_numbers else 0.002 for i in range(52)]
    total = sum(probs)
    return [p / total for p in probs]

if apply_bias:
    # Ball_1 has no parents
    cpd_1 = TabularCPD(variable='Ball_1', variable_card=52, values=[get_biased_distribution()])
    model.add_cpds(cpd_1)

    # Ball_2 to Ball_7 have one parent each
    for i in range(2, 8):
        parent = f'Ball_{i-1}'
        child = f'Ball_{i}'
        # Create a conditional CPD: same biased distribution for each parent state
        values = [get_biased_distribution()] * 52  # 52 rows: one for each parent value
        cpd = TabularCPD(variable=child, variable_card=52, values=list(map(list, zip(*values))), evidence=[parent], evidence_card=[52])
        model.add_cpds(cpd)

    st.success("✅ Biased CPDs added for all balls (conditional where needed).")
else:
    uniform = [[1 / 52 for _ in range(52)]]
    model.add_cpds(TabularCPD("Ball_1", 52, uniform))
    for i in range(2, 8):
        parent = f'Ball_{i-1}'
        child = f'Ball_{i}'
        values = uniform * 52
        cpd = TabularCPD(variable=child, variable_card=52, values=list(map(list, zip(*values))), evidence=[parent], evidence_card=[52])
        model.add_cpds(cpd)
    st.info("ℹ️ Uniform distributions used for all CPDs.")

# -------------- DAG Visualization --------------
st.header("3️⃣ DAG Visualization")

G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=2000, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)
for src, dst in G.edges():
    ax.annotate(
        "", xy=pos[dst], xytext=pos[src],
        arrowprops=dict(arrowstyle="->", lw=2, shrinkA=15, shrinkB=15)
    )
ax.set_axis_off()
st.pyplot(fig)

# -------------- Sampling --------------
st.header("4️⃣ Sample Lotto Draws")

num_samples = st.slider("How many draws?", 1, 100, 10)
if st.button("🎲 Generate Samples"):
    try:
        sampler = BayesianModelSampling(model)
        samples = sampler.forward_sample(size=num_samples)
        st.dataframe(samples)
    except Exception as e:
        st.error(f"❌ Sampling failed: {e}")
