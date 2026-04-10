"""
Customer Segmentation — Streamlit Version
Run: streamlit run streamlit_app.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────
st.set_page_config(page_title="SegmentIQ", page_icon="◈", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0a0a14; color: #d0d0ee; }
  h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── Cluster profiles ──────────────────────────
PROFILES = {
    0: {"name": "Careful Spenders",  "icon": "💼", "tag": "Low Priority",     "color": "#6c757d"},
    1: {"name": "Target Customers",  "icon": "🎯", "tag": "High Value",        "color": "#198754"},
    2: {"name": "Potential Churners","icon": "⚠️", "tag": "At Risk",           "color": "#dc3545"},
    3: {"name": "Budget Shoppers",   "icon": "🛒", "tag": "Low Value",         "color": "#fd7e14"},
    4: {"name": "Impulsive Buyers",  "icon": "✨", "tag": "Growing Segment",   "color": "#0d6efd"},
}
COLORS = ['#6c757d','#198754','#dc3545','#fd7e14','#0d6efd']

# ── Train model ───────────────────────────────
@st.cache_resource
def get_model():
    np.random.seed(42)
    n = 200
    data = np.vstack([
        np.random.multivariate_normal([45,60,45],  [[80,10,10],[10,120,10],[10,10,200]], n//5),
        np.random.multivariate_normal([35,110,82], [[60,5,5],[5,100,5],[5,5,150]], n//5),
        np.random.multivariate_normal([42,115,18], [[70,5,5],[5,100,5],[5,5,100]], n//5),
        np.random.multivariate_normal([50,28,20],  [[80,5,5],[5,80,5],[5,5,120]], n//5),
        np.random.multivariate_normal([28,32,78],  [[60,5,5],[5,80,5],[5,5,150]], n//5),
    ])
    data[:,0] = np.clip(data[:,0], 18, 70)
    data[:,1] = np.clip(data[:,1], 15, 137)
    data[:,2] = np.clip(data[:,2], 1, 99)
    scaler = StandardScaler()
    ds = scaler.fit_transform(data)
    km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    km.fit(ds)
    return km, scaler, data

km, scaler, raw = get_model()

# ── UI ────────────────────────────────────────
st.title("◈ SegmentIQ — Customer Segmentation")
st.caption("K-Means clustering · k = 5 segments")
st.divider()

col_inp, col_res = st.columns([1, 2], gap="large")

with col_inp:
    st.subheader("Customer Input")
    age    = st.slider("Age",              18,  70, 30)
    income = st.slider("Annual Income (k$)", 15, 137, 60)
    score  = st.slider("Spending Score",    1,  100, 50)
    go = st.button("Classify →", use_container_width=True, type="primary")

with col_res:
    if go:
        user_raw    = np.array([[age, income, score]])
        user_scaled = scaler.transform(user_raw)
        cluster     = int(km.predict(user_scaled)[0])
        p = PROFILES[cluster]

        st.subheader(f"{p['icon']} {p['name']}")
        st.caption(f"**{p['tag']}** · Cluster {cluster}")

        labels  = km.predict(scaler.transform(raw))
        centers = scaler.inverse_transform(km.cluster_centers_)

        # 2D chart
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#111128')
        for c in range(5):
            m = labels == c
            ax.scatter(raw[m,1], raw[m,2], c=COLORS[c], s=35, alpha=0.65,
                       label=PROFILES[c]['name'])
        ax.scatter(centers[:,1], centers[:,2], c='white', s=200, marker='*')
        ax.scatter(income, score, c=COLORS[cluster], s=350, marker='^',
                   edgecolors='white', linewidths=2, zorder=20, label='You ▲')
        ax.set_xlabel('Annual Income (k$)', color='#aaaacc')
        ax.set_ylabel('Spending Score', color='#aaaacc')
        ax.set_title('Income vs Spending Score', color='white')
        ax.tick_params(colors='#666688')
        for s in ax.spines.values(): s.set_edgecolor('#333355')
        ax.legend(fontsize=7, framealpha=0.1, labelcolor='white')
        ax.grid(color='#ffffff0d', linestyle='--')
        st.pyplot(fig)
    else:
        st.info("Adjust the sliders and click **Classify →** to see results.")

st.divider()
st.subheader("Segment Profiles")
cols = st.columns(5)
for i, (cid, p) in enumerate(PROFILES.items()):
    with cols[i]:
        st.markdown(f"**{p['icon']} {p['name']}**")
        st.caption(p['tag'])
