"""
Customer Segmentation Web App
Backend: Flask + scikit-learn KMeans
"""

import os
import io
import base64
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ─────────────────────────────────────────────
# Cluster Profiles (based on typical Mall data)
# ─────────────────────────────────────────────
CLUSTER_PROFILES = {
    0: {
        "name": "Careful Spenders",
        "icon": "💼",
        "tag": "Low Priority",
        "color": "#6c757d",
        "description": "Middle-aged customers with moderate income who spend cautiously. They are value-seekers who respond well to discounts and loyalty programs.",
        "strategy": "Offer targeted promotions and bundle deals to increase engagement.",
    },
    1: {
        "name": "Target Customers",
        "icon": "🎯",
        "tag": "High Value",
        "color": "#198754",
        "description": "High-income, high-spending customers. These are the most profitable segment and brand loyalists.",
        "strategy": "Retain with premium memberships, exclusive offers, and VIP experiences.",
    },
    2: {
        "name": "Potential Churners",
        "icon": "⚠️",
        "tag": "At Risk",
        "color": "#dc3545",
        "description": "High income but very low spending. They have the budget but aren't engaging. Likely dissatisfied or unaware of offerings.",
        "strategy": "Re-engage with personalized campaigns and premium product showcases.",
    },
    3: {
        "name": "Budget Shoppers",
        "icon": "🛒",
        "tag": "Low Value",
        "color": "#fd7e14",
        "description": "Low income and low spending. Price-sensitive customers who make infrequent purchases.",
        "strategy": "Focus on affordable product lines and clearance promotions.",
    },
    4: {
        "name": "Impulsive Buyers",
        "icon": "✨",
        "tag": "Growing Segment",
        "color": "#0d6efd",
        "description": "Young customers with lower income but high spending scores. Trend-driven and responsive to marketing.",
        "strategy": "Leverage social media, flash sales, and influencer partnerships.",
    },
}


def train_model():
    """
    Train KMeans on synthetic Mall Customers-like data.
    In production, replace np.random with pd.read_csv('Mall_Customers.csv').
    """
    np.random.seed(42)
    n = 200

    # Simulate 5 distinct customer clusters
    data = np.vstack([
        np.random.multivariate_normal([45, 60, 45], [[80, 10, 10], [10, 120, 10], [10, 10, 200]], n // 5),
        np.random.multivariate_normal([35, 110, 82], [[60, 5, 5], [5, 100, 5], [5, 5, 150]], n // 5),
        np.random.multivariate_normal([42, 115, 18], [[70, 5, 5], [5, 100, 5], [5, 5, 100]], n // 5),
        np.random.multivariate_normal([50, 28, 20], [[80, 5, 5], [5, 80, 5], [5, 5, 120]], n // 5),
        np.random.multivariate_normal([28, 32, 78], [[60, 5, 5], [5, 80, 5], [5, 5, 150]], n // 5),
    ])

    # Clip to realistic ranges
    data[:, 0] = np.clip(data[:, 0], 18, 70)   # Age
    data[:, 1] = np.clip(data[:, 1], 15, 137)   # Annual Income
    data[:, 2] = np.clip(data[:, 2], 1, 99)     # Spending Score

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(data_scaled)

    # Save model artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump({'kmeans': kmeans, 'scaler': scaler, 'raw_data': data}, f)

    return kmeans, scaler, data


def load_model():
    """Load saved model or train a fresh one."""
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts['kmeans'], artifacts['scaler'], artifacts['raw_data']
    return train_model()


# ── Train on startup ──────────────────────────
kmeans, scaler, raw_data = load_model()

CLUSTER_COLORS = ['#6c757d', '#198754', '#dc3545', '#fd7e14', '#0d6efd']


def make_3d_plot(user_point=None, user_cluster=None):
    """Render a 3D scatter plot and return as base64 PNG."""
    fig = plt.figure(figsize=(9, 6.5))
    fig.patch.set_facecolor('#0f0f1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f1a')

    labels = kmeans.predict(scaler.transform(raw_data))
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    for c in range(5):
        mask = labels == c
        color = CLUSTER_COLORS[c]
        ax.scatter(
            raw_data[mask, 0], raw_data[mask, 1], raw_data[mask, 2],
            c=color, s=30, alpha=0.65, label=f"Cluster {c}"
        )

    # Plot cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='white', s=160, marker='*', zorder=10, label='Centers')

    # Highlight user point
    if user_point is not None and user_cluster is not None:
        uc = CLUSTER_COLORS[user_cluster]
        ax.scatter(*user_point, c=uc, s=350, marker='^',
                   edgecolors='white', linewidths=1.5, zorder=20, label='You')

    # Styling
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#ffffff20')
        axis.label.set_color('#aaaacc')
        axis.set_tick_params(colors='#666688')

    ax.set_xlabel('Age', labelpad=8)
    ax.set_ylabel('Annual Income (k$)', labelpad=8)
    ax.set_zlabel('Spending Score', labelpad=8)
    ax.set_title('Customer Segments — 3D View', color='white', pad=14, fontsize=12)
    ax.view_init(elev=22, azim=135)

    legend = ax.legend(loc='upper left', fontsize=7, framealpha=0.15, labelcolor='white')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_2d_plot(user_point=None, user_cluster=None):
    """Render a 2D Income vs Spending plot and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#111128')

    labels = kmeans.predict(scaler.transform(raw_data))
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    for c in range(5):
        mask = labels == c
        ax.scatter(raw_data[mask, 1], raw_data[mask, 2],
                   c=CLUSTER_COLORS[c], s=40, alpha=0.7,
                   label=CLUSTER_PROFILES[c]['name'])

    ax.scatter(centers[:, 1], centers[:, 2],
               c='white', s=200, marker='*', zorder=10)

    if user_point is not None and user_cluster is not None:
        uc = CLUSTER_COLORS[user_cluster]
        ax.scatter(user_point[1], user_point[2],
                   c=uc, s=400, marker='^',
                   edgecolors='white', linewidths=2, zorder=20, label='You ▲')

    ax.set_xlabel('Annual Income (k$)', color='#aaaacc')
    ax.set_ylabel('Spending Score (1-100)', color='#aaaacc')
    ax.set_title('Income vs Spending Score', color='white', fontsize=12, pad=10)
    ax.tick_params(colors='#666688')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.legend(fontsize=7, framealpha=0.15, labelcolor='white', loc='upper left')
    ax.grid(color='#ffffff0d', linestyle='--')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page with the cluster overview chart."""
    plot_2d = make_2d_plot()
    plot_3d = make_3d_plot()
    return render_template('index.html',
                           plot_2d=plot_2d,
                           plot_3d=plot_3d,
                           profiles=CLUSTER_PROFILES)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept user input, predict cluster, return result + updated plots."""
    try:
        data = request.get_json()
        age = float(data['age'])
        income = float(data['income'])
        score = float(data['score'])

        # Validate ranges
        if not (10 <= age <= 100):
            return jsonify({'error': 'Age must be between 10 and 100'}), 400
        if not (1 <= income <= 200):
            return jsonify({'error': 'Annual Income must be between 1 and 200 (k$)'}), 400
        if not (1 <= score <= 100):
            return jsonify({'error': 'Spending Score must be between 1 and 100'}), 400

        user_raw = np.array([[age, income, score]])
        user_scaled = scaler.transform(user_raw)
        cluster = int(kmeans.predict(user_scaled)[0])

        profile = CLUSTER_PROFILES[cluster]
        user_point = [age, income, score]

        plot_2d = make_2d_plot(user_point, cluster)
        plot_3d = make_3d_plot(user_point, cluster)

        return jsonify({
            'cluster': cluster,
            'profile': profile,
            'plot_2d': plot_2d,
            'plot_3d': plot_3d,
        })

    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5000)