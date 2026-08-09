import streamlit as st
import pandas as pd
import numpy as np

# Handle optional plotting libraries gracefully with helpful Streamlit messages
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("Required package 'matplotlib' is not installed. Install it with `pip install matplotlib` or add it to requirements.txt and redeploy.")
    raise

try:
    import seaborn as sns
except ModuleNotFoundError:
    st.error("Required package 'seaborn' is not installed. Install it with `pip install seaborn` or add it to requirements.txt and redeploy.")
    raise

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation AI App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 20px;
    }
    .creator-badge {
        background-color: #E0E7FF;
        border-left: 5px solid #3B82F6;
        padding: 12px;
        border-radius: 6px;
        font-weight: 500;
        color: #1E40AF;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<div class='main-header'>Customer Segmentation using K-Means Clustering</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>An Industry-Oriented Unsupervised Machine Learning Application</div>", unsafe_allow_html=True)

# Required Branding & Authority Header
st.markdown("""
<div class='creator-badge'>
    🎓 <b>Created by:</b> Ravi Kumar Singh <br>
    🌐 <i>International Data Scientist & Microsoft Certified Trainer</i>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Overview & Business Value", 
        "Dataset & EDA", 
        "Model Building & Elbow Method", 
        "Live Customer Predictor"
    ]
)

# ------------------------------------------------------------
# 3. DATASET & MODEL PREPARATION
# ------------------------------------------------------------
@st.cache_data
def load_data():
    data = {
        "Customer_ID": [f"C{i:03d}" for i in range(1, 21)],
        "Age": [22, 45, 29, 52, 31, 25, 41, 36, 28, 50, 23, 48, 34, 55, 27, 39, 30, 46, 33, 58],
        "Annual_Income": [25000, 75000, 42000, 90000, 48000, 30000, 68000, 55000, 38000, 85000, 28000, 82000, 52000, 95000, 35000, 62000, 45000, 78000, 50000, 100000],
        "Spending_Score": [85, 35, 72, 25, 68, 90, 40, 55, 75, 20, 88, 30, 60, 15, 80, 45, 70, 32, 65, 10],
        "Purchases_Per_Year": [18, 5, 14, 3, 12, 20, 7, 10, 15, 2, 19, 4, 11, 1, 17, 8, 13, 6, 12, 2]
    }
    return pd.DataFrame(data)

df = load_data()
features = ["Annual_Income", "Spending_Score", "Purchases_Per_Year"]
X = df[features]

@st.cache_resource
def build_model(data_x):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_x)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    
    return scaler, kmeans, X_scaled, clusters, sil_score

scaler, kmeans, X_scaled, df["Cluster"], sil_score = build_model(X)

# ------------------------------------------------------------
# PAGE 1: OVERVIEW & BUSINESS VALUE
# ------------------------------------------------------------
if page == "Overview & Business Value":
    st.header("📋 Problem Statement & Industry Solution")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔴 The Problem")
        st.write("""
        Treating all customers uniformly leads to generic advertising, high Customer Acquisition Costs (CAC), and low retention rates.
        """)
        
    with col2:
        st.subheader("🟢 Machine Learning Solution")
        st.write("""
        K-Means Clustering segments customers by income, spending behavior, and purchase frequency to automate personalized marketing strategies.
        """)

    st.markdown("---")
    st.subheader("💼 Key Industry Use Cases")
    
    uc1, uc2, uc3, uc4 = st.columns(4)
    with uc1:
        st.markdown("**🎯 Precision Targeting**")
        st.caption("Custom campaigns instead of broad spam emails.")
    with uc2:
        st.markdown("**🔄 Churn Prevention**")
        st.caption("Identify high-value clients losing engagement early.")
    with uc3:
        st.markdown("**🏷️ Dynamic Pricing**")
        st.caption("Target luxury tiers vs promo discount tiers.")
    with uc4:
        st.markdown("**⚡ VIP Allocation**")
        st.caption("Assign direct account reps to top revenue tiers.")

# ------------------------------------------------------------
# PAGE 2: DATASET & EDA
# ------------------------------------------------------------
elif page == "Dataset & EDA":
    st.header("📊 Data Inspection & Distribution")
    
    st.subheader("Customer DataFrame")
    st.dataframe(df, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Missing Values", df.isnull().sum().sum())
    c3.metric("Duplicate Rows", df.duplicated().sum())

    st.markdown("---")
    st.subheader("Exploratory Data Plots")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Age"], kde=True, color="#3B82F6", ax=ax)
        ax.set_title("Customer Age Distribution")
        st.pyplot(fig)
        
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score", s=120, color="#E11D48", ax=ax)
        ax.set_title("Annual Income vs Spending Score")
        st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 3: MODEL BUILDING & ELBOW METHOD
# ------------------------------------------------------------
elif page == "Model Building & Elbow Method":
    st.header("⚙️ Model Architecture & Optimal K Selection")
    
    st.subheader("1. Elbow Method Curve")
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(X_scaled)
        inertia.append(m.inertia_)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_range, inertia, marker="o", color="#8B5CF6", linewidth=2)
    ax.set_title("Elbow Method (WCSS vs Number of Clusters)")
    ax.set_xlabel("Clusters (K)")
    ax.set_ylabel("Inertia / WCSS")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    
    st.success(" Optimal K selection: **K = 4** based on the sharp elbow bend.")

    st.markdown("---")
    st.subheader("2. Final Segments Visualization & Silhouette Score")
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.metric("Silhouette Score", round(sil_score, 4))
        st.caption("Measures cluster density and separation strength.")
        st.markdown("#### Mean Values per Cluster:")
        st.dataframe(df.groupby("Cluster")[features].mean().round(2))
        
    with col_m2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df, x="Annual_Income", y="Spending_Score", hue="Cluster", palette="viridis", s=150, ax=ax
        )
        ax.set_title("Customer Segments (K=4)")
        st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 4: LIVE CUSTOMER PREDICTOR
# ------------------------------------------------------------
elif page == "Live Customer Predictor":
    st.header("🔮 Real-Time Customer Segment Predictor")
    st.write("Input new customer attributes below to determine their cluster and strategic action plan.")
    
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        in_income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=65000, step=5000)
    with col_i2:
        in_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=60)
    with col_i3:
        in_purchases = st.number_input("Purchases Per Year", min_value=1, max_value=50, value=12, step=1)
        
    if st.button("Predict Customer Segment", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[in_income, in_score, in_purchases]], columns=features)
        scaled_input = scaler.transform(input_data)
        pred_cluster = kmeans.predict(scaled_input)[0]
        
        st.markdown("---")
        st.success(f"### Predicted Cluster Assignment: **Cluster {pred_cluster}**")
        
        st.subheader("🎯 Business Recommendation Strategy:")
        if in_income >= 60000 and in_score >= 60:
            st.markdown("💎 **High-Value VIP Customer**")
            st.write("👉 Strategy: Priority support, exclusive rewards programs, VIP event invites.")
        elif in_income >= 60000 and in_score < 60:
            st.markdown("🎯 **Potential High-Spender**")
            st.write("👉 Strategy: Targeted email marketing highlighting product quality and value.")
        elif in_income < 60000 and in_score >= 60:
            st.markdown("🛍️ **Frequent Deal Shopper**")
            st.write("👉 Strategy: Send discount codes, flash sales, and bundle offers.")
        else:
            st.markdown("🏷️ **Budget Customer**")
            st.write("👉 Strategy: Low-cost automated engagement, seasonal sales notifications.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Created by <b>Ravi Kumar Singh</b> | International Data Scientist & Microsoft Certified Trainer</p>", unsafe_allow_html=True)
