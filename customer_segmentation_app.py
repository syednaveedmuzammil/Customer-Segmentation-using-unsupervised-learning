# customer_segmentation_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Customer Segmentation using Unsupervised Machine Learning")
st.markdown("""
This app performs **RFM-based customer segmentation** using **K-Means clustering** on an e-commerce dataset.
Upload your data file below (CSV format) or use the **sample Online Retail dataset**.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# --- Load Data ---
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    st.success("✅ Dataset uploaded successfully!")
else:
    st.info("ℹ️ Using a small sample of the Online Retail dataset from UCI Repository...")
    sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    df = pd.read_csv(sample_url)
    st.warning("⚠️ This is just a sample dataset (replace with Online Retail).")

st.write("### Dataset Preview", df.head())

# --- Data Cleaning ---
st.header("🧹 Data Cleaning & Preparation")
if "CustomerID" in df.columns and "InvoiceDate" in df.columns:
    df = df.dropna(subset=["CustomerID"])
    if "InvoiceNo" in df.columns:
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # --- RFM Calculation ---
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,  # Recency
        "InvoiceNo": "nunique",                              # Frequency
        "TotalPrice": "sum"                                  # Monetary
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

    # --- Preprocessing ---
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # --- Choose number of clusters ---
    st.sidebar.subheader("Clustering Parameters")
    n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    sil_score = silhouette_score(rfm_scaled, rfm["Cluster"])
    st.sidebar.metric("Silhouette Score", f"{sil_score:.3f}")

    # --- PCA Visualization ---
    pca = PCA(n_components=2, random_state=42)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm["PCA1"], rfm["PCA2"] = rfm_pca[:, 0], rfm_pca[:, 1]

    # --- Cluster Summary ---
    st.header("📊 Cluster Summary")
    summary = rfm.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": ["mean", "count"]
    }).round(2)
    st.dataframe(summary)

    # --- Visualizations ---
    st.header("🎨 Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=rfm, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=60, ax=ax)
        ax.set_title("PCA Projection of Customer Segments")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        rfm_melted = rfm.melt(id_vars="Cluster", value_vars=["Recency", "Frequency", "Monetary"])
        sns.boxplot(x="variable", y="value", hue="Cluster", data=rfm_melted, ax=ax2)
        ax2.set_title("Distribution of RFM Values by Cluster")
        st.pyplot(fig2)

    # --- Download Results ---
    st.header("💾 Download Segmented Data")
    csv = rfm.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download Clustered Data as CSV", csv, "customer_segments.csv", "text/csv")

    # --- Interpretation ---
    st.header("🧠 Business Insights")
    st.markdown("""
    **Example interpretations:**
    - 🟢 **Cluster 0:** High spenders, frequent buyers → **VIP Customers**
    - 🔵 **Cluster 1:** Medium spenders, recent activity → **Loyal Customers**
    - 🟡 **Cluster 2:** Low frequency, low spending → **At-risk or Lost Customers**
    - 🟠 **Cluster 3:** Occasional buyers → **Potential Loyalists**

    Adjust the number of clusters from the sidebar to explore different groupings.
    """)
else:
    st.error("❌ Dataset does not have expected columns like `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice`. Please upload a valid Online Retail dataset.")
