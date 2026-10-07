import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Buyer Segmentation & Investment Profiling")
st.write(
    "Machine Learning based customer segmentation using K-Means clustering."
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    possible_files = [
        "Client Segmentation Data.xlsx",
        "Real_Estate_Market_Intelligent_data.xlsx",
        "final_customer_segmentation.xlsx"
    ]

    for file in possible_files:
        try:
            excel = pd.ExcelFile(file)

            if "clients" in excel.sheet_names:
                return pd.read_excel(file, sheet_name="clients")

            return pd.read_excel(file)

        except Exception:
            continue

    return None


df = load_data()

if df is None:
    st.error(
        "Dataset not found. Please make sure the Excel dataset "
        "is uploaded in the GitHub repository."
    )
    st.stop()

st.success(f"Dataset loaded successfully: {df.shape[0]} records")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Model Settings")

max_k = min(10, len(df) - 1)

if max_k < 2:
    st.error("Not enough records for clustering.")
    st.stop()

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=max_k,
    value=min(4, max_k)
)

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Total Features", len(df.columns))

with col3:
    st.metric("Selected Clusters", k)

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

# Remove obvious ID / target columns
excluded = [
    "client_id",
    "customer_id",
    "id",
    "cluster"
]

features = [
    col for col in numeric_columns
    if col.lower() not in excluded
]

if len(features) < 2:
    st.error(
        "At least two numerical features are required for clustering."
    )
    st.stop()

st.sidebar.subheader("📌 Features Used")

selected_features = st.sidebar.multiselect(
    "Select features",
    features,
    default=features
)

if len(selected_features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

X = df[selected_features].copy()

# Handle missing values
X = X.fillna(X.median(numeric_only=True))

# ---------------------------------------------------
# K-MEANS MODEL
# ---------------------------------------------------

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# ---------------------------------------------------
# CLUSTER RESULTS
# ---------------------------------------------------

st.header("🎯 Customer Segmentation")

cluster_counts = df["Cluster"].value_counts().sort_index()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    cluster_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers by Cluster")
    st.pyplot(fig)

with col2:
    st.subheader("Cluster Summary")

    summary = df.groupby("Cluster")[selected_features].mean()
    st.dataframe(summary.round(2), use_container_width=True)

# ---------------------------------------------------
# SILHOUETTE SCORE
# ---------------------------------------------------

if len(df) > k:

    score = silhouette_score(X, df["Cluster"])

    st.subheader("📈 Model Evaluation")

    st.metric(
        "Silhouette Score",
        round(score, 4)
    )

    if score >= 0.5:
        st.success("Good quality clustering.")
    elif score >= 0.25:
        st.info("Moderate clustering quality.")
    else:
        st.warning("Weak clustering structure.")

# ---------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------

st.header("📉 Elbow Method")

inertias = []

for i in range(2, max_k + 1):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X)
    inertias.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    range(2, max_k + 1),
    inertias,
    marker="o"
)

ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")

st.pyplot(fig)

# ---------------------------------------------------
# SILHOUETTE ANALYSIS
# ---------------------------------------------------

st.header("📊 Silhouette Analysis")

silhouette_scores = []

for i in range(2, max_k + 1):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    silhouette_scores.append(
        silhouette_score(X, labels)
    )

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    range(2, max_k + 1),
    silhouette_scores,
    marker="o"
)

ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score by Number of Clusters")

st.pyplot(fig)

# ---------------------------------------------------
# FINAL DATA
# ---------------------------------------------------

st.header("👥 Segmented Customers")

st.dataframe(
    df,
    use_container_width=True
)

# ---------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Segmented Customer Data",
    data=csv,
    file_name="final_customer_segmentation.csv",
    mime="text/csv"
)

st.success("✅ Customer segmentation completed successfully!")
