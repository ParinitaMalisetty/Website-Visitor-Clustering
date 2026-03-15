import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Page Configuration
st.set_page_config(page_title="Visitor Clustering Dashboard", layout="wide")

st.title("🌐 Website Visitor Clustering")
st.write("Segment your website audience using K-Means Clustering to uncover hidden user behaviors.")


# --- 1. Data Loading (Training Data) ---
@st.cache_data
def load_data():
    return pd.read_csv("website_traffic.csv")


try:
    df = load_data()
    st.sidebar.success("Base Dataset Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: 'website_traffic.csv' not found. Please add it to your PyCharm project folder.")
    st.stop()

# --- 2. Preprocessing & Feature Selection ---
clustering_features = [
    'Page Views',
    'Session Duration',
    'Bounce Rate',
    'Time on Page',
    'Previous Visits',
    'Conversion Rate'
]

# Ensure the columns exist in the dataframe
missing_cols = [col for col in clustering_features if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in base dataset: {missing_cols}")
    st.stop()

X = df[clustering_features]

# Fit the scaler on the base data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Clustering Logic (Training) ---
st.sidebar.header("1. Train the Model")
k_value = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=8, value=3)

# Fit the K-Means model on base data
model = KMeans(n_clusters=k_value, random_state=42)
cluster_labels = model.fit_predict(X_scaled)
df['Cluster'] = cluster_labels.astype(str)  # Convert to string for discrete colors

# Calculate Performance Metrics
inertia = model.inertia_
sil_score = silhouette_score(X_scaled, cluster_labels)
db_index = davies_bouldin_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

# Fit PCA on base data for 2D graphing
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PCA1'] = components[:, 0]
df['PCA2'] = components[:, 1]

# Display Base Visualizations via Tabs
tab1, tab2, tab3 = st.tabs(["Base Data Clusters", "Cluster Analytics", "Model Evaluation"])

with tab1:
    st.subheader(f"Visualizing {k_value} Visitor Segments (Base Data)")
    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='Cluster',
        hover_data=['Traffic Source'] + clustering_features if 'Traffic Source' in df.columns else clustering_features,
        title="Visitor Clusters (PCA Reduced)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Cluster Characteristics (Averages)")
    cluster_summary = df.groupby('Cluster')[clustering_features].mean().reset_index()
    st.dataframe(cluster_summary, use_container_width=True)

with tab3:
    st.subheader("📊 Clustering Performance Metrics")
    st.write(
        "Since clustering is unsupervised, we measure performance based on how well-separated and dense the clusters are.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Silhouette Score", f"{sil_score:.3f}",
                help="Closer to 1 is better. Measures how tight and well-separated the clusters are.")
    col2.metric("Davies-Bouldin Index", f"{db_index:.3f}",
                help="Lower is better. Measures the average 'similarity' between clusters.")
    col3.metric("Calinski-Harabasz", f"{ch_score:.0f}",
                help="Higher is better. Ratio of between-cluster variance to within-cluster variance.")
    col4.metric("Inertia", f"{inertia:.0f}",
                help="Lower is better. Sum of squared distances of samples to their closest cluster center.")

st.divider()

# --- 4. Test Data Upload & Prediction ---
st.header("Upload Test Data for Prediction")
st.write("Upload a new CSV file with the same columns to assign new visitors to the existing clusters.")

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=['csv'])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    missing_test_cols = [col for col in clustering_features if col not in test_df.columns]

    if missing_test_cols:
        st.error(f"Your test data is missing these required columns: {missing_test_cols}")
    else:
        # Preprocess test data
        X_test = test_df[clustering_features]
        X_test_scaled = scaler.transform(X_test)

        # Predict clusters
        test_predictions = model.predict(X_test_scaled)
        test_df['Predicted_Cluster'] = test_predictions.astype(str)

        # Apply PCA
        test_components = pca.transform(X_test_scaled)
        test_df['PCA1'] = test_components[:, 0]
        test_df['PCA2'] = test_components[:, 1]

        st.success("Test data successfully clustered!")

        st.subheader("Visualizing Test Data Predictions")
        fig_test = px.scatter(
            test_df, x='PCA1', y='PCA2', color='Predicted_Cluster',
            hover_data=clustering_features,
            title="Test Data Mapped to Existing Clusters",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_test, use_container_width=True)

        st.subheader("Download Results")
        csv_data = test_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Test Data as CSV",
            data=csv_data,
            file_name="clustered_test_data.csv",
            mime="text/csv",
        )