import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page Configuration
st.set_page_config(page_title="Visitor Clustering Dashboard", layout="wide")

st.title("🌐 Website Visitor Clustering")
st.write("Segment your website audience using K-Means Clustering to uncover hidden user behaviors.")

# 1. Data Loading
@st.cache_data
def load_data():
    # Make sure 'website_traffic.csv' matches your actual file name in PyCharm
    df = pd.read_csv("website_traffic.csv")
    return df

try:
    df = load_data()
    st.sidebar.success("Dataset Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: 'website_traffic.csv' not found. Please add it to your PyCharm project folder.")
    st.stop()

# 2. Preprocessing & Feature Selection
st.subheader("1. Feature Engineering & Selection")

# We use only the numerical columns for standard K-Means clustering
clustering_features = [
    'Page Views',
    'Session Duration',
    'Bounce Rate',
    'Time on Page',
    'Previous Visits',
    'Conversion Rate'
]

# Ensure the columns exist in the dataframe to prevent errors
missing_cols = [col for col in clustering_features if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

X = df[clustering_features]

# Scaling the data so large numbers (like Session Duration) don't overpower small ones (like Bounce Rate)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with st.expander("View Raw Dataset"):
    st.dataframe(df.head())

# 3. Clustering Logic
st.sidebar.subheader("Clustering Settings")
k_value = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=8, value=3)

# Fit the K-Means model
model = KMeans(n_clusters=k_value, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)
# Convert Cluster to string for better categorical color plotting in Plotly
df['Cluster'] = df['Cluster'].astype(str)

# 4. Visualization with PCA
# Reduce the 6 numerical features down to 2 components for 2D graphing
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PCA1'] = components[:, 0]
df['PCA2'] = components[:, 1]

st.subheader(f"2. Visualizing {k_value} Visitor Segments")
fig = px.scatter(
    df,
    x='PCA1',
    y='PCA2',
    color='Cluster',
    hover_data=['Traffic Source'] + clustering_features, # Show traffic source on hover!
    title="Visitor Clusters (PCA Reduced)",
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig, use_container_width=True)

# 5. Cluster Analysis
st.subheader("3. Cluster Characteristics (Averages)")
# Group by cluster and calculate the mean for our numerical features
cluster_summary = df.groupby('Cluster')[clustering_features].mean().reset_index()
st.dataframe(cluster_summary, use_container_width=True)

# Look at Traffic Source distribution within clusters
st.subheader("4. Traffic Source Breakdown per Cluster")
traffic_breakdown = df.groupby(['Cluster', 'Traffic Source']).size().reset_index(name='Count')
fig_bar = px.bar(
    traffic_breakdown,
    x='Cluster',
    y='Count',
    color='Traffic Source',
    title="Traffic Sources by Cluster",
    barmode='group'
)
st.plotly_chart(fig_bar, use_container_width=True)