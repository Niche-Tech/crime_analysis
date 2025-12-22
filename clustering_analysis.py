# clustering_analysis.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def show_clustering_analysis(df):
    """Display the k-means clustering analysis page."""
    st.title("Clustering Analysis (k-means)")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")

        # Year selection
        years = sorted(df['Year'].unique())
        selected_year = st.selectbox(
            "Select Year",
            years,
            index=len(years) - 1  # default: latest year
        )

        # Clustering parameters
        n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=min(10, len(df['Region'].unique()) - 1), value=3)

        # Optional: feature selection (if many crime types)
        crime_types_all = sorted(df['Type of crime'].unique())
        selected_crime_types = st.multiselect(
            "Select Crime Types (optional)",
            crime_types_all,
            default=crime_types_all  # all by default
        )
        if not selected_crime_types:
            selected_crime_types = crime_types_all

        # Action button (optional, but improves UX)
        run_clustering = st.button("Run Clustering")

    with col2:
        if not run_clustering:
            st.info("👈 Adjust settings and click **Run Clustering** to start.")
            return

        # === Preprocess: filter → pivot to wide format ===
        required_cols = {'Type of crime', 'Region', 'Year', 'The number of crimes (cases per 100.000 population)'}
        if not required_cols.issubset(df.columns):
            st.error("Required columns missing. Check data schema.")
            return

        df_filtered = df[
            (df['Year'] == selected_year) &
            (df['Type of crime'].isin(selected_crime_types))
        ]

        try:
            df_wide = df_filtered.pivot(
                index='Region',
                columns='Type of crime',
                values='The number of crimes (cases per 100.000 population)'
            )
        except ValueError:
            st.error("Duplicate entries detected (Region + Crime Type + Year). Please clean data.")
            return

        # Handle missing values
        if df_wide.isnull().values.any():
            st.warning("Missing values detected → filled with column means.")
            df_wide = df_wide.fillna(df_wide.mean())

        if df_wide.empty or len(df_wide) < 2:
            st.error("Not enough data for clustering (need ≥2 regions).")
            return

        # === Scale data ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_wide)
        X_scaled = pd.DataFrame(X_scaled, index=df_wide.index, columns=df_wide.columns)

        # === Run k-means ===
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df_wide['Cluster'] = cluster_labels

        # === Compute centroids (in original scale) ===
        centroids_scaled = kmeans.cluster_centers_
        centroids_original = scaler.inverse_transform(centroids_scaled)
        centroids_df = pd.DataFrame(
            centroids_original,
            columns=df_wide.columns.drop('Cluster'),
            index=[f"Cluster {i}" for i in range(n_clusters)]
        ).round(2)

        # === PCA for 2D visualization ===
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.sum()

        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': [f"Cluster {label}" for label in cluster_labels],
            'Region': df_wide.index
        })

        # === Plot: PCA scatter ===
        fig_pca = px.scatter(
            plot_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=['Region'],
            title=f"Region Clusters ({selected_year}) — PCA (Explained Variance: {explained_var:.1%})",
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
        )
        fig_pca.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_pca, use_container_width=True)

        # === Cluster centroids table ===
        st.subheader("Cluster Centroids (Crimes per 100,000 Population)")
        st.dataframe(centroids_df)

        # === Cluster size bar chart ===
        cluster_counts = plot_df['Cluster'].value_counts().sort_index()
        fig_counts = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Regions'},
            title="Cluster Sizes"
        )
        fig_counts.update_layout(showlegend=False)
        st.plotly_chart(fig_counts, use_container_width=True)

        # === Detailed results table ===
        st.subheader("Region Assignments")
        result_df = pd.DataFrame({
            'Region': df_wide.index,
            'Cluster': [f"Cluster {label}" for label in cluster_labels]
        })
        for crime in centroids_df.columns:
            result_df[crime] = df_wide[crime].values
        st.dataframe(result_df)

        # === Export ===
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Clustering Results (CSV)",
            csv,
            f"clustering_kmeans_{selected_year}.csv",
            "text/csv",
            key='download-clusters'
        )