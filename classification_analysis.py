import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def show_classification_analysis(df):
    """Display the classification analysis page."""
    st.title("Classification Analysis of Crime Statistics")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Analysis Parameters")

        # Select classification type
        classification_type = st.radio(
            "Classification Type",
            ('Decision Tree', 'K-means Clustering')
        )

        if classification_type == 'Decision Tree':
            st.subheader("Decision Tree Parameters")

            # Select region
            region = st.selectbox('Select Region', sorted(df['Region'].unique()), key='dt_region')

            # Select crime type for prediction
            crime_type = st.selectbox('Select Crime Type', sorted(df['Type of crime'].unique()), key='dt_crime')

            # Define threshold for classification (high/low crime)
            years = sorted(df['Year'].unique())
            year_range = st.slider('Year Range', min_value=min(years), max_value=max(years),
                                   value=(min(years), max(years)), key='dt_year_range')

            # Decision tree parameters
            max_depth = st.slider('Maximum Tree Depth', 1, 10, 3)
            min_samples_split = st.slider('Minimum Samples to Split', 2, 20, 2)

        else:  # K-means Clustering
            st.subheader("K-means Clustering Parameters")

            # Select what to cluster
            cluster_type = st.radio('Cluster By', ['Regions', 'Crime Types'])

            if cluster_type == 'Regions':
                # Select crime types to use as features
                crime_types = sorted(df['Type of crime'].unique())
                selected_crimes = st.multiselect(
                    'Select Crime Types as Features',
                    crime_types,
                    default=crime_types[:3]
                )
                year = st.selectbox('Select Year', sorted(df['Year'].unique()), key='km_year')

            else:  # Cluster Crime Types
                # Select regions to use as features
                regions = sorted(df[df['Region'] != 'Russian Federation']['Region'].unique())
                selected_regions = st.multiselect(
                    'Select Regions as Features',
                    regions,
                    default=regions[:3]
                )
                year = st.selectbox('Select Year', sorted(df['Year'].unique()), key='km_crime_year')

            # K-means parameters
            n_clusters = st.slider('Number of Clusters', 2, 10, 3)
            random_state = st.number_input('Random State', value=42, min_value=0, step=1)

    with col2:
        if classification_type == 'Decision Tree':
            # Filter data for the selected region, crime type, and year range
            filtered_df = df[
                (df['Region'] == region) &
                (df['Type of crime'] == crime_type) &
                (df['Year'].between(year_range[0], year_range[1]))
                ].sort_values('Year')

            if filtered_df.empty:
                st.warning("No data available for the selected parameters.")
            else:
                # Prepare features (year) and target (crime levels)
                X = filtered_df[['Year']].values

                # Use crimes per 100,000 population for classification
                crime_col = 'The number of crimes (cases per 100.000 population)'
                crime_values = filtered_df[crime_col].values

                # Create multi-class target based on quantiles
                quantiles = np.percentile(crime_values, [25, 50, 75])
                y = np.zeros(len(crime_values), dtype=int)
                y[crime_values > quantiles[2]] = 3  # Very High
                y[(crime_values > quantiles[1]) & (crime_values <= quantiles[2])] = 2  # High
                y[(crime_values > quantiles[0]) & (crime_values <= quantiles[1])] = 1  # Medium
                # y <= quantiles[0] remains 0 # Low

                # Create descriptive labels for the classes
                class_labels = ['Low', 'Medium', 'High', 'Very High']

                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train decision tree classifier
                dt_classifier = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                dt_classifier.fit(X_train, y_train)

                # Make predictions
                y_pred = dt_classifier.predict(X_test)
                train_pred = dt_classifier.predict(X_train)

                # Calculate accuracy
                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, y_pred)

                # Create visualization for the decision boundaries
                fig = go.Figure()

                # Add training data points with color by class
                for class_idx, class_name in enumerate(class_labels):
                    mask = y_train == class_idx
                    if np.any(mask):
                        fig.add_trace(go.Scatter(
                            x=X_train[mask].flatten(),
                            y=y_train[mask],
                            mode='markers',
                            name=f'Training ({class_name})',
                            marker=dict(size=10, symbol='circle')
                        ))

                # Add test data points with color by class
                for class_idx, class_name in enumerate(class_labels):
                    mask = y_test == class_idx
                    if np.any(mask):
                        fig.add_trace(go.Scatter(
                            x=X_test[mask].flatten(),
                            y=y_test[mask],
                            mode='markers',
                            name=f'Test ({class_name})',
                            marker=dict(size=10, symbol='triangle-up')
                        ))

                # Add decision boundaries
                x_boundary = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
                y_boundary = dt_classifier.predict(x_boundary)
                fig.add_trace(go.Scatter(
                    x=x_boundary.flatten(),
                    y=y_boundary,
                    mode='lines',
                    name='Decision Boundary',
                    line=dict(color='red', width=3, dash='dash')
                ))

                fig.update_layout(
                    title=f'Decision Tree Classification: {crime_type} in {region} (Level Classification)',
                    xaxis_title='Year',
                    yaxis_title='Crime Level',
                    yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=class_labels),
                    legend_title='Data Type'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display classification results
                st.subheader("Classification Results")

                col_acc1, col_acc2 = st.columns(2)
                with col_acc1:
                    st.write(f"**Training Accuracy:** {train_accuracy:.2f}")
                with col_acc2:
                    st.write(f"**Test Accuracy:** {test_accuracy:.2f}")

                st.write("**Classification Thresholds (Crimes per 100k population):**")
                st.write(f"- Low: ≤ {quantiles[0]:.1f}")
                st.write(f"- Medium: {quantiles[0]:.1f} - {quantiles[1]:.1f}")
                st.write(f"- High: {quantiles[1]:.1f} - {quantiles[2]:.1f}")
                st.write(f"- Very High: > {quantiles[2]:.1f}")

                # Display classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                # Display confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_labels,
                            yticklabels=class_labels,
                            ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(fig_cm)

                # Display decision tree
                st.subheader("Decision Tree Visualization")

                # Limit tree depth for visualization clarity
                vis_depth = min(max_depth, 4)
                vis_classifier = DecisionTreeClassifier(
                    max_depth=vis_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                vis_classifier.fit(X, y)

                fig_tree, ax = plt.subplots(figsize=(20, 10))
                plot_tree(vis_classifier,
                          feature_names=['Year'],
                          class_names=class_labels,
                          filled=True,
                          rounded=True,
                          fontsize=10,
                          ax=ax)
                plt.title(f'Decision Tree for {crime_type} in {region} (Depth={vis_depth})')
                st.pyplot(fig_tree)

                # Display feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': ['Year'],
                    'Importance': [dt_classifier.feature_importances_[0]]
                })
                st.dataframe(feature_importance)

        else:  # K-means Clustering
            if cluster_type == 'Regions':
                if not selected_crimes:
                    st.warning("Please select at least one crime type for clustering.")
                    return

                # Filter data for the selected year and crime types
                filtered_df = df[
                    (df['Year'] == year) &
                    (df['Type of crime'].isin(selected_crimes)) &
                    (df['Region'] != 'Russian Federation')
                    ]

                if filtered_df.empty:
                    st.warning("No data available for the selected parameters.")
                    return

                # Pivot data to have regions as rows and crime types as columns
                pivot_df = filtered_df.pivot_table(
                    index='Region',
                    columns='Type of crime',
                    values='The number of crimes (cases per 100.000 population)',
                    aggfunc='mean'
                ).reset_index()

                # Prepare features for clustering
                feature_cols = [col for col in selected_crimes if col in pivot_df.columns]
                if not feature_cols:
                    st.warning("No matching crime types found in the data. Please select different crime types.")
                    return
                X = pivot_df[feature_cols].values

            else:  # Cluster Crime Types
                if not selected_regions:
                    st.warning("Please select at least one region for clustering.")
                    return

                # Filter data for the selected year and regions
                filtered_df = df[
                    (df['Year'] == year) &
                    (df['Region'].isin(selected_regions))
                    ]

                if filtered_df.empty:
                    st.warning("No data available for the selected parameters.")
                    return

                # Pivot data to have crime types as rows and regions as columns
                pivot_df = filtered_df.pivot_table(
                    index='Type of crime',
                    columns='Region',
                    values='The number of crimes (cases per 100.000 population)',
                    aggfunc='mean'
                ).reset_index()

                # Prepare features for clustering
                feature_cols = [col for col in selected_regions if col in pivot_df.columns]
                if not feature_cols:
                    st.warning("No matching regions found in the data. Please select different regions.")
                    return
                X = pivot_df[feature_cols].values

            # Handle missing values
            if np.isnan(X).any():
                st.warning("Data contains missing values. Filling with median values.")
                for i in range(X.shape[1]):
                    col_median = np.nanmedian(X[:, i])
                    X[np.isnan(X[:, i]), i] = col_median

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            kmeans.fit(X_scaled)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            # Add cluster labels to the dataframe
            pivot_df['Cluster'] = labels

            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_scaled, labels)

            # Create visualization based on number of features
            if len(feature_cols) >= 2:
                fig = px.scatter(
                    pivot_df,
                    x=feature_cols[0],
                    y=feature_cols[1],
                    color='Cluster',
                    hover_name='Region' if cluster_type == 'Regions' else 'Type of crime',
                    title=f'K-means Clustering of {cluster_type} ({year})',
                    color_continuous_scale='Viridis'
                )

                # Add centroid markers
                centroid_df = pd.DataFrame(centroids, columns=feature_cols)
                centroid_df['Cluster'] = range(n_clusters)
                centroid_df['Size'] = 20

                fig.add_trace(go.Scatter(
                    x=centroid_df[feature_cols[0]],
                    y=centroid_df[feature_cols[1]],
                    mode='markers',
                    marker=dict(size=centroid_df['Size'], color='red', symbol='x'),
                    name='Centroids'
                ))

                st.plotly_chart(fig, use_container_width=True)

            # For 3D visualization if we have at least 3 features
            if len(feature_cols) >= 3:
                fig_3d = px.scatter_3d(
                    pivot_df,
                    x=feature_cols[0],
                    y=feature_cols[1],
                    z=feature_cols[2],
                    color='Cluster',
                    hover_name='Region' if cluster_type == 'Regions' else 'Type of crime',
                    title=f'3D K-means Clustering of {cluster_type} ({year})',
                    color_continuous_scale='Viridis'
                )

                st.plotly_chart(fig_3d, use_container_width=True)

            # Display clustering results
            st.subheader("Clustering Results")

            col_clust1, col_clust2 = st.columns(2)
            with col_clust1:
                st.write(f"**Number of Clusters:** {n_clusters}")
            with col_clust2:
                st.write(f"**Silhouette Score:** {silhouette_avg:.3f}")
                st.write("*(Higher is better, range: -1 to 1)*")

            # Display cluster distribution
            cluster_counts = pivot_df['Cluster'].value_counts().sort_index()
            fig_dist = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Items'},
                title='Cluster Distribution',
                color=cluster_counts.index
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Display cluster details
            st.subheader("Cluster Details")

            # Create a summary of each cluster
            cluster_summary = []
            for cluster_id in range(n_clusters):
                cluster_data = pivot_df[pivot_df['Cluster'] == cluster_id]
                if cluster_type == 'Regions':
                    items = ', '.join(cluster_data['Region'].tolist())
                else:
                    items = ', '.join(cluster_data['Type of crime'].tolist())

                cluster_summary.append({
                    'Cluster': cluster_id,
                    'Count': len(cluster_data),
                    'Items': items[:200] + '...' if len(items) > 200 else items
                })

            summary_df = pd.DataFrame(cluster_summary)
            st.dataframe(summary_df)

            # Display centroid values
            st.subheader("Cluster Centroids (Standardized)")

            centroid_df = pd.DataFrame(centroids, columns=feature_cols)
            centroid_df.insert(0, 'Cluster', range(n_clusters))
            st.dataframe(centroid_df)

            # Option to download results
            st.subheader("Export Results")
            csv = pivot_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustering Results as CSV",
                data=csv,
                file_name=f'clustering_results_{cluster_type.lower()}_{year}.csv',
                mime='text/csv',
            )