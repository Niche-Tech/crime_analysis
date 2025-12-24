# classification_analysis.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt


def show_classification_analysis(df):
    """Display the classification analysis page with Decision Tree and k-NN."""
    st.title("Classification Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")

        # Classification type selection
        classification_type = st.radio(
            "Classification Type",
            ["Binary (High/Low Risk)", "Multi-class (Risk Levels)"]
        )

        # Target crime type
        crime_types = sorted(df['Type of crime'].unique())
        target_crime = st.selectbox("Target Crime Type", crime_types)

        # Year range
        years = sorted(df['Year'].unique())
        year_range = st.slider(
            "Training Years",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years) - 1)
        )
        train_years = list(range(year_range[0], year_range[1] + 1))
        test_year = year_range[1] + 1 if year_range[1] + 1 <= max(years) else None

        # Algorithm selection
        algorithm = st.radio(
            "Algorithm",
            ["Decision Tree", "k-Nearest Neighbors (kNN)"]
        )

        # Algorithm-specific parameters
        if algorithm == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 5)
            criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
            # NEW: Visualization depth control
            viz_depth = st.slider("Visualization Depth", 1, max_depth, min(3, max_depth))
        else:  # kNN
            n_neighbors = st.slider("Number of Neighbors (k)", 1, 30, 5)
            metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])

        # Features selection
        use_other_crimes = st.checkbox("Use other crime types as features", value=True)
        include_year = st.checkbox("Include Year as feature", value=True)

        run_model = st.button("Run Classification")

    with col2:
        if not run_model:
            st.info("👈 Configure settings and click **Run Classification** to start.")
            return

        # === Validate data structure ===
        required_cols = {'Type of crime', 'Region', 'Year', 'The number of crimes (cases per 100.000 population)'}
        if not required_cols.issubset(df.columns):
            st.error("Required columns missing. Check data schema.")
            return

        # === Prepare wide-format data ===
        try:
            def pivot_year(y):
                d = df[df['Year'] == y]
                return d.pivot(
                    index='Region',
                    columns='Type of crime',
                    values='The number of crimes (cases per 100.000 population)'
                ).reset_index()

            dfs = []
            for y in train_years:
                d = pivot_year(y)
                d['Year'] = y
                dfs.append(d)
            df_wide = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            st.error(f"Data reshaping failed: {e}")
            return

        # === Define class names ===
        binary_class_names = ["Low Risk", "High Risk"]
        multi_class_names = ["Low Risk", "Medium Risk", "High Risk"]

        # === Create target variable ===
        if classification_type == "Binary (High/Low Risk)":
            threshold = df_wide[target_crime].median()
            df_wide['Target'] = (df_wide[target_crime] > threshold).astype(int)
            class_names = binary_class_names
        else:  # Multi-class
            quantiles = np.quantile(df_wide[target_crime], [0.33, 0.66])
            df_wide['Target'] = np.select(
                [
                    df_wide[target_crime] <= quantiles[0],
                    (df_wide[target_crime] > quantiles[0]) & (df_wide[target_crime] <= quantiles[1]),
                    df_wide[target_crime] > quantiles[1]
                ],
                [0, 1, 2],
                default=2
            )
            class_names = multi_class_names

        # === AUTO-SWITCH: Check if all classes exist ===
        unique_classes = df_wide['Target'].nunique()
        if classification_type == "Multi-class (Risk Levels)" and unique_classes < 3:
            st.warning(f"⚠️ Only {unique_classes} risk classes found in training data! "
                       "Switching to binary classification.")
            threshold = df_wide[target_crime].median()
            df_wide['Target'] = (df_wide[target_crime] > threshold).astype(int)
            classification_type = "Binary (High/Low Risk)"
            class_names = binary_class_names

        # === Prepare features ===
        features = []
        if include_year:
            features.append('Year')
        if use_other_crimes:
            other_crimes = [c for c in df_wide.columns if c != target_crime and c not in ['Region', 'Year', 'Target']]
            features.extend(other_crimes)

        if not features:
            st.error("No features selected. Enable at least one feature source.")
            return

        X = df_wide[features].copy()
        y = df_wide['Target'].copy()

        # Handle missing values
        mask = X.notnull().all(axis=1) & y.notnull()
        X, y = X[mask], y[mask]
        regions = df_wide.loc[mask, 'Region'].reset_index(drop=True)

        if len(X) < 10:
            st.error("Not enough valid samples after filtering.")
            return

        # === Train-test split (temporal if possible) ===
        if test_year and test_year in df['Year'].unique():
            st.info(f"Using {test_year} as test year (temporal split).")
            try:
                df_test = pivot_year(test_year)
                # Recreate target for test year using SAME thresholds
                if classification_type == "Binary (High/Low Risk)":
                    df_test['Target'] = (df_test[target_crime] > threshold).astype(int)
                else:
                    df_test['Target'] = np.select(
                        [
                            df_test[target_crime] <= quantiles[0],
                            (df_test[target_crime] > quantiles[0]) & (df_test[target_crime] <= quantiles[1]),
                            df_test[target_crime] > quantiles[1]
                        ],
                        [0, 1, 2],
                        default=2
                    )
                X_test = df_test[features].copy()
                y_test = df_test['Target'].copy()
                test_regions = df_test['Region'].copy()
                mask_test = X_test.notnull().all(axis=1) & y_test.notnull()
                X_test, y_test, test_regions = X_test[mask_test], y_test[mask_test], test_regions[mask_test]
                X_train, y_train = X, y
                train_regions = regions
            except Exception as e:
                st.warning(f"Could not prepare temporal test set: {e}. Using random split.")
                test_year = None
        else:
            test_year = None

        if not test_year:
            X_train, X_test, y_train, y_test, train_regions, test_regions = train_test_split(
                X, y, regions,
                test_size=0.2,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            test_regions = test_regions.reset_index(drop=True)

        # === Scale features (critical for kNN) ===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === Train model ===
        if algorithm == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                random_state=42
            )
        else:  # kNN
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric
            )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # === Metrics ===
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader(f"{algorithm} Results")
        st.write(f"**Accuracy**: {accuracy:.3f}")
        st.write(f"**Precision**: {precision:.3f}")
        st.write(f"**Recall**: {recall:.3f}")
        st.write(f"**F1-Score**: {f1:.3f}")

        # === Confusion Matrix (SAFE VERSION) ===
        # Get ALL possible classes from training data
        all_possible_classes = sorted(np.unique(y_train))
        n_classes_total = len(all_possible_classes)

        # Classes actually present in test data and predictions
        present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

        # Build confusion matrix with all possible classes
        cm = confusion_matrix(y_test, y_pred, labels=range(n_classes_total))

        # Filter matrix to only present classes
        mask = np.isin(range(n_classes_total), present_classes)
        cm_filtered = cm[mask][:, mask]

        # Get class names for present classes
        dynamic_class_names = [class_names[i] for i in present_classes if i < len(class_names)]

        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(
            cm_filtered,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=dynamic_class_names,
            y=dynamic_class_names,
            title=f"Confusion Matrix ({algorithm})",
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # === NEW: Decision Tree Visualization ===
        if algorithm == "Decision Tree":
            st.subheader("Decision Tree Structure")

            # Handle class names for plot_tree (must be strings)
            class_names_str = [str(name) for name in class_names]

            # Create figure with constrained size for readability
            plt.figure(figsize=(20, 10))
            plot_tree(
                model,
                feature_names=features,
                class_names=class_names_str,
                filled=True,
                rounded=True,
                proportion=True,
                max_depth=viz_depth,
                fontsize=10
            )
            plt.title(f"Decision Tree (Max Depth: {max_depth}, Visualized Depth: {viz_depth})")

            # Render in Streamlit
            st.pyplot(plt, use_container_width=True)
            plt.close()  # Critical to prevent memory leaks

            st.caption("""
            **Interpretation Guide**:
            - **Colored nodes**: Class distribution (red = High Risk, blue = Low Risk, green = Medium Risk)
            - **Numbers**: (samples, value) = (count, [low, medium, high] distribution)
            - **Split conditions**: Feature thresholds used for decisions
            """)

        # === Feature Importance (Decision Tree only) ===
        if algorithm == "Decision Tree":
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            fig_imp = px.bar(
                feat_imp_df,
                x='Feature',
                y='Importance',
                title="Decision Tree Feature Importance"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # === kNN Specific: Distance distribution ===
        if algorithm == "k-Nearest Neighbors (kNN)" and n_neighbors > 1:
            st.subheader("Neighbor Distance Distribution")
            distances, _ = model.kneighbors(X_test_scaled)
            avg_distances = distances.mean(axis=1)

            fig_dist = px.histogram(
                avg_distances,
                nbins=20,
                title="Average Distance to k Neighbors",
                labels={'value': 'Distance', 'count': 'Count'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # === Detailed Results Table ===
        def get_class_name(label_idx):
            """Safely get class name by index"""
            if label_idx < len(class_names):
                return class_names[label_idx]
            return f"Class {label_idx}"

        st.subheader("Prediction Results")
        results_df = pd.DataFrame({
            'Region': test_regions.values,
            'Actual Class': [get_class_name(int(i)) for i in y_test],
            'Predicted Class': [get_class_name(int(i)) for i in y_pred]
        })
        st.dataframe(results_df)

        # === Export ===
        st.markdown("---")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions (CSV)",
            csv,
            f"classification_{algorithm.replace(' ', '_').lower()}_{target_crime}.csv",
            "text/csv",
            key='download-classification'
        )