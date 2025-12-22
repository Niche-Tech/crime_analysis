# neural_networks.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from scipy import stats


def show_neural_networks(df):
    """Display the neural networks analysis page."""
    st.title("Neural Networks")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")

        # Task selection
        task = st.radio(
            "Task Type",
            ['Regression (Predict crime rate)', 'Classification (Risk level)']
        )

        # Year range selection
        years = sorted(df['Year'].unique())
        year_range = st.slider(
            "Training Years",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years) - 1)  # leave last year for test
        )
        train_years = list(range(year_range[0], year_range[1] + 1))
        test_year = year_range[1] + 1 if year_range[1] + 1 <= max(years) else None

        # Crime type selection
        crime_types = sorted(df['Type of crime'].unique())
        selected_crime = st.selectbox("Target Crime Type", crime_types)

        # Feature selection (other crime types as predictors)
        other_crimes = [c for c in crime_types if c != selected_crime]
        use_other_crimes = st.checkbox("Use other crime types as features", value=True)
        features_list = other_crimes if use_other_crimes else []

        # Include year as feature?
        include_year = st.checkbox("Include Year as feature", value=True)

        # Model settings
        st.markdown("---")
        st.write("**Neural Network Architecture**")
        hidden_layers_preset = st.selectbox(
            "Hidden Layers",
            [
                "(32)",
                "(64, 32)",
                "(128, 64, 32)",
                "(16)"
            ],
            index=1
        )
        hidden_layers = eval(hidden_layers_preset)

        # Task-specific settings
        if task == 'Regression (Predict crime rate)':
            metric_choice = st.radio("Target Metric",
                                     ['Absolute Number', 'Per 100,000 Population'],
                                     index=1)
            target_col = 'The number of crimes' if metric_choice == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)'

        else:  # Classification
            n_classes = st.radio("Risk Groups", [2, 3, 4], format_func=lambda x: f"{x} classes")
            threshold_method = st.selectbox("Thresholding", ["Quantile-based", "Fixed thresholds"])

        # Run button
        run_model = st.button("Train Neural Network")

    with col2:
        if not run_model:
            st.info("👈 Configure settings and click **Train Neural Network** to start.")
            return

        # === Validate data ===
        required_cols = {'Type of crime', 'Region', 'Year', 'The number of crimes', 'The number of crimes (cases per 100.000 population)'}
        if not required_cols.issubset(df.columns):
            st.error("Required columns missing. Check data schema.")
            return

        # === Prepare wide-format data per year ===
        try:
            # Pivot: Region × Crime Type → per year
            def pivot_year(y):
                d = df[df['Year'] == y]
                return d.pivot(index='Region', columns='Type of crime',
                               values='The number of crimes (cases per 100.000 population)').reset_index()

            # Combine years
            dfs = []
            for y in train_years:
                d = pivot_year(y)
                d['Year'] = y
                dfs.append(d)
            if not dfs:
                st.error("No training data for selected years.")
                return
            df_wide = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            st.error(f"Failed to reshape data: {e}")
            return

        # === Prepare features & target ===
        all_features = []
        if include_year:
            all_features.append('Year')
        if features_list:
            all_features.extend(features_list)

        # Check if features exist in data
        missing_feats = [f for f in all_features if f not in df_wide.columns]
        if missing_feats:
            st.warning(f"Missing features: {missing_feats}. Skipping them.")
            all_features = [f for f in all_features if f in df_wide.columns]

        if not all_features:
            st.error("No valid features selected. Please enable at least one feature source.")
            return

        X = df_wide[all_features].copy()
        y_col = 'The number of crimes (cases per 100.000 population)'  # always per 100k for stability
        y = df_wide[selected_crime].copy()

        # Handle missing target
        mask = y.notnull() & X.notnull().all(axis=1)
        X, y = X[mask], y[mask]

        if len(X) < 10:
            st.error("Not enough valid samples after filtering.")
            return

        # === Task-specific preprocessing ===
        if task == 'Regression (Predict crime rate)':
            # Train-test split (temporal: latest year as test if available)
            if test_year and test_year in df['Year'].unique():
                st.info(f"Using {test_year} as test year (temporal split).")
                # Prepare test set
                try:
                    df_test_wide = pivot_year(test_year)
                    X_test = df_test_wide[all_features].copy()
                    y_test = df_test_wide[selected_crime].copy()
                    mask_test = y_test.notnull() & X_test.notnull().all(axis=1)
                    X_test, y_test = X_test[mask_test], y_test[mask_test]
                except:
                    st.warning("Could not prepare test set. Using random split.")
                    test_year = None

            if not test_year:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X, y

            # Scale features
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            # Train NN
            nn_reg = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            nn_reg.fit(X_train_scaled, y_train)

            # Baseline
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)

            # Predict
            y_pred_nn = nn_reg.predict(X_test_scaled)
            y_pred_lr = lr.predict(X_test_scaled)

            # Metrics
            nn_mae = mean_absolute_error(y_test, y_pred_nn)
            nn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))
            nn_r2 = r2_score(y_test, y_pred_nn)

            lr_mae = mean_absolute_error(y_test, y_pred_lr)
            lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
            lr_r2 = r2_score(y_test, y_pred_lr)

            # === Results ===
            st.subheader("Neural Network vs Linear Regression")

            # Loss curve (if available)
            if hasattr(nn_reg, 'loss_curve_'):
                fig_loss = px.line(
                    y=nn_reg.loss_curve_,
                    title="Training Loss Curve",
                    labels={'x': 'Epoch', 'y': 'Loss'}
                )
                fig_loss.update_layout(showlegend=False)
                st.plotly_chart(fig_loss, use_container_width=True)

            # Predicted vs Actual
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test, y=y_pred_nn, mode='markers', name='Neural Network',
                marker=dict(color='blue', size=8)
            ))
            fig_scatter.add_trace(go.Scatter(
                x=y_test, y=y_pred_lr, mode='markers', name='Linear Regression',
                marker=dict(color='orange', size=6, symbol='x')
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                mode='lines', name='Ideal', line=dict(dash='dash', color='gray')
            ))
            fig_scatter.update_layout(
                title="Actual vs Predicted (Crime Rate per 100k)",
                xaxis_title="Actual",
                yaxis_title="Predicted"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Metrics table
            metrics_df = pd.DataFrame({
                'Model': ['Neural Network', 'Linear Regression'],
                'MAE': [nn_mae, lr_mae],
                'RMSE': [nn_rmse, lr_rmse],
                'R²': [nn_r2, lr_r2]
            }).round(3)
            st.dataframe(metrics_df)

            # Test set predictions
            if test_year:
                st.subheader(f"Predictions for {test_year}")
                pred_df = pd.DataFrame({
                    'Region': df_test_wide.loc[mask_test, 'Region'],
                    'Actual': y_test.values,
                    'Predicted (NN)': y_pred_nn
                })
                st.dataframe(pred_df)

        else:  # Classification
            # Create risk classes
            if threshold_method == "Quantile-based":
                quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
                thresholds = np.quantile(y, quantiles)
            else:  # Fixed thresholds (example)
                thresholds = {2: [20], 3: [10, 30], 4: [5, 15, 30]}.get(n_classes, [10, 20, 30])[:n_classes-1]

            # Assign labels
            labels = np.digitize(y, thresholds)
            le = LabelEncoder()
            y_class = le.fit_transform(labels)
            class_names = [f"Class {i}" for i in range(n_classes)]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_class, test_size=0.2, stratify=y_class, random_state=42
            )

            # Scale
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            # Train NN
            nn_clf = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            nn_clf.fit(X_train_scaled, y_train)

            # Baseline
            dummy = DummyClassifier(strategy='stratified')
            dummy.fit(X_train_scaled, y_train)

            # Predict
            y_pred_nn = nn_clf.predict(X_test_scaled)
            y_pred_proba_nn = nn_clf.predict_proba(X_test_scaled) if n_classes == 2 else None
            y_pred_dummy = dummy.predict(X_test_scaled)

            # Metrics
            nn_acc = accuracy_score(y_test, y_pred_nn)
            dummy_acc = accuracy_score(y_test, y_pred_dummy)

            # === Results ===
            st.subheader("Neural Network vs Baseline")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_nn)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=class_names,
                y=class_names,
                title="Confusion Matrix (Neural Network)"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Metrics
            st.write(f"**Neural Network Accuracy**: {nn_acc:.3f}")
            st.write(f"**Baseline (Stratified) Accuracy**: {dummy_acc:.3f}")

            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred_nn, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df)

        # === Export ===
        st.markdown("---")
        st.subheader("Export")

        if task == 'Regression (Predict crime rate)':
            if test_year:
                # Temporal split: use test year data
                export_df = pd.DataFrame({
                    'Region': df_test_wide.loc[mask_test, 'Region'].values,
                    'Year': [test_year] * len(y_test),
                    'Actual': y_test.values,
                    'Predicted_NN': y_pred_nn,
                    'Predicted_Linear': y_pred_lr
                })
            else:
                # Random split: get regions from X_test index
                export_df = pd.DataFrame({
                    'Region': X_test.index.astype(str) if hasattr(X_test, 'index') else [f"Region_{i}" for i in
                                                                                         range(len(y_test))],
                    'Year': X_test['Year'].values if 'Year' in X_test.columns else None,
                    'Actual': y_test.values,
                    'Predicted_NN': y_pred_nn,
                    'Predicted_Linear': y_pred_lr
                })
        else:  # Classification
            export_df = pd.DataFrame({
                'Region': X_test.index.astype(str) if hasattr(X_test, 'index') else [f"Region_{i}" for i in
                                                                                     range(len(y_test))],
                'Actual_Class': le.inverse_transform(y_test),
                'Predicted_Class': le.inverse_transform(y_pred_nn)
            })

        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions (CSV)",
            csv,
            f"nn_predictions_{selected_crime}_{task.split()[0].lower()}.csv",
            "text/csv",
            key='download-nn-predictions'
        )