import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm


def show_regression_analysis(df):
    """Display the regression analysis page."""
    st.title("Regression Analysis of Crime Statistics")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Analysis Parameters")

        # Select regression type
        regression_type = st.radio(
            "Regression Type",
            ('Simple Linear Regression', 'Multiple Linear Regression')
        )

        # Common parameters for both regression types
        st.subheader("Common Parameters")

        # Select region
        region = st.selectbox('Select Region', sorted(df['Region'].unique()), key='reg_region')

        # Select crime type
        crime_type = st.selectbox('Select Crime Type', sorted(df['Type of crime'].unique()), key='reg_crime')

        # Select metric
        metric = st.radio('Metric', ['Absolute Number', 'Per 100,000 Population'], key='reg_metric')

        # Select years for analysis
        years = sorted(df['Year'].unique())
        year_range_reg = st.slider(
            'Year Range',
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years)),
            key='reg_year_range'
        )

        # Regression type specific parameters
        if regression_type == 'Simple Linear Regression':
            st.subheader("Simple Linear Regression Parameters")
            # For simple linear regression, we'll use Year as the predictor
            predictor = st.selectbox('Predictor Variable', ['Year'], disabled=True)

        else:  # Multiple Linear Regression
            st.subheader("Multiple Linear Regression Parameters")

            # For multiple regression, we need to select additional predictors
            st.write(
                "Note: For multiple regression, we'll analyze how different crime types in the same region correlate with each other over time.")

            all_crime_types = sorted(df['Type of crime'].unique())
            target_crime = st.selectbox('Target Crime Type', all_crime_types, key='target_crime')

            predictor_crimes = st.multiselect(
                'Predictor Crime Types',
                [crime for crime in all_crime_types if crime != target_crime],
                default=all_crime_types[:3] if target_crime not in all_crime_types[:3] else all_crime_types[1:4],
                key='predictor_crimes'
            )

            # Model options
            st.subheader("Model Options")
            include_intercept = st.checkbox('Include Intercept', value=True)
            polynomial_degree = st.slider('Polynomial Degree', 1, 3, 1)

    with col2:
        if regression_type == 'Simple Linear Regression':
            # Filter data for the selected region, crime type, and year range
            filtered_df = df[
                (df['Region'] == region) &
                (df['Type of crime'] == crime_type) &
                (df['Year'].between(year_range_reg[0], year_range_reg[1]))
                ].sort_values('Year')

            if filtered_df.empty:
                st.warning("No data available for the selected parameters.")
            else:
                # Prepare data for regression
                X = filtered_df[['Year']].values
                y = filtered_df[
                    'The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)'].values

                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Create predictions
                X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_pred = model.predict(X_pred)

                # Calculate metrics
                r2 = model.score(X, y)
                mse = mean_squared_error(y, model.predict(X))

                # Create visualization
                fig = go.Figure()

                # Add actual data points
                fig.add_trace(go.Scatter(
                    x=X.flatten(),
                    y=y,
                    mode='markers',
                    name='Actual Data',
                    marker=dict(size=10, color='blue')
                ))

                # Add regression line
                fig.add_trace(go.Scatter(
                    x=X_pred.flatten(),
                    y=y_pred,
                    mode='lines',
                    name=f'Linear Regression (R² = {r2:.3f})',
                    line=dict(color='red', width=3)
                ))

                # Add predicted points for future years
                future_years = np.array([X.max() + i for i in range(1, 6)]).reshape(-1, 1)
                future_predictions = model.predict(future_years)
                fig.add_trace(go.Scatter(
                    x=future_years.flatten(),
                    y=future_predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=10, color='green', symbol='star')
                ))

                # Add confidence intervals (approximate)
                std_error = np.sqrt(mse)
                upper_bound = y_pred + 1.96 * std_error
                lower_bound = y_pred - 1.96 * std_error

                fig.add_trace(go.Scatter(
                    x=np.concatenate([X_pred.flatten(), X_pred.flatten()[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name='95% Confidence Interval'
                ))

                fig.update_layout(
                    title=f'Simple Linear Regression: {crime_type} in {region} ({metric})',
                    xaxis_title='Year',
                    yaxis_title='Number of Crimes' if metric == 'Absolute Number' else 'Crimes per 100,000 Population',
                    legend_title='Legend'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display regression results
                st.subheader("Regression Results")
                col_results1, col_results2 = st.columns(2)

                with col_results1:
                    st.write("**Model Coefficients:**")
                    st.write(f"Intercept (β₀): {model.intercept_:.4f}")
                    st.write(f"Slope (β₁): {model.coef_[0]:.4f}")

                with col_results2:
                    st.write("**Model Performance:**")
                    st.write(f"R-squared (R²): {r2:.4f}")
                    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                    st.write(f"Root MSE (RMSE): {np.sqrt(mse):.4f}")

                # Display equation
                st.write(f"**Regression Equation:**")
                st.latex(f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f} \\times Year")

                # Display predictions for future years
                st.subheader("Future Predictions")
                future_years = st.slider('Number of Future Years to Predict', 1, 5, 3)
                last_year = int(X.max())
                future_X = np.array([last_year + i for i in range(1, future_years + 1)]).reshape(-1, 1)
                future_predictions = model.predict(future_X)

                pred_df = pd.DataFrame({
                    'Year': future_X.flatten(),
                    'Predicted Crimes': future_predictions
                })

                st.dataframe(pred_df)

        else:  # Multiple Linear Regression
            if len(predictor_crimes) < 1:
                st.warning("Please select at least one predictor crime type for multiple regression analysis.")
            else:
                # Filter data for the selected region and year range
                filtered_df = df[
                    (df['Region'] == region) &
                    (df['Year'].between(year_range_reg[0], year_range_reg[1]))
                    ].sort_values('Year')

                if filtered_df.empty:
                    st.warning("No data available for the selected parameters.")
                else:
                    # Create pivot table with years as rows and crime types as columns
                    pivot_df = filtered_df.pivot_table(
                        index='Year',
                        columns='Type of crime',
                        values='The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)',
                        aggfunc='mean'
                    ).reset_index()

                    # Check if all required crime types are present
                    required_columns = [target_crime] + predictor_crimes
                    missing_columns = [col for col in required_columns if col not in pivot_df.columns]

                    if missing_columns:
                        st.warning(
                            f"The following crime types are not available for the selected region and year range: {', '.join(missing_columns)}")
                    else:
                        # Prepare data for regression
                        X = pivot_df[predictor_crimes].values
                        y = pivot_df[target_crime].values

                        # Remove rows with NaN values
                        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                        X = X[mask]
                        y = y[mask]

                        # Check if we have enough data points
                        if X.shape[0] < X.shape[1] + 1:  # +1 for intercept
                            st.warning(
                                f"Not enough data points for regression. Need at least {X.shape[1] + 1} data points, but have only {X.shape[0]}.")
                            return

                        # Apply polynomial features if degree > 1
                        feature_names = predictor_crimes.copy()
                        if polynomial_degree > 1:
                            poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
                            X_poly = poly.fit_transform(X)
                            feature_names = poly.get_feature_names_out(predictor_crimes)
                        else:
                            X_poly = X

                        # Add intercept if selected
                        if include_intercept:
                            X_final = sm.add_constant(X_poly)
                            feature_names = ['Intercept'] + list(feature_names)
                        else:
                            X_final = X_poly

                        try:
                            # Fit multiple linear regression model
                            model = sm.OLS(y, X_final).fit()

                            # Create predictions
                            y_pred = model.predict(X_final)

                            # Create visualization
                            fig = go.Figure()

                            # Add actual data points
                            fig.add_trace(go.Scatter(
                                x=pivot_df.loc[mask, 'Year'],
                                y=y,
                                mode='markers+lines',
                                name='Actual Data',
                                marker=dict(size=8, color='blue'),
                                line=dict(color='blue', dash='dash')
                            ))

                            # Add predicted values
                            fig.add_trace(go.Scatter(
                                x=pivot_df.loc[mask, 'Year'],
                                y=y_pred,
                                mode='lines',
                                name=f'Predicted Values (R² = {model.rsquared:.3f})',
                                line=dict(color='red', width=3)
                            ))

                            fig.update_layout(
                                title=f'Multiple Linear Regression: {target_crime} in {region} ({metric})',
                                xaxis_title='Year',
                                yaxis_title='Number of Crimes' if metric == 'Absolute Number' else 'Crimes per 100,000 Population',
                                legend_title='Legend'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Display regression results - FIXED VERSION
                            st.subheader("Regression Results")

                            # Get model parameters ensuring consistent lengths
                            params = model.params
                            bse = model.bse
                            tvalues = model.tvalues
                            pvalues = model.pvalues
                            conf_int = model.conf_int()

                            # Determine the number of parameters
                            n_params = len(params)

                            # Create aligned arrays for the DataFrame
                            summary_data = {
                                'Coefficient': params[:n_params],
                                'Standard Error': bse[:n_params] if len(bse) >= n_params else [None] * n_params,
                                't-value': tvalues[:n_params] if len(tvalues) >= n_params else [None] * n_params,
                                'p-value': pvalues[:n_params] if len(pvalues) >= n_params else [None] * n_params,
                                '95% CI Lower': conf_int[:n_params, 0] if len(
                                    conf_int) >= n_params else [None] * n_params,
                                '95% CI Upper': conf_int[:n_params, 1] if len(
                                    conf_int) >= n_params else [None] * n_params
                            }

                            # Create DataFrame with proper index
                            summary_df = pd.DataFrame(summary_data)

                            # Truncate or extend feature_names to match the number of parameters
                            if len(feature_names) > n_params:
                                feature_names = feature_names[:n_params]
                            elif len(feature_names) < n_params:
                                # Add placeholder names for extra parameters
                                feature_names = feature_names + [f'Feature_{i}' for i in
                                                                 range(len(feature_names), n_params)]

                            summary_df.index = feature_names[:n_params]

                            st.write("**Model Summary:**")
                            st.dataframe(summary_df)

                            # Display model performance metrics
                            st.write("**Model Performance:**")
                            metrics_df = pd.DataFrame({
                                'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'Prob (F-statistic)',
                                           'AIC', 'BIC'],
                                'Value': [
                                    model.rsquared,
                                    model.rsquared_adj,
                                    model.fvalue,
                                    model.f_pvalue,
                                    model.aic,
                                    model.bic
                                ]
                            })
                            st.dataframe(metrics_df)

                            # Display regression equation
                            st.write("**Regression Equation:**")
                            equation = f"{target_crime} = {params[0]:.4f}"
                            for i, coef in enumerate(params[1:], 1):
                                if i < len(feature_names):
                                    feature_name = feature_names[i]
                                    equation += f" + ({coef:.4f} × {feature_name})"

                            st.text(equation)

                            # Residual analysis
                            st.subheader("Residual Analysis")

                            residuals = y - y_pred
                            fig_resid = px.scatter(
                                x=y_pred,
                                y=residuals,
                                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                title='Residuals vs Predicted Values'
                            )
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_resid, use_container_width=True)

                            # Residual distribution
                            fig_dist = px.histogram(
                                residuals,
                                nbins=20,
                                title='Residual Distribution',
                                labels={'value': 'Residuals', 'count': 'Frequency'}
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error in regression analysis: {str(e)}")
                            st.write("Debug information:")
                            st.write(f"X_final shape: {X_final.shape}")
                            st.write(f"y shape: {y.shape}")
                            st.write(f"Number of features: {len(feature_names)}")
                            st.write(f"Feature names: {feature_names}")