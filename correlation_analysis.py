import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def show_correlation_analysis(df):
    """Display the correlation analysis page with correlation matrix."""
    st.title("Correlation Analysis of Crime Statistics")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Analysis Parameters")

        # Selection type
        correlation_type = st.radio(
            "Correlation Type",
            ('Between Crime Types', 'Between Regions', 'Crime vs Time')
        )

        if correlation_type == 'Between Crime Types':
            # Select region for analysis
            region = st.selectbox('Select Region', sorted(df['Region'].unique()))

            # Select crime types
            all_crime_types = sorted(df['Type of crime'].unique())
            selected_crimes = st.multiselect(
                'Select Crime Types',
                all_crime_types,
                default=all_crime_types[:5]  # Default to first 5 crime types
            )

            # Select metric
            metric = st.radio('Metric', ['Absolute Number', 'Per 100,000 Population'])

            # Select years for analysis
            years = sorted(df['Year'].unique())
            year_range_corr = st.slider(
                'Year Range',
                min_value=min(years),
                max_value=max(years),
                value=(min(years), max(years))
            )

        elif correlation_type == 'Between Regions':
            # Select crime type
            crime_type = st.selectbox('Select Crime Type', sorted(df['Type of crime'].unique()))

            # Select regions
            all_regions = sorted(df[df['Region'] != 'Russian Federation']['Region'].unique())
            selected_regions = st.multiselect(
                'Select Regions',
                all_regions,
                default=all_regions[:5]  # Default to first 5 regions
            )

            # Select metric
            metric = st.radio('Metric', ['Absolute Number', 'Per 100,000 Population'])

            # Select years for analysis
            years = sorted(df['Year'].unique())
            year_range_corr = st.slider(
                'Year Range',
                min_value=min(years),
                max_value=max(years),
                value=(min(years), max(years))
            )

        else:  # Crime vs Time
            # Select region and crime type
            region = st.selectbox('Select Region', sorted(df['Region'].unique()), key='region_time')
            crime_type = st.selectbox('Select Crime Type', sorted(df['Type of crime'].unique()), key='crime_time')

            # Select metric and max lag
            metric = st.radio('Metric', ['Absolute Number', 'Per 100,000 Population'], key='metric_time')
            max_lag = st.slider('Maximum Lag (Years)', 0, 5, 3)

    with col2:
        if correlation_type == 'Between Crime Types':
            if len(selected_crimes) < 2:
                st.warning("Please select at least 2 crime types for correlation analysis.")
            else:
                # Filter data
                filtered_df = df[
                    (df['Region'] == region) &
                    (df['Type of crime'].isin(selected_crimes)) &
                    (df['Year'].between(year_range_corr[0], year_range_corr[1]))
                    ]

                if not filtered_df.empty:
                    # Pivot data for correlation
                    pivot_df = filtered_df.pivot(
                        index='Year',
                        columns='Type of crime',
                        values='The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)'
                    )

                    # Calculate correlation matrix
                    corr_matrix = pivot_df.corr()

                    # Display correlation matrix
                    st.subheader(
                        f"Correlation Matrix: Crime Types in {region} ({year_range_corr[0]}-{year_range_corr[1]})")

                    # Create heatmap with plotly
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title=f"Correlation Between Crime Types in {region}"
                    )
                    fig.update_layout(width=700, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display correlation values as a table
                    st.subheader("Correlation Values")
                    st.dataframe(corr_matrix)
                else:
                    st.warning("No data available for the selected parameters.")

        elif correlation_type == 'Between Regions':
            if len(selected_regions) < 2:
                st.warning("Please select at least 2 regions for correlation analysis.")
            else:
                # Filter data
                filtered_df = df[
                    (df['Type of crime'] == crime_type) &
                    (df['Region'].isin(selected_regions)) &
                    (df['Year'].between(year_range_corr[0], year_range_corr[1]))
                    ]

                if not filtered_df.empty:
                    # Pivot data for correlation
                    pivot_df = filtered_df.pivot(
                        index='Year',
                        columns='Region',
                        values='The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)'
                    )

                    # Calculate correlation matrix
                    corr_matrix = pivot_df.corr()

                    # Display correlation matrix
                    st.subheader(
                        f"Correlation Matrix: {crime_type} Across Regions ({year_range_corr[0]}-{year_range_corr[1]})")

                    # Create heatmap with plotly
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title=f"Regional Correlation for {crime_type}"
                    )
                    fig.update_layout(width=700, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display correlation values as a table
                    st.subheader("Correlation Values")
                    st.dataframe(corr_matrix)
                else:
                    st.warning("No data available for the selected parameters.")

        else:  # Crime vs Time (Lagged Correlation)
            # Filter data for selected region and crime type
            filtered_df = df[
                (df['Region'] == region) &
                (df['Type of crime'] == crime_type)
                ].sort_values('Year')

            if not filtered_df.empty:
                # Extract time series data
                time_series = filtered_df.set_index('Year')[
                    'The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)'
                ]

                # Create dataframe with lags
                ts_df = pd.DataFrame({'Year': time_series.index, 'Value': time_series.values})

                # Add lagged columns
                for lag in range(1, max_lag + 1):
                    ts_df[f'Lag_{lag}'] = time_series.shift(lag).values

                # Drop NaN values
                ts_df = ts_df.dropna()

                if not ts_df.empty:
                    # Prepare data for correlation matrix
                    lag_columns = ['Value'] + [f'Lag_{i}' for i in range(1, max_lag + 1)]
                    lag_df = ts_df[lag_columns]

                    # Calculate correlation matrix
                    corr_matrix = lag_df.corr()

                    # Display correlation matrix
                    st.subheader(f"Correlation Matrix: {crime_type} in {region} with Time Lags")

                    # Create heatmap with plotly
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title=f"Time Lag Correlation for {crime_type} in {region}"
                    )
                    fig.update_layout(width=700, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display correlation values as a table
                    st.subheader("Correlation Values")
                    st.dataframe(corr_matrix)
                else:
                    st.warning("Not enough data points after adding lags. Try reducing the maximum lag.")
            else:
                st.warning("No data available for the selected parameters.")