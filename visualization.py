import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def show_data_visualization(df):
    """Display the data visualization page."""
    st.title("Crime Statistics in Russia (2008-2023)")

    col1, col2 = st.columns([1, 3])

    with col1:
        analysis_type = st.radio(
            "Analysis Type",
            ('By Region', 'By Crime Type')
        )

        if analysis_type == 'By Region':
            region = st.selectbox('Select Region', sorted(df['Region'].unique()))
            crime_types = sorted(df[df['Region'] == region]['Type of crime'].unique())
            crime_type = st.multiselect('Select Crime Types', crime_types,
                                        default=[crime_types[0]] if crime_types else [])
            metric = st.radio('Metric', ['Absolute Number', 'Per 100,000 Population'])

        elif analysis_type == 'By Crime Type':
            crime_type = st.selectbox('Select Crime Type', sorted(df['Type of crime'].unique()))
            regions = sorted(df[df['Type of crime'] == crime_type]['Region'].unique())
            selected_regions = st.multiselect('Select Regions', regions, default=[regions[0]] if regions else [])
            metric = st.radio('Metric', ['Absolute Number', 'Per 100,000 Population'])

    with col2:
        if analysis_type == 'By Region':
            if not crime_type:
                st.warning("Please select at least one crime type.")
                return

            filtered_df = df[(df['Region'] == region) & (df['Type of crime'].isin(crime_type))]

            if not filtered_df.empty:
                fig = px.line(
                    filtered_df,
                    x='Year',
                    y='The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)',
                    color='Type of crime',
                    title=f'Crime Dynamics in {region} ({metric})',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Number of Crimes' if metric == 'Absolute Number' else 'Crimes per 100,000 Population',
                    legend_title='Crime Type'
                )
                st.plotly_chart(fig, use_container_width=True)

                latest_year = filtered_df['Year'].max()
                latest_data = filtered_df[filtered_df['Year'] == latest_year]
                st.subheader(f'Most Recent Data ({latest_year})')
                st.dataframe(latest_data[['Type of crime', 'The number of crimes',
                                          'The number of crimes (cases per 100.000 population)']])
            else:
                st.warning("No data available for the selected parameters.")

        elif analysis_type == 'By Crime Type':
            if not selected_regions:
                st.warning("Please select at least one region.")
                return

            filtered_df = df[(df['Type of crime'] == crime_type) & (df['Region'].isin(selected_regions))]

            if not filtered_df.empty:
                fig = px.line(
                    filtered_df,
                    x='Year',
                    y='The number of crimes' if metric == 'Absolute Number' else 'The number of crimes (cases per 100.000 population)',
                    color='Region',
                    title=f'{crime_type} Dynamics by Region ({metric})',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Number of Crimes' if metric == 'Absolute Number' else 'Crimes per 100,000 Population',
                    legend_title='Region'
                )
                st.plotly_chart(fig, use_container_width=True)

                latest_year = filtered_df['Year'].max()
                latest_data = filtered_df[filtered_df['Year'] == latest_year]
                st.subheader(f'Most Recent Data ({latest_year})')
                st.dataframe(latest_data[['Region', 'The number of crimes',
                                          'The number of crimes (cases per 100.000 population)']])
            else:
                st.warning("No data available for the selected parameters.")