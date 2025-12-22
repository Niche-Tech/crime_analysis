import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the crime data."""
    try:
        # Read the CSV file
        df = pd.read_csv('The number of crimes in Russia 2008-2023.csv', header=None)

        # If the first row contains column names and data combined, handle it
        if df.shape[1] == 1:
            # Split the single column into multiple columns
            df = df[0].str.split(',', expand=True)

        # Set column names
        if df.shape[1] >= 5:
            df.columns = ['Type of crime', 'Region', 'Year', 'The number of crimes',
                          'The number of crimes (cases per 100.000 population)']
        else:
            # Handle case with missing columns
            column_names = ['Type of crime', 'Region', 'Year', 'The number of crimes',
                            'The number of crimes (cases per 100.000 population)']
            df.columns = column_names[:df.shape[1]]

        # Remove any rows that contain the header text as data
        header_mask = df['Type of crime'].str.contains('Type of crime', na=False, case=False)
        df = df[~header_mask]

        # Convert to appropriate data types
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['The number of crimes'] = pd.to_numeric(df['The number of crimes'], errors='coerce')
        df['The number of crimes (cases per 100.000 population)'] = pd.to_numeric(
            df['The number of crimes (cases per 100.000 population)'], errors='coerce')

        # Drop rows with missing years or both crime count metrics
        df = df.dropna(subset=['Year'])
        df = df.dropna(subset=['The number of crimes', 'The number of crimes (cases per 100.000 population)'],
                       how='all')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Type of crime', 'Region', 'Year'])

        # Handle missing values
        df['The number of crimes'] = df.groupby(['Type of crime', 'Region'])['The number of crimes'].transform(
            lambda x: x.fillna(x.median()))
        df['The number of crimes (cases per 100.000 population)'] = df.groupby(['Type of crime', 'Region'])[
            'The number of crimes (cases per 100.000 population)'].transform(lambda x: x.fillna(x.median()))

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return an empty DataFrame as fallback
        return pd.DataFrame(columns=[
            'Type of crime', 'Region', 'Year',
            'The number of crimes', 'The number of crimes (cases per 100.000 population)'
        ])