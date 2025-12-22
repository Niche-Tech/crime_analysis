import streamlit as st
from data_preprocessing import load_and_preprocess_data
from visualization import show_data_visualization
from correlation_analysis import show_correlation_analysis
from regression_analysis import show_regression_analysis
from classification_analysis import show_classification_analysis
from clustering_analysis import show_clustering_analysis
from neural_networks import show_neural_networks


def main():
    df = load_and_preprocess_data()

    st.set_page_config(layout="wide", page_title="Crime Statistics in Russia", page_icon="📊")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Data Visualization",
        "Correlation Analysis",
        "Regression Analysis",
        "Classification Analysis",
        "Clustering Analysis",
        "Neural Networks"  # ← новая строка
    ])

    if page == "Data Visualization":
        show_data_visualization(df)
    elif page == "Correlation Analysis":
        show_correlation_analysis(df)
    elif page == "Regression Analysis":
        show_regression_analysis(df)
    elif page == "Classification Analysis":
        show_classification_analysis(df)
    elif page == "Clustering Analysis":
        show_clustering_analysis(df)
    elif page == "Neural Networks":
        show_neural_networks(df)


if __name__ == "__main__":
    main()