import streamlit as st
import pandas as pd

# Importing the functions from other files
from data_analysis import data_analysis_page
from visualizations import visualizations_page
from pie_chart import pie_chart_page
from preprocessing import preprocessing_page
from predictive_models import predictive_models_page

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Main Page", "Data Analysis", "Visualizations", "Pie Chart", "Preprocessing", "Predictive Models"]
selected_page = st.sidebar.radio("Select a page:", pages)

# File uploader in sidebar
st.sidebar.title("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Navigation Control
    if selected_page == "Main Page":
        st.title("CSV Data Analyzer Dashboard")
        st.header(f"Data Preview: {uploaded_file.name}")
        st.write(df.head())
    elif selected_page == "Data Analysis":
        data_analysis_page(df)
    elif selected_page == "Visualizations":
        numeric_df = df.select_dtypes(include='number')
        correlation = numeric_df.corr()
        key_variables = correlation.abs().unstack().sort_values(ascending=False).drop_duplicates()
        key_variables = key_variables[key_variables < 1].nlargest(5)
        visualizations_page(df, key_variables)
    elif selected_page == "Pie Chart":
        pie_chart_page(df)
    elif selected_page == "Preprocessing":
        df = preprocessing_page(df)
    elif selected_page == "Predictive Models":
        predictive_models_page(df)
else:
    st.title("CSV Data Analyzer Dashboard")
    st.write("Please upload a CSV file using the sidebar to begin.")
