import streamlit as st
import pandas as pd

def data_analysis_page(df):
    st.title("Data Analysis")
    numeric_df = df.select_dtypes(include='number')

    st.write("Basic Statistics:")
    st.write(df.describe())

    correlation = numeric_df.corr()
    st.write("Correlation Matrix:")
    st.write(correlation)

    key_variables = correlation.abs().unstack().sort_values(ascending=False).drop_duplicates()
    key_variables = key_variables[key_variables < 1].nlargest(5)  # Remove self-correlation (1.0)

    st.write("Key Variables (Top 5 correlations):")
    st.write(key_variables)
