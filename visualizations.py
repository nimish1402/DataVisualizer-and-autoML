import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def visualizations_page(df, key_variables):
    st.title("Visualizations")
    numeric_df = df.select_dtypes(include='number')

    correlation = numeric_df.corr()

    st.write("Correlation Heatmap:")
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    for (var1, var2) in key_variables.index:
        st.write(f"Scatter Plot: {var1} vs {var2}")
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=numeric_df, x=var1, y=var2)
        st.pyplot(plt)
