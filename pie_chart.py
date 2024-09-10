import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def pie_chart_page(df):
    st.title("Pie Chart")

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        selected_column = st.sidebar.selectbox("Select a categorical column for Pie Chart:", categorical_columns)

        pie_data = df[selected_column].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.axis('equal')
        st.pyplot(plt)
    else:
        st.write("No categorical columns available for pie chart.")
