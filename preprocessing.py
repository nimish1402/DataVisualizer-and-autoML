import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocessing_page(df):
    st.title("Preprocessing")

    st.write("Handling missing values...")
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()

    st.write("Encoding categorical variables...")
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    st.write("Normalizing numeric features...")
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    st.write("Data after preprocessing:")
    st.write(df.head())
    
    return df
