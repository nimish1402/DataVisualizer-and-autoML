# Streamlit Data Analyzer and Predictive Modeling App

This project is a web-based application built with Streamlit that allows users to upload a CSV file, visualize data, perform data preprocessing, apply AutoML for predictive modeling, and download the entire analysis as a Jupyter Notebook (`.ipynb`).

## Features

- **Data Upload**: Users can upload their own CSV files.
- **Data Visualization**: Automatic generation of visualizations such as scatter plots, correlation matrices, and pie charts.
- **Data Preprocessing**: Tab for preprocessing data (e.g., handling missing values, encoding categorical variables).
- **AutoML Integration**: Automated machine learning using FLAML (Fast and Lightweight AutoML) to find the best model for prediction.
- **Predictive Modeling**: Allows users to select a target column for prediction and identifies the best-fit model.
- **Performance Metrics**: Displays model performance metrics including accuracy, cross-validation score, precision, recall, and F1 score.
- **Downloadable Jupyter Notebook**: Generates and allows downloading of a Jupyter Notebook file (`.ipynb`) containing all the steps performed in the app.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. **Set up a Virtual Environment**
    python -m venv .venv
    source .venv/bin/activate   # On Windows use: .venv\Scripts\activate

3. **Install Dependencies**
    Install the required Python libraries using pip:

   ``` bash
    pip install -r requirements.txt
4. **Run the App**
  Start the Streamlit app by running:

    ```bash
    streamlit run app.py

## Usage

**Upload a CSV File:** Use the sidebar to upload your CSV file.
**Explore Data:** Navigate through different tabs to explore data analysis, visualizations, and preprocessing options.
**Select Target Column for Prediction:** In the "Predictive Models" tab, select the column you want to predict.
**Start AutoML Training:** Click the button to start the automated machine learning process.
**Download Jupyter Notebook:** After the analysis, download the complete Jupyter Notebook (.ipynb) file for further exploration and customization.

## Project Structure
```bash

your-repo-name/
│
├── app.py                 # Main Streamlit app file
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
