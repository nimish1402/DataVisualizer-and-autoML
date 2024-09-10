import streamlit as st
import pandas as pd
import pickle
import nbformat as nbf
from flaml import AutoML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def generate_notebook_code(df, target_column, best_model_name, test_accuracy, mean_cv_score, precision, recall, f1, classification_report_text):
    # Create a new notebook object
    nb = nbf.v4.new_notebook()
    
    # Add import statements
    nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\n"
                                         "import seaborn as sns\n"
                                         "import matplotlib.pyplot as plt\n"
                                         "from flaml import AutoML\n"
                                         "from sklearn.model_selection import train_test_split\n"
                                         "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report"))

    # Add code to load and display the data
    nb.cells.append(nbf.v4.new_code_cell(f"# Step 1: Load the Data\n"
                                         "df = pd.DataFrame(...)  # Add your data loading logic here\n"
                                         "print(df.head())"))

    # Data visualization
    nb.cells.append(nbf.v4.new_code_cell("# Step 2: Data Visualization\n"
                                         "sns.pairplot(df)\n"
                                         "plt.show()"))

    # Add code for data preprocessing
    nb.cells.append(nbf.v4.new_code_cell(f"# Step 3: Data Preprocessing\n"
                                         "X = df.drop(columns=[target_column])\n"
                                         "y = df[target_column]\n"
                                         "X = pd.get_dummies(X, drop_first=True)\n"
                                         "y = y.astype('category').cat.codes"))

    # Add code for train-test split
    nb.cells.append(nbf.v4.new_code_cell("# Step 4: Train-Test Split\n"
                                         "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"))

    # Add code for prediction using AutoML
    nb.cells.append(nbf.v4.new_code_cell("# Step 5: Model Training with AutoML\n"
                                         "automl = AutoML()\n"
                                         "automl.fit(X_train=X_train, y_train=y_train, time_budget=60, task='classification')\n"
                                         "best_model = automl.model.estimator\n"
                                         "print(f'Best model: {automl.best_estimator}')"))

    # Add code for evaluation
    nb.cells.append(nbf.v4.new_code_cell(f"# Step 6: Model Evaluation\n"
                                         "y_pred = best_model.predict(X_test)\n"
                                         "accuracy = accuracy_score(y_test, y_pred)\n"
                                         "precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)\n"
                                         "recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)\n"
                                         "f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)\n"
                                         "print(f'Accuracy: {accuracy}')\n"
                                         "print(f'Precision: {precision}')\n"
                                         "print(f'Recall: {recall}')\n"
                                         "print(f'F1 Score: {f1}')\n"
                                         "print(classification_report(y_test, y_pred, zero_division=1))"))

    # Return the notebook object
    return nb

def predictive_models_page(df):
    st.title("Predictive Models")

    # Step 1: Ask user to select the target column for prediction
    target_column = st.selectbox("Select target column for prediction:", df.columns)

    # Step 2: Start prediction only if a target column is selected
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Display a button to start the prediction process
        if st.button("Start Prediction"):
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Find the minimum class size in the training data
            min_class_size = y_train.value_counts().min()

            # Dynamically set the number of folds for cross-validation
            cv_folds = min(3, min_class_size)  # Use at least 3 folds or less depending on the smallest class size

            # Initialize and configure AutoML
            automl = AutoML()
            settings = {
                "time_budget": 60,  # Time budget for AutoML in seconds
                "metric": 'accuracy',  # Metric to optimize
                "task": 'classification',  # Task type (classification)
                "log_file_name": "flaml.log",
                "estimator_list": ['lgbm', 'rf', 'xgboost'],  # List of estimators to consider
                "seed": 42,
            }

            try:
                # Train the model with AutoML
                automl.fit(X_train=X_train, y_train=y_train, **settings)

                # Get the best model
                best_model = automl.model.estimator

                # Evaluate the best model using cross-validation
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                mean_cv_score = cv_scores.mean()

                # Test the best model on the test data
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

                # Display the best model details and performance metrics
                st.write("**Best ML Model:**", automl.best_estimator)
                st.write("**Model Accuracy on Test Set:**", round(test_accuracy, 3))
                st.write("**Cross-Validation Score (Mean Accuracy):**", round(mean_cv_score, 3))
                st.write("**Precision:**", round(precision, 3))
                st.write("**Recall:**", round(recall, 3))
                st.write("**F1 Score:**", round(f1, 3))

                st.write("**Classification Report:**")
                st.text(classification_report(y_test, y_pred, zero_division=1))

                # Generate the notebook content
                nb = generate_notebook_code(df, target_column, automl.best_estimator, test_accuracy, mean_cv_score, precision, recall, f1, classification_report(y_test, y_pred, zero_division=1))

                # Save the notebook to a file
                with open("model_analysis.ipynb", "w") as f:
                    nbf.write(nb, f)

                # Provide the download button
                st.download_button("Download the Model", data=open("model_analysis.ipynb", "rb").read(), file_name="model_analysis.ipynb")

            except Exception as e:
                st.write(f"An error occurred during AutoML training: {e}")
        else:
            st.write("Click the button to start the prediction process.")
    else:
        st.write("Please select a target column to proceed.")
