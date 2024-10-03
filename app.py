import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Config
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f4f7f6;
        color: #333;
    }
    .stApp {
        background: #f4f7f6;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    .header {
        background-color: ;
        padding: 10px;
        text-align: center;
        color: white;
    }
    .section {
        background-color: #eafaf1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<div class='header'><h1>Breast Cancer Prediction App</h1></div>", unsafe_allow_html=True)
st.write("""
This app predicts whether a tumor is **malignant** or **benign** using a machine learning model trained on the Breast Cancer dataset. Use the sidebar to navigate.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Exploratory Analysis", "Model Training", "Prediction"])

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv('data (1).csv')  # Update the path
    return data

data = load_data()

# Dataset Overview
if page == "Dataset Overview":
    st.markdown("<div class='section'><h3>Dataset Overview</h3></div>", unsafe_allow_html=True)
    st.write("### Dataset Preview")
    st.dataframe(data.head())
    st.write("### Data Statistics")
    st.write(data.describe())

# Exploratory Data Analysis (EDA)
if page == "Exploratory Analysis":
    st.markdown("<div class='section'><h3>Exploratory Data Analysis (EDA)</h3></div>", unsafe_allow_html=True)
    
    st.write("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    st.write("### Dataset Distribution")
    plt.figure(figsize=(6, 4))
    sns.histplot(data['diagnosis'], kde=False, palette='Set2', color='purple')
    st.pyplot(plt)

# Model Training
if page == "Model Training":
    st.markdown("<div class='section'><h3>Model Training</h3></div>", unsafe_allow_html=True)
    
    # Feature and Label Selection
    target_column = 'diagnosis'  # Replace with correct target column
    X = data.drop(columns=[target_column])  # Drop the target column
    y = data[target_column]

    # Encode target variable
    if y.dtype == 'object':
        y = y.map({'M': 1, 'B': 0})  # Map 'M' (malignant) to 1, 'B' (benign) to 0

    # Split data
    test_size = st.slider("Select test data percentage:", 10, 50, 30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    st.pyplot(plt)

# Prediction
if page == "Prediction":
    st.markdown("<div class='section'><h3>Make Predictions</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(new_data.head())

        if st.button("Predict"):
            new_predictions = model.predict(new_data)
            st.write("Predictions (0 = Benign, 1 = Malignant):")
            st.write(new_predictions)

