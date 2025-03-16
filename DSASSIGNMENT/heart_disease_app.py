import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_heart.csv")  # Change filename if needed
    return df

df = load_data()

# Streamlit UI
st.title("Heart Disease Prediction App")

# Display dataset info
st.subheader("Dataset Overview")
if st.checkbox("Show raw data"):
    st.write(df.head())

# Handle Missing Values
imputer = SimpleImputer(strategy="median")
df.iloc[:, :] = imputer.fit_transform(df)

# Exploratory Data Analysis (EDA)
st.subheader("Data Visualization")

# Correlation Heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig)

# Box Plot
st.write("Box Plot of Features")
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(kind="box", subplots=True, layout=(5, 4), figsize=(15, 10), patch_artist=True)
st.pyplot(fig)

# Splitting Data
st.subheader("Feature Selection & Model Training")

target_column = st.text_input("Enter the target column name:", "target")  # Modify if different

if target_column not in df.columns:
    st.error(f"Column '{target_column}' not found!")
else:
    X = df.drop(columns=[target_column])  # Feature variables
    y = df[target_column]  # Target variable

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save feature names
    feature_names = X.columns.tolist()

    # Model Training
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        st.write(f"**{name} Accuracy:** {accuracy:.4f}")

    # Save the best model
    best_model_name = max(results, key=results.get)
    joblib.dump(models[best_model_name], "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.success(f"Best model ({best_model_name}) saved successfully!")

    # Prediction Function
    def predict_disease(input_data):
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        return "Disease Present" if prediction[0] == 1 else "No Disease"

    # Deployment: User Input Fields
    st.subheader("Make a Prediction")
    user_inputs = {feature: st.number_input(f"{feature}", value=0.0) for feature in feature_names}
    input_data = list(user_inputs.values())

    if st.button("Predict"):
        result = predict_disease(input_data)
        st.write(f"### **Prediction: {result}**")

# Instructions for running the Streamlit app
st.subheader("How to Run the App")
st.write("1. Save this file as `heart_disease_app.py`")
st.write("2. Open a terminal and run:")
st.code("streamlit run heart_disease_app.py")
