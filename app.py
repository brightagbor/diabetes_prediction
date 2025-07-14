# app.py
import streamlit as st
import pandas as pd
import joblib

# Load saved models
model = joblib.load('models_files/logistic_regression_model.pkl')
label_encoders = joblib.load('models_files/label_encoders.pkl')
preprocessor = joblib.load('models_files/onehot_preprocessor.pkl')
scaler = joblib.load('models_files/scaler.pkl')

# App title
st.title("Diabetes Risk Prediction")
st.write("Enter your information to predict diabetes risk")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
smoking_history = st.selectbox("Smoking History", ["never", "No Info", "former", "current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Predict button
if st.button("Predict"):
    # Create input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'age': [age],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'diabetes': ['No']  # Placeholder
    })

    try:
        # Process the data same as training
        # Scale numeric features first
        numeric_data = input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
        scaled_numeric = scaler.transform(numeric_data)
        
        # Put scaled values back
        input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaled_numeric

        # Encode Yes/No columns
        for col in ['hypertension', 'heart_disease', 'diabetes']:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Apply preprocessor (handles categories)
        X_input = input_data.drop('diabetes', axis=1)
        X_processed = preprocessor.transform(X_input)

        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]

        # Show results
        st.subheader("Results")
        if prediction == 1:
            st.error("High Risk: Likely to have diabetes")
        else:
            st.success("Low Risk: Unlikely to have diabetes")
        
        st.write(f"**Probability of diabetes:** {probability:.1%}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Please check your input values and try again.")