# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing components
model = joblib.load('models_files/logistic_regression_model.pkl')
label_encoders = joblib.load('models_files/label_encoders.pkl')
preprocessor = joblib.load('models_files/onehot_preprocessor.pkl')
scaler = joblib.load('models_files/scaler.pkl')  # NEW: Scaler

# Binary and nominal columns
binary_cols = ['hypertension', 'heart_disease', 'diabetes']
nominal_cols = ['gender', 'smoking_history']

# Streamlit app title
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Fill out the form to predict the likelihood of diabetes based on medical indicators.")

#  User Input Form 
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
smoking_history = st.selectbox("Smoking History", ["never", "No Info", "former", "current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Prepare input as DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level],
    'diabetes': ['No']  # Needed for alignment with training columns
})

# Predict button
if st.button("Predict"):
    # Step 1: Label encode binary columns
    for col in binary_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Step 2: One-hot encode nominal features using saved preprocessor
    X_encoded = preprocessor.transform(input_data)

    # Step 3: Scale features using saved scaler
    X_scaled = scaler.transform(X_encoded)

    # Step 4: Predict
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0][1]

    # Step 5: Format and show result
    prediction_message = "Yes (Likely to Have Diabetes)" if pred_class == 1 else "No (Unlikely to Have Diabetes)"
    
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {prediction_message}")
    st.write(f"**Probability of Diabetes (Class 1):** {pred_proba:.4f}")
