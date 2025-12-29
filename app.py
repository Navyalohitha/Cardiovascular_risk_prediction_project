import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI()

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# Load the trained model and scaler
# Ensure these files ('optimal_model.joblib', 'scaler.joblib') exist in the same directory
try:
    model = joblib.load('optimal_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    raise RuntimeError("Model or scaler file not found. Please ensure 'optimal_model.joblib' and 'scaler.joblib' are in the same directory.")

# Define the expected feature order based on X_train (from kernel state)
# This is crucial for consistent input to the scaler and model
# This order includes 'pulse_pressure' because the model was trained with it.
expected_features_order = [
    'age', 'sex', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
    'diabetes', 'totChol', 'BMI', 'heartRate', 'glucose',
    'education_1.0', 'education_2.0', 'education_3.0', 'education_4.0', 'pulse_pressure'
]

# Define Pydantic model for input data
# This model now expects sysBP and diaBP, which will be used to calculate pulse_pressure
class HeartDiseasePredictionInput(BaseModel):
    age: float
    sex: float  # Encoded as 0.0 (Female) or 1.0 (Male)
    cigsPerDay: float
    BPMeds: float  # Encoded as 0.0 (No) or 1.0 (Yes)
    prevalentStroke: float  # Encoded as 0.0 (No) or 1.0 (Yes)
    prevalentHyp: float  # Encoded as 0.0 (No) or 1.0 (Yes)
    diabetes: float  # Encoded as 0.0 (No) or 1.0 (Yes)
    totChol: float
    sysBP: float # Added
    diaBP: float # Added
    BMI: float
    heartRate: float
    glucose: float
    education_1_0: float  # One-hot encoded education level 1.0
    education_2_0: float  # One-hot encoded education level 2.0
    education_3_0: float  # One-hot encoded education level 3.0
    education_4_0: float  # One-hot encoded education level 4.0

# Prediction endpoint
@app.post("/predict")
async def predict_chd(data: HeartDiseasePredictionInput):
    try:
        input_dict = data.dict()

        # Calculate pulse_pressure from sysBP and diaBP
        pulse_pressure = input_dict['sysBP'] - input_dict['diaBP']

        # Create a dictionary that matches the expected_features_order for the model
        model_input_data = {
            'age': input_dict['age'],
            'sex': input_dict['sex'],
            'cigsPerDay': input_dict['cigsPerDay'],
            'BPMeds': input_dict['BPMeds'],
            'prevalentStroke': input_dict['prevalentStroke'],
            'prevalentHyp': input_dict['prevalentHyp'],
            'diabetes': input_dict['diabetes'],
            'totChol': input_dict['totChol'],
            'BMI': input_dict['BMI'],
            'heartRate': input_dict['heartRate'],
            'glucose': input_dict['glucose'],
            'education_1.0': input_dict['education_1_0'], # Map Pydantic field name to model feature name
            'education_2.0': input_dict['education_2_0'],
            'education_3.0': input_dict['education_3_0'],
            'education_4.0': input_dict['education_4_0'],
            'pulse_pressure': pulse_pressure # Use calculated pulse_pressure
        }

        # Convert to DataFrame to ensure correct column order for scaling and prediction
        df_for_prediction = pd.DataFrame([model_input_data], columns=expected_features_order)

        # Scale the input data
        scaled_input = scaler.transform(df_for_prediction)

        # Make prediction
        prediction_class = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1] # Probability of positive class (CHD=1)

        return {
            "predicted_chd": int(prediction_class),
            "probability_chd": float(prediction_proba),
            "message": "Prediction successful."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
