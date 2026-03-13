from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load artifacts
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    train_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

import json

@app.route('/insights')
def insights():
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None

    # Load model comparison data
    comparison_models = []
    try:
        import csv
        with open('model_comparison.csv', 'r') as f:
            reader = csv.DictReader(f)
            comparison_models = list(reader)
    except FileNotFoundError:
        pass

    return render_template('insights.html', metrics=metrics, comparison_models=comparison_models)

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

@app.route('/settings')
def settings():
    # Placeholder returning a simple styled error page or redirect
    return """
    <div style="font-family: sans-serif; height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #0f111a; color: white;">
        <h2 style="color: #5c6bc0;">⚙️ Model Configuration</h2>
        <p style="color: #94a3b8; max-width: 400px; text-align: center;">Settings module is currently under development. To retrain the ML model, use the provided Jupyter Notebook.</p>
        <a href="/" style="margin-top: 20px; padding: 10px 20px; background: #5c6bc0; color: white; text-decoration: none; border-radius: 8px;">Return to Predictor</a>
    </div>
    """

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = request.form.to_dict()
    
    # Convert numeric fields
    numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for field in numeric_fields:
        if field in data and data[field]:
            try:
                data[field] = float(data[field])
            except ValueError:
                data[field] = 0.0
        else:
            data[field] = 0.0
            
    # Convert Yes/No strings exactly as they appeared in the dataset
    # E.g. SeniorCitizen is 0/1 in dataset
    if 'SeniorCitizen' in data:
        data['SeniorCitizen'] = int(data['SeniorCitizen'])
    else:
        data['SeniorCitizen'] = 0

    # Create DataFrame (1 row)
    input_df = pd.DataFrame([data])
    
    # Apply get_dummies
    input_df = pd.get_dummies(input_df)
    
    # Align columns with training data using reindex
    input_df = input_df.reindex(columns=train_columns, fill_value=0)
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    try:
        probability = float(model.predict_proba(input_scaled)[0][1])
    except AttributeError:
        # Fallback if model doesn't support predict_proba
        probability = 0.85 if prediction == 1 else 0.15

    # Risk segmentation based on churn probability
    if probability >= 0.7:
        risk_level = "High Risk"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return render_template('result.html', prediction=int(prediction), probability=probability,
                           risk_level=risk_level, customer_data=data)

if __name__ == '__main__':
    app.run(debug=True)
