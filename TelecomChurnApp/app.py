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

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

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
    
    return render_template('result.html', prediction=int(prediction), probability=probability, customer_data=data)

if __name__ == '__main__':
    app.run(debug=True)
