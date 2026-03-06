# Telecom Customer Churn Prediction System

## Project Overview

Customer churn is a major challenge for telecom service providers, as losing customers directly affects revenue and business growth. This project aims to analyze customer data and predict whether a customer is likely to discontinue telecom services.

Using the UCI Telecom Customer Churn Dataset, a machine learning model is trained to identify patterns associated with customer churn. The trained model is then integrated into a web-based application where users can enter customer details and receive churn predictions in real time.

## Problem Statement

Telecom service providers face high customer churn due to intense competition and changing customer needs. Identifying the factors that lead to customer churn is important for developing effective retention strategies.

This project analyzes telecom customer data to determine the key drivers of churn and builds a predictive system that helps identify customers who are likely to leave the service.

## Objectives

- Analyze telecom customer data to understand churn behavior
- Identify important factors contributing to customer churn
- Train a machine learning model to predict churn
- Develop a web application for real-time churn prediction
- Support telecom companies in making data-driven retention strategies

## Key Features

- **Modern Dashboard UI:** Built with premium glassmorphism aesthetics, responsive sidebars, and interactive KPIs.
- **Real-Time Prediction:** Instantly predicts churn probability using user-inputted customer data.
- **Insights Overview:** Displays comprehensive Exploratory Data Analysis (EDA) findings to visualize churn patterns.
- **Customer Segmentation:** Categorizes customers based on churn risk levels to enable targeted marketing or retention actions.

## Dataset

**Dataset used in this project:**  
[UCI Machine Learning Repository – Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains information about telecom customers including:
- Demographic details
- Account information
- Service subscriptions
- Billing details
- Churn status

### Key Attributes
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- **Churn (Target Variable)**

**Total Records:** 7043 customers

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Pickle
- **Web Framework:** Flask
- **Frontend:** HTML, CSS

## Machine Learning Model

The model used in this project is: **Logistic Regression**

Logistic Regression is suitable for binary classification problems where the output variable has two possible values.

In this project, the model predicts:
- `1` → Customer will churn
- `0` → Customer will not churn

### Model Workflow

1. Data Cleaning
2. Feature Encoding
3. Train-Test Split
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Model Saving using Pickle

**Model Accuracy Achieved:** ~82%

## Project Structure

```text
TelecomChurnApp
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── columns.pkl
├── README.md
│
├── templates
│   ├── index.html
│   ├── result.html
│   ├── insights.html
│   └── segmentation.html
│
└── dataset
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Application Workflow

1. User enters customer information in the web interface.
2. The application sends the data to the Flask backend.
3. Input data is preprocessed and scaled.
4. The trained machine learning model predicts churn probability.
5. The prediction result is displayed to the user.

## How to Run the Project

**Step 1: Install Required Libraries**
```bash
pip install flask pandas numpy scikit-learn
```

**Step 2: Navigate to Project Folder**
```bash
cd TelecomChurnApp
```

**Step 3: Run the Flask Application**
```bash
python app.py
```

**Step 4: Open in Browser**  
Navigate to `http://127.0.0.1:5000/`

## Expected Output

The web application predicts whether a telecom customer is likely to churn based on the provided input data.

**Example Output:**
- Customer is likely to churn
*or*
- Customer is not likely to churn

## Business Impact

This system helps telecom companies:
- Identify customers at risk of leaving
- Design targeted retention strategies
- Improve customer satisfaction
- Reduce revenue loss due to churn

## Future Improvements

- Use advanced models such as Random Forest or XGBoost
- Deploy the application on cloud platforms
- Add customer segmentation features
- Include visualization dashboards

## Conclusion

This project demonstrates how machine learning can be applied to analyze telecom customer behavior and predict churn. By integrating predictive models into a web application, organizations can make informed decisions to improve customer retention and business performance.
