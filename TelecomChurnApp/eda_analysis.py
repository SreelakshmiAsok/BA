import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def run_eda():
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Load data
    data_path = '../WA_Fn-UseC_-Telco-Customer-Churn.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        return
        
    # Preprocessing
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    df = df.dropna()
    
    # Set seaborn style for nicer plots
    sns.set_theme(style="whitegrid")
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/churn_distribution.png', dpi=300)
    plt.close()
    
    # 2. Churn vs Contract
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
    plt.title('Churn by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/churn_vs_contract.png', dpi=300)
    plt.close()
    
    # 3. Churn vs Tenure
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
    plt.title('Churn vs Tenure')
    plt.xlabel('Churn')
    plt.ylabel('Tenure (Months)')
    plt.tight_layout()
    plt.savefig('static/churn_vs_tenure.png', dpi=300)
    plt.close()
    
    # 4. Churn vs MonthlyCharges
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
    plt.title('Churn vs Monthly Charges')
    plt.xlabel('Churn')
    plt.ylabel('Monthly Charges')
    plt.tight_layout()
    plt.savefig('static/churn_vs_monthlycharges.png', dpi=300)
    plt.close()
    
    # 5. Churn vs InternetService
    plt.figure(figsize=(8, 5))
    sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set2')
    plt.title('Churn vs Internet Service')
    plt.xlabel('Internet Service')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/churn_vs_internetservice.png', dpi=300)
    plt.close()
    
    # 6. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # Add dummy variable for Churn to see correlations with churn
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
        numeric_df['Churn_Yes'] = (df['Churn'] == 'Yes').astype(int)
        
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('static/correlation_heatmap.png', dpi=300)
    plt.close()

    print("EDA completed successfully. All 6 plots saved to the static/ directory.")

if __name__ == '__main__':
    run_eda()
