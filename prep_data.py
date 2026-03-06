import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0})

# Get Dummies
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("------------------------\n")

# Ensure static directory exists
os.makedirs('TelecomChurnApp/static', exist_ok=True)

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('TelecomChurnApp/static/confusion_matrix.png', dpi=300)
plt.close()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('TelecomChurnApp/static/roc_curve.png', dpi=300)
plt.close()

# Save artifacts
with open("TelecomChurnApp/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("TelecomChurnApp/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("TelecomChurnApp/columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

metrics_dict = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1
}

with open("TelecomChurnApp/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("Exported churn_model.pkl, scaler.pkl, columns.pkl, and metrics.json to TelecomChurnApp")
print("Exported confusion_matrix.png and roc_curve.png to TelecomChurnApp/static")
