import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

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

# Accuracy
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save artifacts
with open("TelecomChurnApp/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("TelecomChurnApp/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("TelecomChurnApp/columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Exported churn_model.pkl, scaler.pkl, and columns.pkl to TelecomChurnApp")
