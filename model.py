import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import os

os.makedirs("static", exist_ok=True)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open("model.pkl", "wb"))

# Feature importance
importance = model.coef_[0]

plt.figure()
plt.barh(X.columns, importance, color="#00c6ff")
plt.title("Key Factors Affecting Churn")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

print("✅ Model ready")