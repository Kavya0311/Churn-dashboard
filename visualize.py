import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("static", exist_ok=True)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)

# Churn distribution
plt.figure()
df["Churn"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Customers: Stayed vs Left")
plt.xticks([0,1], ["Stayed", "Left"], rotation=0)
plt.savefig("static/churn.png")
plt.close()

# Charges
plt.figure()
df.boxplot(column="MonthlyCharges", by="Churn")
plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.savefig("static/charges.png")
plt.close()

# Tenure
plt.figure()
df.boxplot(column="tenure", by="Churn")
plt.title("Tenure vs Churn")
plt.suptitle("")
plt.savefig("static/tenure.png")
plt.close()

print("Graphs ready")