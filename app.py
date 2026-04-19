from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
from knowledge_base import knowledge

app = Flask(__name__)

# -----------------------------
# Ensure static graphs exist
# -----------------------------
if not os.path.exists("static/churn.png"):
    os.system("python visualize.py")

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# Home route
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# Prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        tenure = float(request.form['tenure'])
        monthly = float(request.form['MonthlyCharges'])
        total = float(request.form['TotalCharges'])

        # Create dataframe
        df = pd.DataFrame([[tenure, monthly, total]],
                          columns=["tenure", "MonthlyCharges", "TotalCharges"])

        # Prediction probability
        proba = model.predict_proba(df)[0][1]
        confidence = round(proba * 100, 2)

        # Final prediction
        result = "High Risk Customer" if proba > 0.5 else "Low Risk Customer"

        # -----------------------------
        # Explainable AI (RAG-style)
        # -----------------------------
        reasons = []

        if tenure < 6:
            reasons.append(knowledge["low_tenure"])

        if monthly > 80:
            reasons.append(knowledge["high_charges"])

        if total < 500:
            reasons.append(knowledge["low_total"])

        if tenure > 40:
            reasons.append(knowledge["high_tenure"])

        explanation = " | ".join(reasons) if reasons else "No strong risk indicators."

        # -----------------------------
        # Business Insight
        # -----------------------------
        if proba > 0.7:
            insight = "⚠️ Immediate retention action required."
        elif proba > 0.4:
            insight = "⚡ Moderate risk – consider engagement offers."
        else:
            insight = "✅ Customer is stable and loyal."

        return render_template(
            "index.html",
            prediction_text=result,
            explanation_text=explanation,
            confidence=confidence,
            insight=insight
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)