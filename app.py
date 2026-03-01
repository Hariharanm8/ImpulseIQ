from feature_utils import compute_features
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

def create_features(transaction_amount,
                    monthly_salary,
                    balance_after,
                    is_weekend,
                    recent_impulse_count,
                    total_count):

    spend_to_salary = transaction_amount / monthly_salary
    balance_stress = 1 if balance_after < 0.1 * monthly_salary else 0
    impulse_rate = recent_impulse_count / total_count

    return np.array([[spend_to_salary,
                      balance_stress,
                      impulse_rate,
                      is_weekend]])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        transaction_amount = float(request.form["transaction_amount"])
        monthly_salary = float(request.form["monthly_salary"])
        balance_after = float(request.form["balance_after"])
        is_weekend = int(request.form.get("is_weekend", 0))
        recent_impulse = float(request.form["recent_impulse"])
        total_txn = float(request.form["total_txn"])

        features = compute_features(transaction_amount,
                                    monthly_salary,
                                    balance_after,
                                    is_weekend,
                                    recent_impulse,
                                    total_txn)

        prob = model.predict_proba(features)[0][1]
        risk_score = prob * 100

        if risk_score > 75:
            tier = "Critical"
        elif risk_score > 60:
            tier = "High"
        elif risk_score > 40:
            tier = "Moderate"
        else:
            tier = "Low"

        prediction = {
            "prob": round(prob, 3),
            "risk_score": round(risk_score, 1),
            "tier": tier
        }

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)