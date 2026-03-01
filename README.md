# ImpulseIQ – Behavioural Intelligence Engine for Financial Impulse Detection

## Project Overview

ImpulseIQ is a behavioural analytics system designed to detect and predict impulsive spending behaviour in young adults.

The system transforms transactional data into behavioural risk indicators and generates:

- Impulse Probability (0–1)
- Risk Score (0–100)
- Risk Tier (Low / Moderate / High / Critical)
- Behavioural Risk Signals
- Actionable Financial Nudges

This solution was developed as part of a Behavioural Analytics Hackathon.

---

## Problem Statement

Young individuals frequently engage in impulsive spending due to:

- Emotional triggers
- Salary-day spending spikes
- Late-night purchases
- Spending bursts
- Financial stress conditions

The objective was to design a system that:

- Detects impulsive spending behaviour
- Identifies behavioural and contextual triggers
- Predicts high-risk spending patterns
- Generates personalized behavioural interventions

---

## Dataset

### Dataset Type: Synthetic

There is no publicly available dataset that labels financial transactions as “impulsive behaviour.” Therefore, we developed a behavioural finance simulation engine.

### Why Synthetic?

Real-world financial datasets:
- Do not contain impulse labels
- Lack behavioural vulnerability indicators
- Do not model psychological drivers

To overcome this, we simulated realistic behavioural patterns.

---

### Dataset Specifications

- ~400 simulated users
- ~25,000+ transactions
- 60-day simulation period
- Monthly salary cycles
- Financial stress modelling
- Behavioural momentum modelling

Each user was assigned behavioural parameters:

- Impulse sensitivity
- Emotional volatility
- Salary-day excitement factor
- Night purchase bias
- Financial discipline score

Transactions were generated probabilistically using these parameters.

---

## Feature Engineering

Raw transactions were converted into behavioural indicators:

| Feature | Description |
|----------|-------------|
| Spend-to-Salary Ratio | Transaction intensity relative to income |
| Balance Stress Indicator | Financial vulnerability (balance < 10% salary) |
| Recent Impulse Rate | Behavioural momentum |
| Spend Burst Signal | Abnormal spending spike detection |
| Weekend Effect | Context-based behavioural modifier |
| Prior Impulse Momentum | Historical impulse persistence |

These engineered features transformed transactional data into behavioural intelligence signals.

---

## Model Development

### Models Evaluated

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Why XGBoost?

XGBoost was selected as the final model because:

- It captured nonlinear behavioural relationships effectively
- It handled feature interactions between stress, timing, and momentum
- It achieved superior ROC-AUC and F1-score compared to other models
- It demonstrated strong recall for high-risk impulse detection

---

## Model Performance

(Replace with your actual values)

| Metric | XGBoost |
|--------|---------|
| Accuracy | 98% |
| Precision | 91.27% |
| Recall | 87.42% |
| F1 Score | 0.8930 |
| ROC-AUC | 0.9895 |

The model was optimized to maintain strong recall to prevent under-detection of high-risk impulsive behaviour.

---

## Risk Scoring System

Impulse probability is converted into:


Risk Score = Probability × 100


Risk tiers:

- 0–40 → Low
- 40–60 → Moderate
- 60–75 → High
- 75–100 → Critical

This enables structured behavioural risk segmentation.

---

## Deployment

The trained XGBoost model was deployed using a Flask-based local web application.

### Architecture

User Input → Feature Engineering → XGBoost Model →  
Impulse Probability → Risk Score → Risk Tier → Behavioural Output

### Running the Application

1. Install dependencies:


pip install flask pandas numpy scikit-learn xgboost joblib


2. Ensure `model.pkl` (trained XGBoost model) is in the project folder.

3. Run:


python app.py


4. Open browser:


http://127.0.0.1:5000


---

## Behavioural Insights

Key behavioural patterns identified:

- High spend-to-salary ratio significantly increases impulse probability.
- Financial stress amplifies behavioural vulnerability.
- Recent impulse behaviour strongly predicts future impulsive spending.
- Spending bursts are strong early warning signals.
- Contextual timing (weekends, salary cycle) influences risk levels.

---

## Practical Impact

ImpulseIQ can be integrated into:

- Digital banking systems
- Fintech platforms
- Personal finance management apps
- Credit risk monitoring systems

Potential benefits:

- Real-time impulse detection
- Behavioural nudges
- Financial stress prevention
- Improved savings behaviour

---

## Future Scope

- Integration with real-world banking transaction data
- Real-time streaming transaction monitoring
- Reinforcement learning for adaptive nudges
- Mobile application deployment
- Advanced behavioural segmentation models

---

## 🏁 Conclusion

ImpulseIQ demonstrates how behavioural modelling and machine learning can be combined to:

- Quantify financial impulse behaviour
- Predict high-risk spending patterns
- Deliver explainable risk scoring
- Provide actionable behavioural interventions

This project bridges behavioural psychology and financial analytics using advanced ML techniques.

---
