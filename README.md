# ImpulseIQ — Behavioural Intelligence Engine for Financial Impulse Detection

##  Overview

ImpulseIQ is a behavioural analytics system designed to detect and predict impulsive spending behaviour in young adults.

The system transforms raw transaction data into behavioural risk indicators and generates:

- Impulse Probability (0–1)
- Risk Score (0–100)
- Risk Tier (Low / Moderate / High / Critical)
- Behavioural Trigger Explanation
- Personalized Financial Nudges

This project was developed for a Behavioural Analytics Hackathon.

---

##  Problem Statement

Young individuals often overspend due to:

- Emotional triggers  
- Salary-day spikes  
- Late-night purchases  
- Spending bursts  
- Financial stress  

The objective was to build a system that:

- Detects impulsive spending patterns  
- Identifies behavioural triggers  
- Predicts high-risk upcoming spending behaviour  
- Generates personalized behavioural nudges  

---

##  Dataset

### Dataset Type: Synthetic

No publicly available dataset explicitly labels “impulsive spending behaviour.” Therefore, we built a behavioural finance simulation engine.

### Why Synthetic?

Real-world financial datasets:
- Do not label impulse behaviour
- Do not include behavioural vulnerability signals
- Lack psychological context features

Hence, we simulated realistic behavioural finance patterns.

### Dataset Specifications

- ~400 simulated users
- ~25,000+ transactions
- 60-day behavioural simulation
- Monthly salary cycles
- Financial stress modelling
- Behavioural momentum modelling

### Behavioural Parameters Simulated

Each user was assigned:

- Impulse sensitivity
- Emotional volatility
- Salary-day excitement factor
- Night purchase bias
- Financial discipline score

These parameters influenced transaction generation probabilistically.

---

##  Feature Engineering

Raw transactions were converted into behavioural indicators:

| Feature  | Behavioural Meaning |
|----------|--------------------|
| Spend-to-Salary Ratio | Financial intensity of purchase |
| Balance Stress Indicator | Financial vulnerability |
| Recent Impulse Rate | Behavioural momentum |
| Spend Burst Signal | Abnormal spending spike |
| Weekend Effect | Contextual behavioural bias |
| Prior Impulse Momentum | Persistence of behaviour |

These engineered features transform transactional data into behavioural intelligence signals.

---

##  Model Development

### Models Used

- Logistic Regression (baseline, interpretable)
- Random Forest Classifier (final selected model)

### Why Random Forest?

- Captures nonlinear behavioural patterns
- Handles interaction effects between stress, timing, and momentum
- Robust to feature scaling issues

### Training Pipeline

1. Data simulation
2. Feature engineering
3. Train-test split
4. Model training
5. Performance evaluation
6. Risk score generation

---

##  Model Performance

(Replace with your actual values)

- Accuracy: 98%
- Precision: 91.27%
- Recall: 87.42%
- F1 Score: 0.8930
- ROC-AUC: 0.9895

The model was optimized to maintain strong recall to avoid under-detecting high-risk impulse behaviour.

---

##  Risk Scoring System

Impulse probability is converted into:


Risk Score = Probability × 100


Risk tiers:

- 0–40 → Low
- 40–60 → Moderate
- 60–75 → High
- 75–100 → Critical

This enables behavioural risk segmentation.

---

##  Deployment

The trained model was deployed as a locally runnable Flask web application.

### Architecture

User Input → Feature Engineering → ML Model →  
Impulse Probability → Risk Score → Risk Tier → Behavioural Insight

### How to Run the App

1. Install dependencies:

pip install flask pandas numpy scikit-learn joblib


2. Ensure `model.pkl` is in project folder.

3. Run:

python app.py


4. Open browser:

http://127.0.0.1:5000


---

##  Behavioural Insights Derived

Key findings from model analysis:

- High spend-to-salary ratio strongly increases impulse probability.
- Balance stress amplifies vulnerability.
- Recent impulse momentum compounds risk.
- Salary-cycle behaviour significantly influences spending patterns.
- Weekend and late-night transactions increase contextual vulnerability.

---

##  Practical Applications

ImpulseIQ can be integrated into:

- Fintech platforms
- Digital banking systems
- Budgeting apps
- Personal finance advisory tools

Potential benefits:

- Real-time impulse detection
- Behavioural nudges
- Financial stress prevention
- Improved savings behaviour

---

##  Future Scope

- Real banking dataset integration
- Reinforcement learning for adaptive nudges
- Real-time transaction stream processing
- Mobile deployment
- Personalized behavioural profiling

---

##  Conclusion

ImpulseIQ demonstrates how behavioural modelling and machine learning can be combined to:

- Detect financial impulse behaviour
- Quantify behavioural risk
- Deliver actionable interventions
- Bridge behavioural psychology and financial analytics

This project highlights the practical feasibility of behavioural AI systems in fintech environments.

---
