import numpy as np

def compute_features(transaction_amount,
                     monthly_salary,
                     balance_after,
                     is_weekend,
                     recent_impulse_count,
                     total_count):
    """
    Generate feature vector for impulse prediction.
    Must match EXACT feature order used during model training.
    """

    # ---- Basic Derived Features ----
    spend_to_salary_ratio = transaction_amount / monthly_salary
    
    balance_stress_indicator = 1 if balance_after < 0.1 * monthly_salary else 0
    
    recent_impulse_rate = (
        recent_impulse_count / total_count
        if total_count > 0 else 0
    )

    # ---- Spend Burst Approximation ----
    # You can modify if your notebook used z-score
    spend_burst_signal = 1 if spend_to_salary_ratio > 0.25 else 0

    # ---- Weekend Encoding ----
    weekend_effect = 1 if is_weekend else 0

    # ---- Final Feature Vector ----
    features = np.array([[
        spend_to_salary_ratio,
        balance_stress_indicator,
        recent_impulse_rate,
        spend_burst_signal,
        weekend_effect
    ]])

    return features