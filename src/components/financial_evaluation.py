import pandas as pd
import joblib
import numpy as np

test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.drop(columns=["default"])
y_test = test_df["default"].values
model_pipeline = joblib.load("models/model_pipeline.pkl")
probabilities = model_pipeline.predict_proba(X_test)[:, 1]
REVENUE_PER_GOOD_LOAN = 5000    
LOSS_PER_DEFAULT = -50000      
MISSED_OPPORTUNITY = -5000
CORRECT_REJECTION = 0
print("Setup complete! Ready to test thresholds.")

def calculate_profit(probabilities, y_true, threshold):
    custom_predictions = (probabilities >= threshold).astype(int)
    
    total_profit = 0
    
    for i in range(len(y_true)):
        actual = y_true[i]
        predicted = custom_predictions[i]
        
        if actual == 0 and predicted == 0:
            total_profit += REVENUE_PER_GOOD_LOAN
        elif actual == 1 and predicted == 0:
            total_profit += LOSS_PER_DEFAULT
        elif actual == 0 and predicted == 1:
            total_profit += MISSED_OPPORTUNITY
        elif actual == 1 and predicted == 1: 
            total_profit += CORRECT_REJECTION
            
    return total_profit

profit_at_50 = calculate_profit(probabilities, y_test, threshold=0.50)
profit_at_15 = calculate_profit(probabilities, y_test, threshold=0.15)

print(f"Bank Profit using 50% Threshold: ${profit_at_50:,}")
print(f"Bank Profit using 15% Threshold: ${profit_at_15:,}")

best_profit = -float('inf')
best_threshold = 0.5

for t in np.arange(0.0, 1.0, 0.01):
    profit = calculate_profit(probabilities, y_test, threshold=t)
    if profit > best_profit:
        best_profit = profit
        best_threshold = t

print(f"\n--- The Goldilocks Threshold ---")
print(f"Optimal Threshold: {best_threshold:.2f}")
print(f"Maximum Profit: ${best_profit:,}")

