import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.drop(columns=["default"])
y_test = test_df["default"].values

model_pipeline = joblib.load("models/model_pipeline.pkl")
probabilities = model_pipeline.predict_proba(X_test)[:, 1]

OPTIMAL_THRESHOLD = 0.62
custom_predictions = (probabilities >= OPTIMAL_THRESHOLD).astype(int)

cm = confusion_matrix(y_test, custom_predictions)
os.makedirs("reports/figures", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Will Pay", "Default"], 
            yticklabels=["Will Pay", "Default"])
plt.ylabel('Actual Reality')
plt.xlabel('Model Prediction')
plt.title(f'Confusion Matrix (Threshold = {OPTIMAL_THRESHOLD})')
plt.savefig("reports/figures/confusion_matrix.png")
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.xlabel('False Positive Rate (Ringing alarm on burnt toast)')
plt.ylabel('True Positive Rate (Catching real fires)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("reports/figures/roc_curve.png")
plt.close()

print("Evaluation charts saved successfully to reports/figures/ !")
