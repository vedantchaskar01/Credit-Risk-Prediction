import os
import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from feature_engineering import get_data_transformer_object

train_df = pd.read_csv("data/processed/train.csv")

X_train = train_df.drop(columns=["default"])
y_train = train_df["default"]

preprocessor = get_data_transformer_object()

xgb_model = XGBClassifier(
    scale_pos_weight=19,
    random_state=42, 
    max_depth=3,        
    learning_rate=0.1
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

print("Training model pipeline...")
model_pipeline.fit(X_train, y_train)
print("Training Complete!")

test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.drop(columns=["default"])
y_test = test_df["default"]

predictions = model_pipeline.predict(X_test)
probabilities = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

with mlflow.start_run():

    mlflow.log_param("scale_pos_weight", 19)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("learning_rate", 0.1)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model ROC-AUC: {roc_auc:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model_pipeline, "models/model_pipeline.pkl")
print("Pipeline saved to models/model_pipeline.pkl!")
