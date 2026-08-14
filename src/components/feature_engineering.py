import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def get_data_transformer_object() -> ColumnTransformer:
    
    numerical_columns = ["age", "income", "credit_score"]
    categorical_columns = ["employment_type"]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ("scaler", StandardScaler(with_mean=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipeline", cat_pipeline, categorical_columns)
        ]
    )

    return preprocessor

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    preprocessor = get_data_transformer_object()
    X_train_transformed = preprocessor.fit_transform(X_train)

    print("Data Transformation Successful!")
    print(f"Transformed features shape: {X_train_transformed.shape}")