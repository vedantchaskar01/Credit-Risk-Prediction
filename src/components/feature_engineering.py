import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def get_data_transformer_object() -> ColumnTransformer:
    """
    Creates and returns a pre-processing pipeline for numerical and categorical features.
    """
    # Define column groups
    numerical_columns = ["age", "income", "credit_score"]
    categorical_columns = ["employment_type"]

    # Pipeline for numerical features: Fill missing values with median -> Scale values
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline for categorical features: Fill missing values with most frequent -> One-Hot Encode
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ("scaler", StandardScaler(with_mean=False))
        ]
    )

    # Combine numerical and categorical pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipeline", cat_pipeline, categorical_columns)
        ]
    )

    return preprocessor

if __name__ == "__main__":
    # Test our transformation pipeline on train data
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Separate features (X) and target (y)
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    preprocessor = get_data_transformer_object()
    X_train_transformed = preprocessor.fit_transform(X_train)

    print("Data Transformation Successful!")
    print(f"Transformed features shape: {X_train_transformed.shape}")