import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    """Stores configuration paths for data ingestion output artifacts."""
    def __init__(self):
        self.raw_data_path: str = os.path.join("data", "raw", "data.csv")
        self.train_data_path: str = os.path.join("data", "processed", "train.csv")
        self.test_data_path: str = os.path.join("data", "processed", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generates realistic credit risk synthetic dataset with missing values & class imbalance."""
        np.random.seed(42)
        n_samples = 1000

        income = np.random.normal(50000, 15000, n_samples)
        credit_score = np.random.randint(300, 850, n_samples)
        age = np.random.randint(21, 65, n_samples)
        employment_type = np.random.choice(["Salaried", "Self-Employed", "Unemployed"], size=n_samples, p=[0.6, 0.3, 0.1])
        
        # Target: 1 = Default (5%), 0 = Non-Default (95%)
        default = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

        df = pd.DataFrame({
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "employment_type": employment_type,
            "default": default
        })

        # Injecting missing values randomly to simulate messy real-world data
        df.loc[df.sample(frac=0.05, random_state=42).index, "income"] = np.nan
        return df

    def initiate_data_ingestion(self):
        print("Starting Data Ingestion Process...")
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

        # Step 1: Fetch/Generate raw dataset
        df = self.generate_synthetic_data()
        df.to_csv(self.ingestion_config.raw_data_path, index=False)
        print(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

        # Step 2: Stratified Train-Test Split (preserves class imbalance ratio)
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["default"])

        # Step 3: Save processed split files
        train_set.to_csv(self.ingestion_config.train_data_path, index=False)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False)
        print(f"Train and Test sets saved at: {self.ingestion_config.train_data_path} & {self.ingestion_config.test_data_path}")

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()