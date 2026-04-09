import pandas as pd
import os
from src.feature_builder import build_features_and_split

# Define paths for raw and processed data
SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'
OUTPUT_DATA_DIR = 'data/processed'
TRAIN_SET_PATH = os.path.join(OUTPUT_DATA_DIR, 'train_set.csv')
TEST_SET_PATH = os.path.join(OUTPUT_DATA_DIR, 'test_set.csv')
MERGED_DAILY_FEATURES_PATH = os.path.join(OUTPUT_DATA_DIR, 'merged_daily_features.csv')

def main():
    """
    Orchestrates the entire data processing and machine learning pipeline.
    Includes data loading, feature engineering, train/test split, and
    placeholders for model training and evaluation.
    """
    print("--- Starting Main Data Processing Pipeline ---")

    train_df = None
    test_df = None

    # Implement "Development Environment Reset" logic
    # If processed files exist, skip preprocessing and load directly
    if os.path.exists(TRAIN_SET_PATH) and os.path.exists(TEST_SET_PATH):
        print(f"Processed files found in '{OUTPUT_DATA_DIR}'. Loading existing train and test sets.")
        train_df = pd.read_csv(TRAIN_SET_PATH)
        test_df = pd.read_csv(TEST_SET_PATH)
        # Ensure 'date' column is datetime if needed for further processing
        train_df['date'] = pd.to_datetime(train_df['date'])
        test_df['date'] = pd.to_datetime(test_df['date'])
    else:
        print("Processed files not found. Starting feature building and splitting.")
        train_df, test_df = build_features_and_split(
            SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH, OUTPUT_DATA_DIR
        )
    
    print("\n--- Data Preparation Complete ---")
    print(f"Train set loaded with shape: {train_df.shape}")
    print(f"Test set loaded with shape: {test_df.shape}")

    # --- Phase 3: Machine Learning with XGBoost ---
    print("\n--- Phase 3: Machine Learning with XGBoost (Placeholder) ---")
    print("TODO: Implement XGBoost model training, hyperparameter tuning, and evaluation here.")
    print("      Refer to GEMINI.md 'Phase 3' and 'Key Execution Tasks' for details.")

    # Example: Accessing features and labels
    # X_train = train_df.drop('death_count', axis=1)
    # y_train = train_df['death_count']
    # X_test = test_df.drop('death_count', axis=1)
    # y_test = test_df['death_count']

    # TODO:
    # - Train XGBoost model
    # - Perform hyperparameter tuning (e.g., using Optuna)
    # - Evaluate model using MAE, RMSE, R-squared
    # - Generate SHAP summary plots for explainability

    print("\n--- Main Data Processing Pipeline Finished ---")

if __name__ == '__main__':
    main()
