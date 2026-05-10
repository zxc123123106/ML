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
    Includes data loading, feature engineering, train/test split, and guidance
    for the model training/evaluation entry point.
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
    print("\n--- Phase 3: Machine Learning with XGBoost ---")
    print("Run `python3 -m src.trainer` to train, tune, evaluate, and export model artifacts.")
    print("      Outputs include models/final_best_model.json and reports/figures/*.png.")

    print("\n--- Main Data Processing Pipeline Finished ---")

if __name__ == '__main__':
    main()
