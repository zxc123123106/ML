import pandas as pd
# from src.data_loader import align_data

def impute_missing_sensor_values(df):
    """
    Applies forward fill to specified sensor columns to handle missing values.

    Args:
        df (pd.DataFrame): The input DataFrame containing sensor data.

    Returns:
        pd.DataFrame: The DataFrame with missing sensor values imputed.
    """
    sensor_cols_to_impute = ['ph', 'turbidity', 'mq135', 'mq137', 'temp_deep', 'temp_shallow']
    
    # Ensure the DataFrame is sorted by time for accurate forward fill
    # Assuming 'created_at' is the time column and 'pond_id' to group by each pond
    df = df.sort_values(by=['pond_id', 'created_at'])

    for col in sensor_cols_to_impute:
        if col in df.columns:
            df[col] = df.groupby('pond_id')[col].ffill()
            # Optionally, backfill any remaining NaNs if the first values are missing
            df[col] = df.groupby('pond_id')[col].bfill()
    return df

def handle_outliers(df):
    """
    Handles outliers for specific water quality data columns by converting values
    outside reasonable bounds to NaN and then filling them with linear interpolation.

    Args:
        df (pd.DataFrame): The input DataFrame containing sensor data.

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    # Define bounds for ph and temp_deep
    ph_min, ph_max = 0, 14
    temp_deep_min, temp_deep_max = 10, 40

    # Process 'ph'
    if 'ph' in df.columns:
        df.loc[(df['ph'] < ph_min) | (df['ph'] > ph_max), 'ph'] = None
        df['ph'] = df.groupby('pond_id')['ph'].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

    # Process 'temp_deep'
    if 'temp_deep' in df.columns:
        df.loc[(df['temp_deep'] < temp_deep_min) | (df['temp_deep'] > temp_deep_max), 'temp_deep'] = None
        df['temp_deep'] = df.groupby('pond_id')['temp_deep'].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

    return df

if __name__ == '__main__':
    # Example Usage:
    SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
    POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'

    print("Step 1: Aligning data...")
    sensor_data_aligned, _ = align_data(SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH)
    print("Alignment complete.")

    print("\nStep 2: Imputing missing sensor values...")
    sensor_data_imputed = impute_missing_sensor_values(sensor_data_aligned.copy()) # Use a copy to avoid modifying original
    print("Imputation complete.")
    print("NaNs after imputation (should be significantly reduced or 0 for specified columns):")
    print(sensor_data_imputed[['ph', 'turbidity', 'mq135', 'mq137', 'temp_deep', 'temp_shallow']].isnull().sum())

    print("\nStep 3: Handling outliers...")
    sensor_data_cleaned = handle_outliers(sensor_data_imputed.copy()) # Use a copy
    print("Outlier handling complete.")
    print("Sample data after outlier handling for 'ph' and 'temp_deep':")
    print(sensor_data_cleaned[['pond_id', 'created_at', 'ph', 'temp_deep']].head())

    print("\nFull cleaned sensor data info:")
    sensor_data_cleaned.info()
    print("\nFull cleaned sensor data head:")
    print(sensor_data_cleaned.head())
