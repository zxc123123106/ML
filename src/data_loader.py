import pandas as pd

def align_data(sensor_data_path, pond_daily_logs_path, min_records_per_day=10):
    """
    Loads sensor data and pond daily logs, aligns them by date and pond_id,
    handles time zones, extracts dates, and performs data integrity checks.

    Args:
        sensor_data_path (str): Path to the sensor data CSV file.
        pond_daily_logs_path (str): Path to the pond daily logs CSV file.
        min_records_per_day (int): Minimum number of sensor records required per day per pond.

    Returns:
        tuple: (aligned_sensor_data_df, pond_daily_logs_df)
            aligned_sensor_data_df (pd.DataFrame): Processed sensor data.
            pond_daily_logs_df (pd.DataFrame): Processed pond daily logs.
    """
    # 1.1 Load Data
    sensor_data = pd.read_csv(sensor_data_path)
    pond_daily_logs = pd.read_csv(pond_daily_logs_path)

    # Convert to datetime
    sensor_data['created_at'] = pd.to_datetime(sensor_data['created_at'])
    pond_daily_logs['log_date'] = pd.to_datetime(pond_daily_logs['log_date'])

    # 1.2 Time Zone Handling & Date Extraction
    # Convert to Taiwan time zone (+0800)
    # Ensure 'created_at' is timezone-aware before converting
    if sensor_data['created_at'].dt.tz is None:
        # Assuming UTC if no timezone info, adjust as necessary based on source
        sensor_data['created_at'] = sensor_data['created_at'].dt.tz_localize('UTC')
    sensor_data['created_at'] = sensor_data['created_at'].dt.tz_convert('Asia/Taipei') # Taiwan uses Asia/Taipei timezone

    # Extract date for alignment
    sensor_data['date'] = sensor_data['created_at'].dt.date
    # Convert date column to datetime objects for consistent merging later
    sensor_data['date'] = pd.to_datetime(sensor_data['date'])

    # 1.3 Data Integrity Check
    # Keep only records where status == 'complete'
    sensor_data = sensor_data[sensor_data['status'] == 'complete']

    # Ensure minimum number of sensor records per day per pond
    record_counts = sensor_data.groupby(['pond_id', 'date']).size().reset_index(name='record_count')
    valid_days = record_counts[record_counts['record_count'] >= min_records_per_day]

    # Merge to filter sensor_data
    sensor_data = pd.merge(
        sensor_data,
        valid_days[['pond_id', 'date']],
        on=['pond_id', 'date'],
        how='inner'
    )

    return sensor_data, pond_daily_logs

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
    # Adjust these paths to your actual file locations
    SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
    POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'

    print(f"Loading sensor data from: {SENSOR_DATA_PATH}")
    print(f"Loading pond daily logs from: {POND_DAILY_LOGS_PATH}")

    processed_sensor_data, processed_pond_daily_logs = align_data(
        SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH
    )
    print("\nProcessed Sensor Data Head after alignment:")
    print(processed_sensor_data.head())

    print("\nImputing missing sensor values...")
    processed_sensor_data = impute_missing_sensor_values(processed_sensor_data.copy())
    print("Imputation complete.")
    print("NaNs after imputation (should be significantly reduced or 0 for specified columns):")
    print(processed_sensor_data[['ph', 'turbidity', 'mq135', 'mq137', 'temp_deep', 'temp_shallow']].isnull().sum())

    print("\nHandling outliers...")
    processed_sensor_data = handle_outliers(processed_sensor_data.copy())
    print("Outlier handling complete.")
    print("Sample data after outlier handling for 'ph' and 'temp_deep':")
    print(processed_sensor_data[['pond_id', 'created_at', 'ph', 'temp_deep']].head())

    print("\nFull cleaned sensor data info:")
    processed_sensor_data.info()
    print("\nFull cleaned sensor data head:")
    print(processed_sensor_data.head())